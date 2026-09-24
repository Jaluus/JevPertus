"""Read the repository's labelled JSONL data as one causal row per question."""

import json
import os
from functools import partial
from typing import NotRequired, TypedDict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

SPECIAL_TOKENS = {
    "context": "<SPECIAL_100>",
    "question": "<SPECIAL_101>",
    "option_start": "<SPECIAL_102>",
    "option_end": "<SPECIAL_103>",
    "decide": "<SPECIAL_104>",
}


class QuestionExample(TypedDict):
    """One encoded question; option metadata indexes the unpadded token row."""

    ids: list[int] | torch.Tensor  # [tokens]; inference uses a 1D long tensor.
    option_ends: list[int]  # Token index of each option's end delimiter.
    decide: int  # Token index of the final decision delimiter.
    label: int | None  # Zero-based option index; None for unlabelled inference.
    keys: list[str | bool | int]  # Answer values in the same order as options.
    qid: NotRequired[str]  # Source question ID, added when loading JSONL data.


class QuestionBatch(TypedDict):
    """Right-padded questions; examples retain their original Python metadata."""

    ids: torch.Tensor  # Long [batch_size, max_tokens], padded with pad_id.
    mask: torch.Tensor  # Bool [batch_size, max_tokens]; True for real tokens.
    examples: list[QuestionExample]  # One example per tensor row, in order.


def collate_questions(
    examples: list[QuestionExample], pad_id: int = 0
) -> QuestionBatch:
    """Pad token rows on CPU, retaining each question's option metadata."""
    rows = [torch.as_tensor(example["ids"], dtype=torch.long) for example in examples]
    ids = pad_sequence(rows, batch_first=True, padding_value=pad_id)
    lengths = torch.tensor([len(row) for row in rows])
    mask = torch.arange(ids.shape[1])[None, :] < lengths[:, None]
    return {"ids": ids, "mask": mask, "examples": examples}


def batch_to_device(batch: QuestionBatch, device: str | torch.device) -> QuestionBatch:
    """Move batch tensors while leaving Python metadata on the CPU."""
    return {**batch, "ids": batch["ids"].to(device), "mask": batch["mask"].to(device)}


def question_options(question):
    """Return option text, answer keys, and an optional training label."""
    kind = question["type"]
    criteria = question.get("criteria") or {}

    if kind == "choice":
        keys = list(criteria)
        options = [f"{key}: {criteria[key]}" if criteria[key] else key for key in keys]
        label = keys.index(question["label"]) if "label" in question else None

    elif kind == "noul":
        keys = [False, True]
        options = [
            f"{key}: {criteria[key]}" if criteria.get(key) else key
            for key in ["false", "true"]
        ]
        label = int(question["label"]) if "label" in question else None

    elif kind == "score":
        options = list(criteria)
        keys = list(range(len(options)))
        label = question.get("label")  # Score labels are already zero-based indices.

    else:
        raise ValueError(f"Unknown question type: {kind}")

    if not options or (label is not None and not 0 <= label < len(options)):
        raise ValueError("Question has no options or an invalid label")
    return options, keys, label


class ContextOverflow(ValueError):
    """A question exceeds the chosen input budget; no content is truncated."""


def encode_question(
    state,
    question,
    tokenizer,
    max_length=1024,
    max_state=384,
) -> QuestionExample:
    """Shared training/inference encoding; inference questions need no label."""
    special = {
        name: tokenizer.convert_tokens_to_ids(token)
        for name, token in SPECIAL_TOKENS.items()
    }
    if tokenizer.unk_token_id in special.values() or len(set(special.values())) != 5:
        raise ValueError(
            "Tokenizer must contain the five distinct Apertus delimiter tokens"
        )
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, sort_keys=True)
    ids = [special["context"]] + tokenizer.encode(state, add_special_tokens=False)
    if len(ids) > max_state:
        raise ContextOverflow("State exceeds max_state")
    options, keys, label = question_options(question)
    ids += [special["question"]]
    instructions = question["instructions"]
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False, sort_keys=True)
    ids += tokenizer.encode(instructions, add_special_tokens=False)
    ends = []
    for option in options:
        ids += [special["option_start"]]
        ids += tokenizer.encode(option, add_special_tokens=False)
        ids += [special["option_end"]]
        ends.append(len(ids) - 1)
    ids += [special["decide"]]
    if len(ids) > max_length:
        raise ContextOverflow("Question exceeds max_length")
    return {
        "ids": ids,
        "option_ends": ends,
        "decide": len(ids) - 1,
        "label": label,
        "keys": keys,
    }


def load_examples(
    path, tokenizer, max_length=1024, max_state=384
) -> list[QuestionExample]:
    """Read labelled questions; report over-limit rows instead of truncating.

    Labels and metadata never enter the model input. Options stay in data order.
    Each row repeats the state, so questions cannot attend to each other.
    """
    examples, skipped = [], 0
    with open(path) as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            for qid, question in record["questions"].items():
                if "label" not in question:
                    raise ValueError(f"Training question {qid} is missing its label")
                try:
                    example = encode_question(
                        record["state"], question, tokenizer, max_length, max_state
                    )
                except ContextOverflow:
                    skipped += 1
                    continue
                if example["label"] is None:
                    raise ValueError(f"Training question {qid} has no label")
                examples.append({**example, "qid": qid})
    print(
        f"{path}: loaded {len(examples)} questions; skipped {skipped} over token limits"
    )
    if not examples:
        raise ValueError(f"No usable questions in {path}")
    return examples


def build_trainloader(
    data_dir,
    tokenizer,
    batch_size=8,
    max_length=1024,
    max_state=384,
    seed=42,
):
    trainset = load_examples(
        os.path.join(data_dir, "train.jsonl"),
        tokenizer,
        max_length,
        max_state,
    )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_questions, pad_id=pad_id),
        generator=torch.Generator().manual_seed(seed),
    )

    return train_loader


def build_testloader(
    data_dir,
    tokenizer,
    batch_size=8,
    max_length=1024,
    max_state=384,
):
    testset = load_examples(
        os.path.join(data_dir, "test.jsonl"),
        tokenizer,
        max_length,
        max_state,
    )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=partial(collate_questions, pad_id=pad_id),
    )

    return test_loader

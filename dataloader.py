"""Read the repository's labelled JSONL data as one causal row per question."""

import json
import os
from functools import partial
from typing import Literal, NotRequired, TypedDict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

SPECIAL_TOKENS = {
    "state": "<SPECIAL_100>",
    "question": "<SPECIAL_101>",
    "option_start": "<SPECIAL_102>",
    "option_end": "<SPECIAL_103>",
    "decide": "<SPECIAL_104>",
}


class Question(TypedDict):
    """Raw training/inference question, before tokenization.

    Choice criteria map answer keys to descriptions; score criteria list options
    in order. Noul criteria optionally describe the "false" and "true" answers.
    Labels are zero-based option indices; noul uses 0 for false and 1 for true.
    Omit the label for inference.
    """

    type: Literal["choice", "score", "noul"]
    state: str  # The State the question is asked in.
    instructions: str  # The actual question text
    criteria: NotRequired[dict[str, str | None] | list[str] | None]
    label: NotRequired[int]


class QuestionExample(TypedDict):
    """One encoded question; option metadata indexes the unpadded token row."""

    ids: torch.Tensor  # [tokens]; inference uses a 1D long tensor.
    option_ends: list[int]  # Token index of each option's end delimiter.
    decide: int  # Token index of the final decision delimiter.
    label: NotRequired[int]  # Zero-based option index; omitted for inference.
    keys: (
        list[str] | list[bool] | list[int]
    )  # Answer values in the same order as options.
    qid: NotRequired[str]  # Source question ID, added when loading JSONL data.


class QuestionBatch(TypedDict):
    """Right-padded questions; examples retain their original Python metadata."""

    ids: torch.Tensor  # Long [batch_size, max_tokens], padded with pad_id.
    mask: torch.Tensor  # Bool [batch_size, max_tokens]; True for real tokens.
    examples: list[QuestionExample]  # One example per tensor row, in order.


def collate_questions(
    examples: list[QuestionExample],
    pad_id: int = 0,
) -> QuestionBatch:
    """Pad token rows on CPU, retaining each question's option metadata."""
    rows = [torch.as_tensor(example["ids"], dtype=torch.long) for example in examples]
    ids = pad_sequence(rows, batch_first=True, padding_value=pad_id)
    lengths = torch.tensor([len(row) for row in rows])
    mask = torch.arange(ids.shape[1])[None, :] < lengths[:, None]
    return {"ids": ids, "mask": mask, "examples": examples}


def batch_to_device(
    batch: QuestionBatch,
    device: str | torch.device,
) -> QuestionBatch:
    """Move batch tensors while leaving Python metadata on the CPU."""
    return {**batch, "ids": batch["ids"].to(device), "mask": batch["mask"].to(device)}


def parse_options(
    question: Question,
) -> tuple[
    list[str],
    list[str] | list[bool] | list[int],
    int | None,
]:
    """Return option text, answer keys, and an optional training label."""
    kind = question["type"]
    criteria = question.get("criteria") or {}

    if kind == "choice":
        keys = list(criteria)
        options = [
            f"{key}: {criteria[key]}" if criteria.get(key) else key for key in keys
        ]

    elif kind == "noul":
        keys = [False, True]
        options = [
            f"{key}: {criteria[key]}" if criteria.get(key) else key
            for key in ["false", "true"]
        ]

    elif kind == "score":
        options = list(criteria)
        keys = list(range(len(options)))

    else:
        raise ValueError(f"Unknown question type: {kind}")

    return options, keys, question.get("label")


class ContextOverflow(ValueError):
    """A question exceeds the chosen input budget; no content is truncated."""


def encode_question(
    question: Question,
    tokenizer,
    device: str | torch.device = "cpu",
) -> QuestionExample:
    """Shared training/inference encoding; inference questions need no label."""

    special = {
        name: tokenizer.convert_tokens_to_ids(token)
        for name, token in SPECIAL_TOKENS.items()
    }

    ids = [special["state"]]
    ids += tokenizer.encode(question["state"], add_special_tokens=False)

    options, keys, label = parse_options(question)

    ids += [special["question"]]
    ids += tokenizer.encode(question["instructions"], add_special_tokens=False)

    option_idxs = []
    for option in options:
        ids += [special["option_start"]]
        ids += tokenizer.encode(option, add_special_tokens=False)
        ids += [special["option_end"]]
        option_idxs.append(len(ids) - 1)

    ids += [special["decide"]]

    ids = torch.as_tensor(ids, dtype=torch.long, device=device)

    example: QuestionExample = {
        "ids": ids,
        "option_ends": option_idxs,
        "decide": len(ids) - 1,
        "keys": keys,
    }
    if label is not None:
        example["label"] = label
    return example


def load_examples(path, tokenizer) -> list[QuestionExample]:
    """Read labelled questions; report over-limit rows instead of truncating.

    Labels and metadata never enter the model input. Options stay in data order.
    Each row repeats the state, so questions cannot attend to each other.
    """
    examples, skipped = [], 0
    for qid, question in load_questions(path):
        example = encode_question(question, tokenizer)
        examples.append({**example, "qid": qid})

    print(
        f"{path}: loaded {len(examples)} questions; skipped {skipped} over token limits"
    )
    return examples


def load_questions(path: str) -> list[Question]:
    """Read raw JSONL questions, converting legacy labels to option indices."""
    questions = []

    with open(path) as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            state = record["state"]

            if not isinstance(state, str):
                state = json.dumps(state, ensure_ascii=False, sort_keys=True)

            for qid, raw in record["questions"].items():

                if "label" not in raw:
                    raise ValueError(f"Training question {qid} is missing its label")

                label = raw["label"]
                instructions = raw["instructions"]
                type_ = raw["type"]

                if type_ == "choice" and isinstance(label, str):
                    label = list(raw["criteria"]).index(label)

                elif type_ == "noul" and isinstance(label, bool):
                    label = int(label)

                if not isinstance(instructions, str):
                    instructions = json.dumps(
                        instructions, ensure_ascii=False, sort_keys=True
                    )

                question: Question = {
                    "type": type_,
                    "state": state,
                    "instructions": instructions,
                    "label": label,
                    "criteria": raw.get("criteria"),
                }

                questions.append(question)
    return questions


def build_trainloader(
    data_dir,
    tokenizer,
    batch_size=8,
    seed=42,
):
    trainset = load_examples(os.path.join(data_dir, "train.jsonl"), tokenizer)

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
):
    testset = load_examples(os.path.join(data_dir, "test.jsonl"), tokenizer)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=partial(collate_questions, pad_id=pad_id),
    )

    return test_loader

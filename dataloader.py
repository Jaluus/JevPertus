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


class EncodedQuestion(TypedDict):
    """One encoded question; option metadata indexes the unpadded token row."""

    ids: torch.Tensor  # [tokens]; inference uses a 1D long tensor.
    option_idxs: list[int]  # Token index of each option's end delimiter.
    decide_idx: int  # Token index of the final decision delimiter.
    label: NotRequired[int]  # Zero-based option index; omitted for inference.


class QuestionBatch(TypedDict):
    """Right-padded questions; examples retain their original Python metadata."""

    ids: torch.Tensor  # Long [batch_size, max_tokens], padded with pad_id.
    mask: torch.Tensor  # Bool [batch_size, max_tokens]; True for real tokens.
    examples: list[EncodedQuestion]  # One example per tensor row, in order.


def collate_questions(
    examples: list[EncodedQuestion],
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


def encode_question(
    question: Question,
    tokenizer,
    device: str | torch.device = "cpu",
) -> EncodedQuestion:
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

    example: EncodedQuestion = {
        "ids": ids,
        "option_idxs": option_idxs,
        "decide_idx": len(ids) - 1,
    }
    if label is not None:
        example["label"] = label
    return example


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

    train_questions = load_questions(os.path.join(data_dir, "train.jsonl"))
    encoded_questions = [
        encode_question(question, tokenizer) for question in train_questions
    ]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    train_loader = DataLoader(
        encoded_questions,
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
    test_questions = load_questions(os.path.join(data_dir, "test.jsonl"))
    encoded_questions = [
        encode_question(question, tokenizer) for question in test_questions
    ]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    test_loader = DataLoader(
        encoded_questions,
        batch_size=batch_size,
        collate_fn=partial(collate_questions, pad_id=pad_id),
    )

    return test_loader

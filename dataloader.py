"""Load, encode, and batch questions for training and inference."""

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
    """Describe a question before tokenization.

    Attributes:
        type: Question kind: "choice", "score", or "noul".
        state: Context in which the question is asked.
        instructions: Question text.
        criteria: Optional answer-key descriptions for choice questions, ordered
            option text for score questions, or descriptions keyed by "false"
            and "true" for noul questions.
        label: Optional zero-based option index. Noul uses 0 for false and 1 for
            true. Omit for inference.
    """

    type: Literal["choice", "score", "noul"]
    state: str  # The State the question is asked in.
    instructions: str  # The actual question text
    criteria: NotRequired[dict[str, str | None] | list[str] | None]
    label: NotRequired[int]


class EncodedQuestion(TypedDict):
    """Store one tokenized question with unpadded token positions.

    Attributes:
        ids: Long tensor of shape (tokens,) on the encoding device.
        option_idxs: Token position of each option-end delimiter, in option
            order.
        decide_idx: Token position of the final decision delimiter.
        label: Optional zero-based option index; omitted for inference.
    """

    ids: torch.Tensor  # [tokens]; inference uses a 1D long tensor.
    option_idxs: list[int]  # Token index of each option's end delimiter.
    decide_idx: int  # Token index of the final decision delimiter.
    label: NotRequired[int]  # Zero-based option index; omitted for inference.


class QuestionBatch(TypedDict):
    """Store right-padded questions and their original examples.

    Attributes:
        ids: Long tensor of shape (batch_size, max_tokens).
        mask: Boolean tensor of shape (batch_size, max_tokens), true for real
            tokens.
        examples: Original encoded questions in batch order, including their
            token tensors.
    """

    ids: torch.Tensor  # Long [batch_size, max_tokens], padded with pad_id.
    mask: torch.Tensor  # Bool [batch_size, max_tokens]; True for real tokens.
    examples: list[EncodedQuestion]  # One example per tensor row, in order.


def collate_questions(
    examples: list[EncodedQuestion],
    pad_id: int = 0,
) -> QuestionBatch:
    """Right-pad CPU token rows and retain the original examples.

    Args:
        examples: Nonempty list of encoded questions with CPU token tensors.
        pad_id: Token ID used for right padding.

    Returns:
        A batch containing padded token IDs, a validity mask, and the original
        examples.
    """
    rows = [torch.as_tensor(example["ids"], dtype=torch.long) for example in examples]
    ids = pad_sequence(rows, batch_first=True, padding_value=pad_id)
    lengths = torch.tensor([len(row) for row in rows])
    mask = torch.arange(ids.shape[1])[None, :] < lengths[:, None]
    return {"ids": ids, "mask": mask, "examples": examples}


def batch_to_device(
    batch: QuestionBatch,
    device: str | torch.device,
) -> QuestionBatch:
    """Move the padded batch tensors to a device.

    Args:
        batch: Batch to transfer.
        device: Destination device for ids and mask.

    Returns:
        A new batch mapping with transferred ids and mask. The examples list and
        its tensors are left unchanged.
    """
    return {**batch, "ids": batch["ids"].to(device), "mask": batch["mask"].to(device)}


def parse_options(
    question: Question,
) -> tuple[
    list[str],
    list[str] | list[bool] | list[int],
    int | None,
]:
    """Resolve option text, answer keys, and the optional label.

    Args:
        question: Raw choice, score, or noul question.

    Returns:
        A tuple of option text, corresponding answer keys, and the label or
        None. Keys are strings for choice, integer indices for score, and
        [False, True] for noul.

    Raises:
        ValueError: The question type is unsupported.
    """
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
    """Encode a question using the shared training and inference layout.

    Args:
        question: Raw question with an optional zero-based label.
        tokenizer: Tokenizer supporting encode and convert_tokens_to_ids for
            SPECIAL_TOKENS.
        device: Device on which to create the token tensor.

    Returns:
        Token IDs, option-end positions, and the decision position, plus a label
        when supplied.

    Raises:
        ValueError: The question type is unsupported.
    """

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
    """Read labeled JSONL records and flatten their questions.

    Args:
        path: Path to a JSONL file containing state and questions records.

    Returns:
        Questions with string state and instructions. Choice-key labels become
        option indices, and boolean noul labels become integers.

    Raises:
        ValueError: A question has no label, a choice label is absent from its
            criteria, or a record contains invalid JSON.
    """
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
    """Build a shuffled loader from train.jsonl.

    Args:
        data_dir: Directory containing the labeled training file.
        tokenizer: Tokenizer used to encode each question on the CPU.
        batch_size: Maximum number of questions per batch.
        seed: Seed for the loader generator used for shuffling.

    Returns:
        A DataLoader yielding right-padded question batches.
    """
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
    """Build an ordered loader from test.jsonl.

    Args:
        data_dir: Directory containing the labeled evaluation file.
        tokenizer: Tokenizer used to encode each question on the CPU.
        batch_size: Maximum number of questions per batch.

    Returns:
        A DataLoader yielding right-padded question batches in file order.
    """
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

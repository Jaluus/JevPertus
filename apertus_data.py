"""Read the repository's labelled JSONL data as one causal row per question."""

import json

SPECIAL_TOKENS = {
    "context": "<SPECIAL_100>",
    "question": "<SPECIAL_101>",
    "option_start": "<SPECIAL_102>",
    "option_end": "<SPECIAL_103>",
    "decide": "<SPECIAL_104>",
}


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


def encode_question(state, question, tokenizer, max_length=1024, max_state=384):
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


def load_examples(path, tokenizer, max_length=1024, max_state=384):
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

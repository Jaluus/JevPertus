"""Evaluate Jev checkpoints on multiple-choice benchmarks without generation."""

import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from string import ascii_uppercase

BENCHMARKS = {
    "mmlu": ("cais/mmlu", "all"),
    "mmlu-pro": ("TIGER-Lab/MMLU-Pro", "default"),
    "arc-challenge": ("allenai/ai2_arc", "ARC-Challenge"),
    "global-mmlu": ("CohereLabs/Global-MMLU", None),
}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory, e.g. runs/jevpertus-v1.5-8B/epoch_2",
    )
    parser.add_argument(
        "--output-dir",
        help="New result directory (default: <checkpoint>/evals)",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS)
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="Global-MMLU language codes, or all (default: en)",
    )
    parser.add_argument(
        "--split",
        choices=["test", "validation", "dev"],
        default="test",
        help="Must exist in every selected dataset",
    )
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument(
        "--device", default=None, help="Default: cuda:0 if available, otherwise cpu"
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="Random sample size per benchmark/language; omit for full evaluation",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=8192,
        help="Fail on longer inputs; never silently truncate",
    )
    parser.add_argument(
        "--state",
        default="Choose the correct answer.",
        help="Fixed context used for every language",
    )
    parser.add_argument(
        "--dataset-revision",
        action="append",
        default=[],
        metavar="BENCHMARK=REVISION",
        help="Repeat to pin datasets; defaults resolve main to a commit",
    )
    args = parser.parse_args()
    args.benchmarks = list(dict.fromkeys(args.benchmarks))
    args.languages = list(dict.fromkeys(args.languages))
    revisions = {}
    for item in args.dataset_revision:
        name, separator, revision = item.partition("=")
        if not separator or name not in BENCHMARKS or not revision:
            parser.error("--dataset-revision requires BENCHMARK=REVISION")
        revisions[name] = revision
    args.dataset_revision = revisions
    if "all" in args.languages and len(args.languages) != 1:
        parser.error("--languages all cannot be combined with language codes")
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint, "evals")
    if os.path.exists(args.output_dir):
        parser.error(
            f"Result directory already exists: {args.output_dir}. "
            "Choose a new --output-dir to preserve previous results."
        )
    return args


def adapt_row(benchmark, row, index, state):
    """Preserve source option order and convert the answer to a zero-based index."""
    if benchmark == "arc-challenge":
        options = row["choices"]["text"]
        keys = row["choices"]["label"]
        label = keys.index(row["answerKey"])
    elif benchmark == "global-mmlu":
        keys = list("ABCD")
        options = [row[f"option_{key.lower()}"] for key in keys]
        label = keys.index(row["answer"])
    else:
        options = row["options"] if benchmark == "mmlu-pro" else row["choices"]
        keys = list(ascii_uppercase[: len(options)])
        label = int(row["answer_index"] if benchmark == "mmlu-pro" else row["answer"])
    if (
        not 2 <= len(options) <= 26
        or len(keys) != len(options)
        or len(set(keys)) != len(keys)
    ):
        raise ValueError(f"Invalid choices in {benchmark} row {index}")
    if not 0 <= label < len(options):
        raise ValueError(f"Invalid answer in {benchmark} row {index}")
    if not isinstance(row["question"], str) or not all(
        isinstance(option, str) for option in options
    ):
        raise ValueError(f"Non-text input in {benchmark} row {index}")
    return {
        "id": str(row.get("sample_id", row.get("question_id", row.get("id", index)))),
        "row_index": index,
        "subject": row.get("subject", row.get("category", "science")),
        "question": {
            "type": "choice",
            "state": state,
            "instructions": row["question"],
            "criteria": dict(zip(keys, options)),
            "label": label,
        },
    }


def summarize(records):
    """Return question-weighted accuracy/loss and equally weighted subject accuracy."""

    def metrics(rows):
        return {
            "count": len(rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "cross_entropy": sum(row["cross_entropy"] for row in rows) / len(rows),
        }

    subjects = defaultdict(list)
    for record in records:
        subjects[record["subject"]].append(record)
    by_subject = {subject: metrics(rows) for subject, rows in sorted(subjects.items())}
    return {
        **metrics(records),
        "subject_macro_accuracy": sum(row["accuracy"] for row in by_subject.values())
        / len(by_subject),
        "by_subject": by_subject,
    }


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def main():
    args = parse_args()
    # Keep --help available without installing the model runtime.
    import torch
    from datasets import get_dataset_config_names, load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    from dataloader import (
        SPECIAL_TOKENS,
        batch_to_device,
        collate_questions,
        encode_question,
    )
    from modeling.apertus import load_apertus
    from modeling.jev import JevModel

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda = torch.device(device).type == "cuda"
    dtype = torch.bfloat16 if cuda else torch.float32
    with open(
        os.path.join(args.checkpoint, "jev_config.json"), encoding="utf-8"
    ) as file:
        config = json.load(file)
    tokenizer = AutoTokenizer.from_pretrained(
        config["backbone"]["model_id"], revision=config["backbone"].get("revision")
    )
    token_ids = [
        tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS.values()
    ]
    if (
        None in token_ids
        or tokenizer.unk_token_id in token_ids
        or len(set(token_ids)) != len(token_ids)
    ):
        raise ValueError("Tokenizer does not provide distinct Jev special tokens")
    # Refuse to overwrite previous results, including interrupted runs.
    os.makedirs(args.output_dir, exist_ok=False)
    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "zero-shot Jev pointer-head scoring; source option order; no generation",
        "arguments": vars(args),
        "device": device,
        "dtype": str(dtype),
        "checkpoint_config": config,
        "datasets": [],
    }
    manifest_path = os.path.join(args.output_dir, "run.json")
    write_json(manifest_path, manifest)
    model = JevModel.from_pretrained(
        args.checkpoint, llm_loader=load_apertus, device=device, dtype=dtype
    )
    model.eval()
    max_tokens = min(args.max_tokens, model.llm.context_len)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    api = HfApi()
    results = {}
    for benchmark in args.benchmarks:
        repo, default_config = BENCHMARKS[benchmark]
        revision = api.dataset_info(
            repo, revision=args.dataset_revision.get(benchmark, "main")
        ).sha
        configurations = [default_config]
        if benchmark == "global-mmlu":
            available = get_dataset_config_names(repo, revision=revision)
            configurations = (
                sorted(available) if args.languages == ["all"] else args.languages
            )
            if set(configurations) - set(available):
                raise ValueError(
                    f"Unknown Global-MMLU languages: {set(configurations) - set(available)}"
                )
        for configuration in configurations:
            name = (
                f"{benchmark}-{configuration}"
                if benchmark == "global-mmlu"
                else benchmark
            )
            print(f"Loading {name} ({args.split})...", flush=True)
            dataset = load_dataset(
                repo, configuration, split=args.split, revision=revision
            )
            indices = list(range(len(dataset)))
            if args.limit and args.limit < len(indices):
                indices = sorted(random.Random(args.seed).sample(indices, args.limit))
            if not indices:
                raise ValueError(f"Empty dataset: {name}")
            manifest["datasets"].append(
                {
                    "name": name,
                    "repo": repo,
                    "config": configuration,
                    "revision": revision,
                    "split": args.split,
                    "available_count": len(dataset),
                    "selected_count": len(indices),
                    "fingerprint": dataset._fingerprint,
                }
            )
            write_json(manifest_path, manifest)
            records = []
            with open(
                os.path.join(args.output_dir, f"{name}.jsonl"),
                "x",
                encoding="utf-8",
                buffering=1,
            ) as output:
                with torch.inference_mode():
                    for start in range(0, len(indices), args.batch_size):
                        items = [
                            adapt_row(benchmark, dataset[index], index, args.state)
                            for index in indices[start : start + args.batch_size]
                        ]
                        examples = [
                            encode_question(item["question"], tokenizer)
                            for item in items
                        ]
                        for item, example in zip(items, examples):
                            if len(example["ids"]) > max_tokens:
                                raise ValueError(
                                    f"{name} row {item['row_index']} exceeds {max_tokens} tokens; increase --max-tokens if supported"
                                )
                        batch = batch_to_device(
                            collate_questions(examples, pad_id), device
                        )
                        with torch.autocast(
                            device_type=torch.device(device).type,
                            dtype=torch.bfloat16,
                            enabled=cuda,
                        ):
                            logits = model.forward_batch(batch)
                        for item, scores in zip(items, logits):
                            if not torch.isfinite(scores).all():
                                raise ValueError(
                                    f"Non-finite logits in {name} row {item['row_index']}"
                                )
                            log_probs = scores.float().log_softmax(dim=0)
                            prediction = scores.argmax().item()
                            label = item["question"]["label"]
                            record = {
                                **item,
                                "benchmark": benchmark,
                                "language": (
                                    configuration
                                    if benchmark == "global-mmlu"
                                    else "en"
                                ),
                                "prediction": prediction,
                                "correct": prediction == label,
                                "probabilities": log_probs.exp().tolist(),
                                "cross_entropy": -log_probs[label].item(),
                            }
                            output.write(
                                json.dumps(record, ensure_ascii=False, allow_nan=False)
                                + "\n"
                            )
                            records.append(
                                {
                                    key: record[key]
                                    for key in ("subject", "correct", "cross_entropy")
                                }
                            )
                        print(
                            f"{name}: {len(records)}/{len(indices)}",
                            end="\r",
                            flush=True,
                        )
            results[name] = summarize(records)
            write_json(os.path.join(args.output_dir, "summary.json"), results)
            print(f"\n{name}: accuracy={results[name]['accuracy']:.2%}", flush=True)
    manifest["status"] = "complete"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()

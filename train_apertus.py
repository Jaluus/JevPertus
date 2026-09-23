"""Minimal JevType training: frozen Apertus + LoRA + an option pointer head."""

import json
import os
from functools import partial

import torch
import torch.nn.functional as F
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from apertus_data import batch_to_device, collate_questions, load_examples
from modeling.apertus.apertus_8b import ApertusModel
from modeling.jev import JevModel, PointerHead

# Edit these constants before running: python train_apertus.py
DATA_DIR = "data"
BASE_MODEL = "swiss-ai/Apertus-v1.5-8B"
BASE_REVISION = "main"
DEVICE = "cuda:0"
EPOCHS = 4
BATCH_SIZE = 16
LORA_RANK = 16
LEARNING_RATE = 5e-5
SEED = 0
OUTPUT_DIR = "runs/jevpertus_V2"


def build_model(
    base,
    revision,
    device,
    rank,
):
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

    backbone = ApertusModel.from_pretrained(
        base,
        revision=revision,
        device=device,
        dtype=dtype,
    )

    config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.05,
        bias="none",
        target_modules=["w_q", "w_k", "w_v", "w_o", "up", "down"],
    )

    head = PointerHead(backbone.input_layer.embedding.embedding_dim).to(
        device=device, dtype=dtype
    )

    return JevModel(backbone, head).add_lora(config)


def question_losses(logits, examples):
    """One cross-entropy per question, allowing different option counts."""
    return torch.stack(
        [
            F.cross_entropy(
                scores[None].float(),
                torch.tensor([example["label"]], device=scores.device),
            )
            for scores, example in zip(logits, examples)
        ]
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct = 0.0, 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        examples = batch["examples"]
        logits = model.forward_batch(batch)
        total_loss += question_losses(logits, examples).sum().item()
        correct += sum(
            scores.argmax().item() == example["label"]
            for scores, example in zip(logits, examples)
        )

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)

    print(f"Loading training data from {DATA_DIR}...")
    train = load_examples(os.path.join(DATA_DIR, "train.jsonl"), tokenizer)
    development = load_examples(os.path.join(DATA_DIR, "development.jsonl"), tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate = partial(collate_questions, pad_id=pad_id)
    train_loader = DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(SEED),
    )
    dev_loader = DataLoader(
        development,
        batch_size=BATCH_SIZE,
        collate_fn=collate,
    )

    print("Building model...")
    model = build_model(BASE_MODEL, BASE_REVISION, DEVICE, LORA_RANK)

    parameters = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in parameters):,}")

    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.01)

    step = 0
    # Line buffering keeps completed steps on disk even if training is interrupted.
    with open(
        os.path.join(OUTPUT_DIR, "loss_history.jsonl"),
        "w",
        encoding="utf-8",
        buffering=1,
    ) as history:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            seen = 0
            for batch in train_loader:
                batch = batch_to_device(batch, DEVICE)
                examples = batch["examples"]

                optimizer.zero_grad(set_to_none=True)
                logits = model.forward_batch(batch)
                losses = question_losses(logits, examples)
                batch_loss = losses.sum().item() / len(examples)
                total_loss += batch_loss * len(examples)
                seen += len(examples)
                losses.mean().backward()

                optimizer.step()
                step += 1
                history.write(
                    json.dumps(
                        {
                            "split": "train",
                            "epoch": epoch + 1,
                            "step": step,
                            "questions": seen,
                            "batch_size": len(examples),
                            "loss": batch_loss,
                            "epoch_loss": total_loss / seen,
                        }
                    )
                    + "\n"
                )

                print(
                    f"epoch {epoch + 1} questions {seen}/{len(train)} loss {total_loss / seen:.4f}",
                    end="\r",
                    flush=True,
                )

            dev_loss, accuracy = evaluate(model, dev_loader, DEVICE)
            history.write(
                json.dumps(
                    {
                        "split": "development",
                        "epoch": epoch + 1,
                        "step": step,
                        "loss": dev_loss,
                        "accuracy": accuracy,
                    }
                )
                + "\n"
            )

            print(
                f"epoch {epoch + 1}: development loss {dev_loss:.4f}, accuracy {accuracy:.3%}"
            )

            model.save_pretrained(
                OUTPUT_DIR,
                backbone_config={
                    "model_id": BASE_MODEL,
                    "revision": BASE_REVISION,
                },
                training_config={
                    "data": DATA_DIR,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LEARNING_RATE,
                    "seed": SEED,
                },
            )


if __name__ == "__main__":
    main()

"""Minimal JevType training: frozen Apertus + LoRA + an option pointer head."""

import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from dataloader import batch_to_device, build_trainloader, build_testloader
from modeling.jev import build_jev

# Edit these constants before running: python train_apertus.py
DATA_DIR = "data"
BASE_MODEL = "swiss-ai/Apertus-v1.5-8B"
DEVICE = "cuda:0"
EPOCHS = 1
BATCH_SIZE = 4
LORA_RANK = 16
LEARNING_RATE = 5e-5
SEED = 0
OUTPUT_DIR = "runs/jevpertus_V3"


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

        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            logits = model.forward_batch(batch)
            losses = question_losses(logits, examples)

        total_loss += losses.sum().item()
        correct += sum(
            scores.argmax().item() == example["label"]
            for scores, example in zip(logits, examples)
        )

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_loader = build_trainloader(
        DATA_DIR,
        tokenizer,
        batch_size=BATCH_SIZE,
    )
    test_loader = build_testloader(
        DATA_DIR,
        tokenizer,
        batch_size=BATCH_SIZE,
    )

    print(f"Loading training data from {DATA_DIR}...")

    print("Building model...")
    model = build_jev(BASE_MODEL, DEVICE, LORA_RANK)

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

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=torch.device(DEVICE).type,
                    dtype=torch.bfloat16,
                    enabled=torch.device(DEVICE).type == "cuda",
                ):
                    logits = model.forward_batch(batch)
                    losses = question_losses(logits, batch["examples"])
                    loss = losses.mean()

                total_loss += loss.item() * BATCH_SIZE
                seen += BATCH_SIZE
                loss.backward()

                optimizer.step()
                step += 1
                history.write(
                    json.dumps(
                        {
                            "split": "train",
                            "epoch": epoch + 1,
                            "step": step,
                            "questions": seen,
                            "batch_size": BATCH_SIZE,
                            "loss": loss.item(),
                            "epoch_loss": total_loss / seen,
                        }
                    )
                    + "\n"
                )

                print(
                    f"epoch {epoch + 1} questions {seen}/{len(train_loader)} loss {total_loss / seen:.4f}",
                    end="\r",
                    flush=True,
                )

            dev_loss, accuracy = evaluate(model, test_loader, DEVICE)
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
                os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}"),
                backbone_config={
                    "model_id": BASE_MODEL,
                    "revision": "main",
                },
                training_config={
                    "data": DATA_DIR,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LEARNING_RATE,
                    "seed": SEED,
                    "precision": (
                        "torch.bfloat16"
                        if torch.device(DEVICE).type == "cuda"
                        else "torch.float32"
                    ),
                },
            )


if __name__ == "__main__":
    main()

"""Minimal JevType training: frozen Apertus + LoRA + an option pointer head."""

import os
import random

import torch
import torch.nn.functional as F
from peft import LoraConfig
from transformers import AutoTokenizer

from apertus_data import load_examples
from modeling.apertus.apertus_8b import ApertusModel
from modeling.jev import JevModel, PointerHead

# Edit these constants before running: python train_apertus.py
DATA_DIR = "data"
BASE_MODEL = "swiss-ai/Apertus-v1.5-8B"
BASE_REVISION = "main"
DEVICE = "cuda"
EPOCHS = 1
LORA_RANK = 16
LEARNING_RATE = 5e-5
SEED = 0
OUTPUT_DIR = "runs/jevpertus"


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


@torch.no_grad()
def evaluate(model, examples, device):
    model.eval()
    total_loss, correct = 0.0, 0

    for example in examples:
        example = {**example, "ids": torch.tensor(example["ids"], device=device)}
        logits = model(example)
        target = torch.tensor([example["label"]], device=logits.device)
        total_loss += F.cross_entropy(logits[None].float(), target).item()
        correct += logits.argmax().item() == example["label"]

    return total_loss / len(examples), correct / len(examples)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=False)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)

    train = load_examples(os.path.join(DATA_DIR, "train.jsonl"), tokenizer)
    development = load_examples(os.path.join(DATA_DIR, "development.jsonl"), tokenizer)

    model = build_model(BASE_MODEL, BASE_REVISION, DEVICE, LORA_RANK)

    parameters = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in parameters):,}")

    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.01)

    for epoch in range(EPOCHS):
        rng.shuffle(train)
        model.train()
        total_loss = 0.0
        for index, example in enumerate(train, start=1):

            optimizer.zero_grad(set_to_none=True)
            example = {
                **example,
                "ids": torch.tensor(example["ids"], device=DEVICE),
            }

            logits = model(example)

            target = torch.tensor([example["label"]], device=logits.device)
            loss = F.cross_entropy(logits[None].float(), target)
            total_loss += loss.item()
            loss.backward()

            optimizer.step()

            if index % 20 == 0:
                print(
                    f"epoch {epoch + 1} questions {index}/{len(train)} loss {total_loss / index:.4f}",
                    flush=True,
                )

        dev_loss, accuracy = evaluate(model, development, DEVICE)

        print(
            f"epoch {epoch + 1}: development loss {dev_loss:.4f}, accuracy {accuracy:.3%}"
        )

        model.save_pretrained(
            OUTPUT_DIR,
            tokenizer,
            backbone_config={"model_id": BASE_MODEL, "revision": BASE_REVISION},
            training_config={
                "data": DATA_DIR,
                "epochs": EPOCHS,
                "lr": LEARNING_RATE,
                "seed": SEED,
            },
        )


if __name__ == "__main__":
    main()

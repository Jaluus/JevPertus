"""Load a trained Jev model and answer one unlabelled question."""

import json
import os

import torch
from transformers import AutoTokenizer

from apertus_data import encode_question
from modeling.apertus.apertus_8b import ApertusModel
from modeling.jev import JevModel

CHECKPOINT = "runs/jevpertus"
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
STATE = "I was charged twice for the same order."
QUESTION = {
    "type": "choice",
    "instructions": "Which department should handle this?",
    "criteria": {"billing": None, "shipping": None, "returns": None},
}


def main():
    with open(os.path.join(CHECKPOINT, "jev_config.json"), encoding="utf-8") as file:
        config = json.load(file)
    tokenizer = AutoTokenizer.from_pretrained(
        config["backbone"]["model_id"],
        revision=config["backbone"].get("revision"),
    )
    model = JevModel.from_pretrained(
        CHECKPOINT,
        backbone_loader=ApertusModel.from_pretrained,
        device=DEVICE,
        dtype=torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32,
    )
    example = encode_question(
        STATE,
        QUESTION,
        tokenizer,
    )
    example["ids"] = torch.tensor(example["ids"], device=DEVICE)
    print(model.predict(example))


if __name__ == "__main__":
    main()

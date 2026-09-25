"""Load a trained Jev model and print predictions for example questions."""

import json
import os

import torch
from transformers import AutoTokenizer

from dataloader import Question, encode_question
from modeling.apertus import load_apertus
from modeling.jev import JevModel

CHECKPOINT = "runs/jevpertus-v1.5-8B/epoch_2"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
QUESTIONS: list[Question] = [
    {
        "type": "choice",
        "state": "You are a medical expert.",
        "instructions": "A scientist is studying the properties of myosin-actin interactions in a sample of human muscle tissue. She has identified a drug that selectively inhibits phosphate release by the myosin head. If she gives this drug to a sample of human muscle tissue under physiologic conditions, which of the following steps in cross-bridge cycling will most likely be blocked?",
        "criteria": {
            "A": "Myosin head release from actin",
            "B": "Myosin head cocking",
            "C": "Exposure of myosin-binding sites on actin",
            "D": "Myosin head binding to actin",
            "E": "Power stroke",
        },
    },
    {
        "type": "score",
        "state": "You went to a resturant, but got served cold soup, after ordering hot soup.",
        "instructions": "How would you rate the quality of the service you received?",
        "criteria": [
            "Very poor",
            "Poor",
            "Average",
            "Good",
            "Excellent",
        ],
    },
    {
        "type": "noul",
        "state": "You live in New York.",
        "instructions": "Boston is closer to you than Los Angeles.",
    },
]


def print_probabilities(question: Question, probabilities: torch.Tensor):
    """Print question context and formatted option probabilities.

    For noul questions, display the probability of true from element 1,
    matching the encoder's [False, True] option order.

    Args:
        question: Raw question specifying the option labels to display.
        probabilities: One-dimensional probability tensor in encoded option
            order.
    """

    print(f"Context: {question['state']}")
    print(f"Question: {question['instructions']}")

    if question["type"] == "choice":
        for key, value, prob in zip(
            question["criteria"].keys(), question["criteria"].values(), probabilities
        ):
            print(f"Option {key} ({value}): {prob:.2f}")

    elif question["type"] == "score":
        for criterion, prob in zip(question["criteria"], probabilities):
            print(f"Score '{criterion}': {prob:.2f}")

    elif question["type"] == "noul":
        print(f"Probability of statement being true: {probabilities[1]:.2f}")


def main():
    """Load the configured checkpoint and print predictions for QUESTIONS."""
    with open(os.path.join(CHECKPOINT, "jev_config.json"), encoding="utf-8") as file:
        config = json.load(file)
    tokenizer = AutoTokenizer.from_pretrained(
        config["backbone"]["model_id"],
        revision=config["backbone"].get("revision"),
    )

    model = JevModel.from_pretrained(
        CHECKPOINT,
        llm_loader=load_apertus,
        device=DEVICE,
        dtype=torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32,
    )

    for question in QUESTIONS:
        encoded_question = encode_question(question, tokenizer, device=DEVICE)
        probabilities = model.predict(encoded_question)
        print_probabilities(question, probabilities)
        print("-" * 80)


if __name__ == "__main__":
    main()

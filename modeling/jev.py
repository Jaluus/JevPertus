"""A language backbone + a pluggable option-scoring head, with LoRA checkpoints."""

import json
import os

import torch
from peft import (
    LoraConfig,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from torch import nn

from .apertus import load_apertus, ApertusModel
from .pointerhead import PointerHead
from dataloader import QuestionBatch, EncodedQuestion


class JevModel(nn.Module):
    """Backbone: partial_forward(ids) -> [batch, tokens, hidden_dim].

    Head: forward(decide, options) -> one logit per option. For saving, the
    head exposes a JSON-serializable `config` of its constructor arguments.
    forward scores one question; forward_batch scores an already padded batch
    whose ids and mask are on the model device.
    The caller places example["ids"] on the model device as a 1D tensor and
    initializes the backbone and head with matching device and dtype.
    """

    def __init__(
        self,
        llm_backbone: ApertusModel,
        pointerhead: PointerHead,
    ):
        super().__init__()
        self.llm_backbone = llm_backbone
        self.pointerhead = pointerhead

    def add_lora(self, config):

        inject_adapter_in_model(config, self.llm_backbone)

        # We keep precision for the backbone, but the LoRA weights are in float32.
        # The pointer head is also in float32, since it is small and we want to avoid precision loss.
        for parameter in self.llm_backbone.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        return self

    def forward(self, example: EncodedQuestion) -> torch.Tensor:

        # Unsqueeze to add a batch dimension, since the backbone expects [batch, tokens].
        token_ids = example["ids"].unsqueeze(0)

        hidden = self.llm_backbone.partial_forward(token_ids)[0]

        decide_state = hidden[example["decide_idx"]]
        option_states = hidden[example["option_idxs"]]
        return self.pointerhead(decide_state, option_states)

    def forward_batch(self, batch: QuestionBatch) -> list[torch.Tensor]:
        """Run one backbone pass and return logits per question.

        Option counts may differ, so the small pointer head runs per question.
        """

        hidden = self.llm_backbone.partial_forward(batch["ids"], mask=batch["mask"])

        return [
            self.pointerhead(row[example["decide_idx"]], row[example["option_idxs"]])
            for row, example in zip(hidden, batch["examples"])
        ]

    def save_pretrained(
        self,
        directory,
        backbone_config,
        training_config=None,
    ):
        """Save adapter + head, not the frozen base or optimizer state.

        backbone_config contains keyword arguments for the base weight loader
        (e.g. model_id and a pinned revision). The caller chooses that loader
        explicitly when reloading; no classes are dynamically imported.
        """

        os.makedirs(directory, exist_ok=True)

        self.llm_backbone.peft_config["default"].save_pretrained(directory)

        config = {
            "backbone": backbone_config,
            "head": self.pointerhead.config,
            "training": training_config or {},
        }

        config_path = os.path.join(directory, "jev_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

        torch.save(
            {
                "adapter": get_peft_model_state_dict(self.llm_backbone),
                "head": self.pointerhead.state_dict(),
            },
            os.path.join(directory, "jev.pt"),
        )

    @classmethod
    def from_pretrained(
        cls,
        directory,
        backbone_loader,
        head_loader=PointerHead,
        device="cpu",
        dtype=torch.float32,
    ):
        """Recreate base + LoRA + head. Pass the same head class used in training."""

        with open(
            os.path.join(directory, "jev_config.json"),
            encoding="utf-8",
        ) as f:
            config = json.load(f)

        backbone = backbone_loader(**config["backbone"], device=device, dtype=dtype)
        head = head_loader(**config["head"]).to(device=device, dtype=dtype)
        model = cls(backbone, head).add_lora(LoraConfig.from_pretrained(directory))

        weight_path = os.path.join(directory, "jev.pt")
        weights = torch.load(weight_path, map_location="cpu", weights_only=True)

        set_peft_model_state_dict(model.llm_backbone, weights["adapter"])
        model.pointerhead.load_state_dict(weights["head"])
        return model.eval()


def build_jev(
    base_model: str,
    lora_rank: int,
    device: str,
    revision: str = "main",
):
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32

    llm_backbone = load_apertus(
        base_model,
        device=device,
        dtype=dtype,
        revision=revision,
    )

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.05,
        bias="none",
        target_modules=["w_q", "w_k", "w_v", "w_o", "up", "down"],
    )

    pointerhead = PointerHead(llm_backbone.input_layer.embedding_dim).to(
        device=device,
        dtype=torch.float32,
    )

    return JevModel(llm_backbone, pointerhead).add_lora(lora_config)

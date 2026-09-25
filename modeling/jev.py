"""Combine an Apertus backbone and option-scoring head with LoRA checkpoints."""

import json
import os
from typing import cast

import torch
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from torch import nn

from dataloader import EncodedQuestion, QuestionBatch

from .apertus import ApertusModel, load_apertus
from .pointerhead import PointerHead


class JevModel(nn.Module):
    """Score question options with an Apertus backbone and pointer head.

    Inputs must be on the backbone device. Pointer-head inputs must match its
    weight dtype; use autocast when the backbone and head have different dtypes.

    Attributes:
        llm: Backbone providing normalized token hidden states.
        pointerhead: Option-scoring head with a JSON-serializable constructor
            config.
    """

    def __init__(
        self,
        llm: ApertusModel,
        pointerhead: PointerHead,
    ):
        """Attach the backbone and option-scoring head.

        Args:
            llm: Apertus backbone.
            pointerhead: Head compatible with the backbone hidden-state width.
        """
        super().__init__()
        self.llm = llm
        self.pointerhead = pointerhead

    def add_lora(self, config):
        """Inject LoRA adapters and cast trainable backbone weights to float32.

        Args:
            config: PEFT adapter configuration to inject into the backbone.

        Returns:
            This model with adapters attached. The pointer-head dtype is unchanged.
        """
        inject_adapter_in_model(config, self.llm)

        # We keep precision for the backbone, but the LoRA weights are in float32.
        # The pointer head is also in float32, since it is small and we want to avoid precision loss.
        for parameter in self.llm.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        return self

    def forward(self, example: EncodedQuestion) -> torch.Tensor:

        # Unsqueeze to add a batch dimension, since the backbone expects [batch, tokens].
        """Compute option logits for one question.

        Args:
            example: Encoded question with one-dimensional token IDs on the model
                device.

        Returns:
            Unnormalized logits of shape (num_options,).
        """
        token_ids = example["ids"].unsqueeze(0)

        hidden = self.llm.partial_forward(token_ids)[0]

        decide_state = hidden[example["decide_idx"]]
        option_states = hidden[example["option_idxs"]]
        return self.pointerhead(decide_state, option_states)

    def forward_batch(self, batch: QuestionBatch) -> list[torch.Tensor]:
        """Score a padded batch with one backbone pass.

        Args:
            batch: Right-padded question batch with ids and mask on the model
                device.

        Returns:
            One logit tensor of shape (num_options,) per question, in batch order.
            Option counts may differ.
        """

        hidden = self.llm.partial_forward(batch["ids"], mask=batch["mask"])

        return [
            self.pointerhead(row[example["decide_idx"]], row[example["option_idxs"]])
            for row, example in zip(hidden, batch["examples"])
        ]

    @torch.inference_mode()
    def predict(self, example: EncodedQuestion) -> torch.Tensor:
        """Compute option probabilities for one question.

        This method does not change evaluation mode or disable gradient tracking.

        Args:
            example: Encoded question on the model device.

        Returns:
            Probabilities of shape (num_options,) in encoded option order.
        """
        logits = self.forward(example)
        return torch.softmax(logits, dim=0)

    def save_pretrained(
        self,
        directory,
        backbone_config,
        training_config=None,
    ):
        """Save the default LoRA adapter, pointer head, and configuration.

        Requires an attached default PEFT adapter. Frozen backbone weights and
        optimizer state are not saved. Floating-point weights are saved as CPU
        bfloat16 copies to reduce checkpoint size without changing live parameters.

        Args:
            directory: Output directory, created if needed.
            backbone_config: JSON-serializable keyword arguments for the backbone
                loader, such as model_id and revision.
            training_config: Optional JSON-serializable training metadata.
        """

        os.makedirs(directory, exist_ok=True)

        # PEFT attaches this config mapping dynamically during adapter injection.
        peft_config = cast(dict[str, PeftConfig], self.llm.peft_config)
        peft_config["default"].save_pretrained(directory)

        config = {
            "backbone": backbone_config,
            "head": self.pointerhead.config,
            "training": training_config or {},
        }

        config_path = os.path.join(directory, "jev_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

        weights = {
            "adapter": get_peft_model_state_dict(self.llm),
            "head": self.pointerhead.state_dict(),
        }
        weights = {
            name: {
                key: value.detach().to(
                    device="cpu",
                    dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
                )
                for key, value in state.items()
            }
            for name, state in weights.items()
        }
        torch.save(weights, os.path.join(directory, "jev.pt"))

    @classmethod
    def from_pretrained(
        cls,
        directory,
        llm_loader,
        head_loader=PointerHead,
        device="cpu",
        dtype=torch.float32,
    ):
        """Restore the backbone, LoRA adapter, and pointer head.

        Args:
            directory: Directory containing a saved Jev checkpoint.
            llm_loader: Callable accepting the saved backbone config plus device and
                dtype.
            head_loader: Head constructor accepting the saved head config; use the
                class used in training.
            device: Target device for the restored model.
            dtype: Dtype for the backbone and head. Trainable backbone parameters
                are cast to float32 when adapters are attached.

        Returns:
            The restored model in evaluation mode.
        """

        with open(
            os.path.join(directory, "jev_config.json"),
            encoding="utf-8",
        ) as f:
            config = json.load(f)

        llm = llm_loader(**config["backbone"], device=device, dtype=dtype)
        head = head_loader(**config["head"]).to(device=device, dtype=dtype)
        model = cls(llm, head).add_lora(LoraConfig.from_pretrained(directory))

        jev_weights = torch.load(
            os.path.join(directory, "jev.pt"),
            map_location="cpu",
            weights_only=True,
        )

        set_peft_model_state_dict(model.llm, jev_weights["adapter"])
        model.pointerhead.load_state_dict(jev_weights["head"])
        return model.eval()


def build_jev(
    base_model: str,
    lora_rank: int,
    device: str,
    revision: str = "main",
):
    """Build an Apertus model with LoRA adapters and a float32 pointer head.

    Args:
        base_model: Hugging Face model ID or local checkpoint directory.
        lora_rank: Rank of the LoRA adapters.
        device: Device for the backbone and pointer head.
        revision: Checkpoint revision to load for a remote model.

    Returns:
        A JevModel with adapters attached. The backbone uses bfloat16 on CUDA
        and float32 otherwise; trainable backbone parameters use float32.
    """
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32

    llm = load_apertus(
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

    pointerhead = PointerHead(llm.input_layer.embedding_dim).to(
        device=device,
        dtype=torch.float32,
    )

    return JevModel(llm, pointerhead).add_lora(lora_config)

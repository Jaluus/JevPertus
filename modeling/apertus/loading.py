"""Build Apertus text models from original or V1.5 Hugging Face checkpoints."""

import json
import os

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

from .apertus import ApertusModel


def checkpoint_names(model, model_type="apertus1p5"):
    """Map our parameter names to the appropriate checkpoint layout."""
    prefix = "model.language_model" if model_type == "apertus1p5" else "model"
    names = {
        "input_layer.weight": f"{prefix}.embed_tokens.weight",
        "final_norm.weight": f"{prefix}.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    block = {
        "rms_norm1.weight": "attention_layernorm.weight",
        "rms_norm2.weight": "feedforward_layernorm.weight",
        "attention.w_q.weight": "self_attn.q_proj.weight",
        "attention.w_k.weight": "self_attn.k_proj.weight",
        "attention.w_v.weight": "self_attn.v_proj.weight",
        "attention.w_o.weight": "self_attn.o_proj.weight",
        "attention.q_norm.weight": "self_attn.q_norm.weight",
        "attention.k_norm.weight": "self_attn.k_norm.weight",
        "ff.up.weight": "mlp.up_proj.weight",
        "ff.down.weight": "mlp.down_proj.weight",
        "ff.activation.alpha_p": "mlp.act_fn.alpha_p",
        "ff.activation.alpha_n": "mlp.act_fn.alpha_n",
    }
    for i in range(len(model.transformer_blocks)):
        names.update(
            {
                f"transformer_blocks.{i}.{ours}": f"{prefix}.layers.{i}.{official}"
                for ours, official in block.items()
            }
        )
    return names


def load_apertus(
    model_id="swiss-ai/Apertus-v1.5-8B",
    *,
    device="cpu",
    dtype=torch.bfloat16,
    cache_dir=None,
    revision="main",
):
    """Build and load an original or V1.5 Apertus text model from an HF ID or folder.

    Model dimensions (including 8B/70B) are read from the checkpoint config.
    Returns the model in evaluation mode on the selected device.

    Requires `pip install huggingface_hub safetensors` and Hugging Face access
    to the gated V1.5 repository. Uses your saved HF login or HF_TOKEN.
    Only text tensors are loaded; image/audio tokenizers are skipped.
    """

    if not dtype.is_floating_point or torch.device(device).type == "meta":
        raise ValueError("Use a floating-point dtype and a real device")
    folder = os.fspath(model_id)
    if not os.path.isdir(folder):

        # Resolve metadata first; use its immutable revision for the shards.
        folder = snapshot_download(
            repo_id=str(model_id),
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=["config.json", "model.safetensors.index.json"],
        )
        revision = os.path.basename(folder)

    with open(os.path.join(folder, "config.json"), encoding="utf-8") as file:
        config = json.load(file)
    model_type = config.get("model_type")
    if model_type == "apertus1p5":
        c = config.get("text_config", {})
        if c.get("model_type") != "apertus1p5_text":
            raise ValueError("Expected an Apertus V1.5 text configuration")
    elif model_type == "apertus":
        c = config
    else:
        raise ValueError(f"Unsupported Apertus model type: {model_type!r}")
    rope = c.get("rope_parameters") or c.get("rope_scaling") or {}
    rope_theta = rope.get("rope_theta", c.get("rope_theta"))
    if (
        config.get("quantization_config")
        or c.get("quantization_config")
        or c.get("tie_word_embeddings", config.get("tie_word_embeddings", False))
        or c.get("attention_bias", False)
        or c.get("mlp_bias", False)
        or c.get("post_norm", False)
        or not c.get("qk_norm", True)
        or c.get("hidden_act") != "xielu"
        or c.get("rms_norm_eps") != 1e-5
        or rope.get("rope_type", rope.get("type")) != "llama3"
        or rope.get("original_max_position_embeddings") != 8192
        or rope.get("low_freq_factor") != 1.0
        or rope.get("high_freq_factor") != 4.0
        or rope.get("factor", 0) < 1
        or rope_theta is None
    ):
        raise ValueError(
            "Checkpoint architecture is not supported by this text implementation"
        )

    # Meta tensors allocate no storage: pretrained tensors replace them below.
    with torch.device("meta"):
        model = ApertusModel(
            vocab_size=c["vocab_size"],
            embed_dim=c["hidden_size"],
            hidden_dim=c["intermediate_size"],
            context_len=c["max_position_embeddings"],
            num_heads=c["num_attention_heads"],
            num_kv_groups=c["num_key_value_heads"],
            head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
            num_attn_blocks=c["num_hidden_layers"],
            rope_theta=rope_theta,
            rope_factor=rope["factor"],
            output_vocab_size=c.get("output_vocab_size"),
        )
    names = checkpoint_names(model, model_type)
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as file:
            weight_map = json.load(file)["weight_map"]
    else:
        weight_map = {name: "model.safetensors" for name in names.values()}
    missing = set(names.values()) - weight_map.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing text weights: {sorted(missing)}")

    shards = {}
    for ours, official in names.items():
        shards.setdefault(weight_map[official], []).append((ours, official))
    for filename, entries in shards.items():
        path = os.path.join(folder, filename)
        if not os.path.exists(path) and not os.path.isdir(model_id):

            path = hf_hub_download(
                str(model_id), filename, revision=revision, cache_dir=cache_dir
            )
        with safe_open(path, framework="pt", device="cpu") as shard:
            missing = {official for _, official in entries} - set(shard.keys())
            if missing:
                raise ValueError(
                    f"Checkpoint is missing text weights: {sorted(missing)}"
                )
            for ours, official in entries:
                tensor = shard.get_tensor(official)
                expected = model.get_parameter(ours).shape
                if tensor.shape != expected:
                    raise ValueError(
                        f"{official}: expected {expected}, got {tensor.shape}"
                    )
                module_name, parameter_name = ours.rsplit(".", 1)
                module = model.get_submodule(module_name)
                setattr(
                    module,
                    parameter_name,
                    nn.Parameter(tensor.to(device=device, dtype=dtype)),
                )

    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Some model parameters were not loaded")
    return model.eval()

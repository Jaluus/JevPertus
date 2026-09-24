"""Apertus V1.5 text checkpoint loading and parameter-name mapping."""

import json
import os

import torch
import torch.nn as nn


def checkpoint_names(model):
    """Map our readable parameter names to the official V1.5 checkpoint names."""
    names = {
        "input_layer.weight": "model.language_model.embed_tokens.weight",
        "final_norm.weight": "model.language_model.norm.weight",
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
                f"transformer_blocks.{i}.{ours}": f"model.language_model.layers.{i}.{official}"
                for ours, official in block.items()
            }
        )
    return names


def load_pretrained(
    model_class,
    model_id="swiss-ai/Apertus-v1.5-8B",
    *,
    device="cpu",
    dtype=torch.bfloat16,
    cache_dir=None,
    revision="main",
):
    """Load an official unquantized checkpoint from Hugging Face or a folder.

    Requires `pip install huggingface_hub safetensors` and Hugging Face access
    to the gated V1.5 repository. Uses your saved HF login or HF_TOKEN.
    Only text tensors are loaded; image/audio tokenizers are skipped.
    """
    from safetensors import safe_open

    if not dtype.is_floating_point or torch.device(device).type == "meta":
        raise ValueError("Use a floating-point dtype and a real device")
    folder = os.fspath(model_id)
    if not os.path.isdir(folder):
        from huggingface_hub import snapshot_download

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
    c = config["text_config"]
    rope = c["rope_parameters"]
    if (
        config.get("quantization_config")
        or c.get("quantization_config")
        or config.get("model_type") != "apertus1p5"
        or c.get("model_type") != "apertus1p5_text"
        or c.get("tie_word_embeddings", config.get("tie_word_embeddings", False))
        or c.get("attention_bias", False)
        or c.get("mlp_bias", False)
        or c.get("post_norm", False)
        or not c.get("qk_norm", True)
        or c["hidden_act"] != "xielu"
        or c["rms_norm_eps"] != 1e-5
        or rope["rope_type"] != "llama3"
        or rope["original_max_position_embeddings"] != 8192
        or rope["low_freq_factor"] != 1.0
        or rope["high_freq_factor"] != 4.0
        or rope["factor"] < 1
    ):
        raise ValueError(
            "Checkpoint architecture is not supported by this text implementation"
        )

    # Meta tensors allocate no storage: pretrained tensors replace them below.
    with torch.device("meta"):
        model = model_class(
            vocab_size=c["vocab_size"],
            embed_dim=c["hidden_size"],
            hidden_dim=c["intermediate_size"],
            context_len=c["max_position_embeddings"],
            num_heads=c["num_attention_heads"],
            num_kv_groups=c["num_key_value_heads"],
            head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
            num_attn_blocks=c["num_hidden_layers"],
            rope_theta=rope["rope_theta"],
            rope_factor=rope["factor"],
            output_vocab_size=c.get("output_vocab_size"),
        )
    names = checkpoint_names(model)
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
            from huggingface_hub import hf_hub_download

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

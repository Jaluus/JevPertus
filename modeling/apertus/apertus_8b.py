"""Self-contained PyTorch Apertus 8B with an Apertus V1.5 text-weight loader.

Defaults: https://huggingface.co/swiss-ai/Apertus-8B-2509/blob/main/config.json
"""

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(
        self,
        head_dim,
        theta=12_000_000,
        factor=8.0,
    ):
        super().__init__()

        self.rotary_dim = head_dim
        self.theta = theta
        self.factor = factor

    def _scaled_inverse_frequencies(self, device):
        dimensions = torch.arange(
            0, self.rotary_dim, 2, device=device, dtype=torch.float32
        )
        inv_freq = self.theta ** (-dimensions / self.rotary_dim)

        # Apertus uses Llama-3 scaling of the original 8192-token context,
        # with interpolation between the low/high frequency factors 1 and 4.
        wavelength = 2 * math.pi / inv_freq
        blend = ((8192 / wavelength - 1) / 3).clamp(0, 1)
        return inv_freq * (blend + (1 - blend) / self.factor)

    def forward(self, x):
        """Rotate inputs shaped (batch, heads, tokens, head_dim)."""
        # Each token position gets one angle per pair of rotary dimensions.
        positions = torch.arange(x.shape[2], device=x.device, dtype=torch.float32)
        inv_freq = self._scaled_inverse_frequencies(x.device)
        angles = torch.outer(positions, inv_freq)

        # Apertus pairs the first half of each head with the second half.
        angles = einops.repeat(angles, "t d -> t (halves d)", halves=2)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        first, second = x.chunk(2, dim=-1)
        rotated_half = torch.cat((-second, first), dim=-1)
        return x * cos + rotated_half * sin


class GroupQueryAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_groups,
        head_dim,
        rope_theta=12_000_000,
        rope_factor=8.0,
    ):
        super().__init__()
        if num_heads <= 0 or num_kv_groups <= 0 or num_heads % num_kv_groups:
            raise ValueError("num_heads must be a positive multiple of num_kv_groups")

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim

        self.w_q = nn.Linear(embed_dim, self.hidden_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, num_kv_groups * head_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, num_kv_groups * head_dim, bias=False)
        self.w_o = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-5)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-5)
        self.rope = RoPE(head_dim, rope_theta, rope_factor)

    def forward(self, x):

        # We start with (batch, tokens, embed_dim) and project to queries, keys, values.
        # We then reshape to (batch, heads, tokens, head_dim) for queries and (batch, kv_groups, tokens, head_dim) for keys and values.
        # This allows us to compute attention with grouped keys and values, where each group attends to multiple heads.
        q = einops.rearrange(self.w_q(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = einops.rearrange(self.w_k(x), "b t (g d) -> b g t d", g=self.num_kv_groups)
        v = einops.rearrange(self.w_v(x), "b t (g d) -> b g t d", g=self.num_kv_groups)

        # Apertus applies RMS normalization to the queries and keys before computing attention.
        # This is a form of pre-normalization that can help stabilize training and improve performance.
        # You can find it under the name "qk_norm" or "sandwich_norm".
        # After that RoPE is applied to the queries and keys, which encodes positional information into the attention mechanism.
        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))

        # Finally, we compute the scaled dot-product attention using PyTorch's built-in function.
        # We could be fancy and implement it ourselves, but this is more efficient and takes advantage of any optimizations in the library.
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            enable_gqa=True,
        )

        # Merge heads into (batch, tokens, hidden_dim) for the output projection.
        context = einops.rearrange(context, "b h t d -> b t (h d)")

        # Now we project the context back to the original embedding dimension using the output linear layer.
        output = self.w_o(context)
        return output


class XIELU(nn.Module):
    def __init__(self):
        super().__init__()
        # Each layer learns two scalars, constrained through softplus.
        self.alpha_p = nn.Parameter(torch.tensor([math.log(math.expm1(0.8))]))
        self.alpha_n = nn.Parameter(torch.tensor([math.log(math.expm1(0.3))]))

    def forward(self, x):
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = 0.5 + F.softplus(self.alpha_n)
        positive = alpha_p * x.square() + 0.5 * x
        negative = alpha_n * (x.clamp(max=-1e-6).expm1() - x) + 0.5 * x
        return torch.where(x > 0, positive, negative)


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.up = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.activation = XIELU()
        self.down = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        return self.down(self.activation(self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        attention: GroupQueryAttention,
    ):
        super().__init__()
        self.rms_norm1 = nn.RMSNorm(embed_dim, eps=1e-5)
        self.attention = attention
        self.rms_norm2 = nn.RMSNorm(embed_dim, eps=1e-5)
        self.ff = FeedForwardNetwork(embed_dim, hidden_dim)

    def forward(self, x):
        x = x + self.attention(self.rms_norm1(x))
        x = x + self.ff(self.rms_norm2(x))
        return x


class ApertusModel(nn.Module):
    def __init__(
        self,
        vocab_size=131072,
        embed_dim=4096,
        hidden_dim=21504,
        context_len=65536,
        num_heads=32,
        num_kv_groups=8,
        head_dim=128,
        num_attn_blocks=32,
        rope_theta=12_000_000,
        rope_factor=8.0,
        output_vocab_size=None,
    ):
        super().__init__()
        self.context_len = context_len
        self.vocab_size = vocab_size

        self.input_layer = nn.Embedding(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    hidden_dim,
                    GroupQueryAttention(
                        embed_dim,
                        num_heads,
                        num_kv_groups,
                        head_dim,
                        rope_theta,
                        rope_factor,
                    ),
                )
                for _ in range(num_attn_blocks)
            ]
        )

        self.final_norm = nn.RMSNorm(embed_dim, eps=1e-5)

        self.output_layer = nn.Linear(
            embed_dim,
            output_vocab_size or vocab_size,
            bias=False,
        )

    def checkpoint_names(self):
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
        for i in range(len(self.transformer_blocks)):
            names.update(
                {
                    f"transformer_blocks.{i}.{ours}": f"model.language_model.layers.{i}.{official}"
                    for ours, official in block.items()
                }
            )
        return names

    @classmethod
    def from_pretrained(
        cls,
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
        import json
        from pathlib import Path

        from safetensors import safe_open

        if not dtype.is_floating_point or torch.device(device).type == "meta":
            raise ValueError("Use a floating-point dtype and a real device")
        folder = Path(model_id)
        if not folder.is_dir():
            from huggingface_hub import snapshot_download

            # Resolve metadata first; use its immutable revision for the shards.
            folder = Path(
                snapshot_download(
                    repo_id=str(model_id),
                    revision=revision,
                    cache_dir=cache_dir,
                    allow_patterns=["config.json", "model.safetensors.index.json"],
                )
            )
            revision = folder.name

        config = json.loads((folder / "config.json").read_text())
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
            model = cls(
                vocab_size=c["vocab_size"],
                embed_dim=c["hidden_size"],
                hidden_dim=c["intermediate_size"],
                context_len=c["max_position_embeddings"],
                num_heads=c["num_attention_heads"],
                num_kv_groups=c["num_key_value_heads"],
                head_dim=c.get(
                    "head_dim", c["hidden_size"] // c["num_attention_heads"]
                ),
                num_attn_blocks=c["num_hidden_layers"],
                rope_theta=rope["rope_theta"],
                rope_factor=rope["factor"],
                output_vocab_size=c.get("output_vocab_size"),
            )
        names = model.checkpoint_names()
        index = folder / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
        else:
            weight_map = {name: "model.safetensors" for name in names.values()}
        missing = set(names.values()) - weight_map.keys()
        if missing:
            raise ValueError(f"Checkpoint is missing text weights: {sorted(missing)}")

        shards = {}
        for ours, official in names.items():
            shards.setdefault(weight_map[official], []).append((ours, official))
        for filename, entries in shards.items():
            path = folder / filename
            if not path.exists() and not Path(model_id).is_dir():
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

    def forward(self, x, mask=None):
        """Token IDs (batch, tokens) -> logits (batch, tokens, vocab_size)."""
        logits = self.output_layer(self.partial_forward(x, mask))
        # V1.5's input-only image/audio IDs have no output-head rows.
        return F.pad(
            logits,
            (0, self.vocab_size - logits.shape[-1]),
            value=torch.finfo(logits.dtype).min,
        )

    def partial_forward(self, x, mask=None):
        """Token IDs (batch, tokens) -> final normalized hidden states.

        Optional mask: (batch, tokens), 1 for real tokens and 0 for right padding.
        Only right padding is supported; callers must ignore padded outputs.
        Each call recomputes the entire sequence; there is no KV cache.
        """

        if mask is not None:
            mask = mask.to(device=x.device, dtype=torch.bool)

        x = self.input_layer(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.final_norm(x)

"""Self-contained PyTorch Apertus 8B with an Apertus V1.5 text-weight loader.

Defaults: https://huggingface.co/swiss-ai/Apertus-8B-2509/blob/main/config.json
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.weight


class RoPE(nn.Module):
    def __init__(self, head_dim, theta=12_000_000, factor=8.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be positive and even")
        self.rotary_dim = head_dim
        self.theta = theta
        self.factor = factor

    def forward(self, x):
        # x: (batch, heads, tokens, head_dim). Apertus rotates the full head.
        positions = torch.arange(x.shape[2], device=x.device, dtype=torch.float32)
        frequencies = torch.arange(
            0, self.rotary_dim, 2, device=x.device, dtype=torch.float32
        )
        inv_freq = self.theta ** (-frequencies / self.rotary_dim)
        # Apertus uses Llama-3 scaling of the original 8192-token context,
        # with interpolation between the low/high frequency factors 1 and 4.
        wavelength = 2 * math.pi / inv_freq
        blend = ((8192 / wavelength - 1) / 3).clamp(0, 1)
        inv_freq = inv_freq * (blend + (1 - blend) / self.factor)
        angles = positions[:, None] * inv_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

        first, second = x.chunk(2, dim=-1)
        rotated = x * cos + torch.cat((-second, first), dim=-1) * sin
        return rotated


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
        self.group_size = num_heads // num_kv_groups
        self.hidden_dim = num_heads * head_dim

        self.w_q = nn.Linear(embed_dim, self.hidden_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, num_kv_groups * head_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, num_kv_groups * head_dim, bias=False)
        self.w_o = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = RoPE(head_dim, rope_theta, rope_factor)

    def forward(self, x, mask=None):
        batch_size, num_tokens, _ = x.shape
        q = self.w_q(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        k = self.w_k(x).view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)
        v = self.w_v(x).view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)

        q = self.rope(self.q_norm(q).transpose(1, 2))
        k = self.rope(self.k_norm(k).transpose(1, 2))
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.transpose(1, 2).repeat_interleave(self.group_size, dim=1)

        allowed = torch.ones(
            num_tokens, num_tokens, device=x.device, dtype=torch.bool
        ).tril()
        if mask is not None:
            allowed = allowed & mask[:, None, None, :]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.float().masked_fill(~allowed, float("-inf"))
        # Fully padded rows have no keys: return zero attention instead of NaNs.
        has_keys = allowed.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~has_keys, 0)
        weights = F.softmax(scores, dim=-1).masked_fill(~allowed, 0).to(q.dtype)
        context = (
            (weights @ v)
            .transpose(1, 2)
            .reshape(batch_size, num_tokens, self.hidden_dim)
        )
        return self.w_o(context)


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
    def __init__(self, embed_dim, hidden_dim, attention):
        super().__init__()
        self.rms_norm1 = RMSNorm(embed_dim)
        self.attention = attention
        self.rms_norm2 = RMSNorm(embed_dim)
        self.ff = FeedForwardNetwork(embed_dim, hidden_dim)

    def forward(self, x, mask=None):
        x = x + self.attention(self.rms_norm1(x), mask)
        x = x + self.ff(self.rms_norm2(x))
        return x


class OutputLayer(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.output_layer(x)


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
        if output_vocab_size is not None and not 0 < output_vocab_size <= vocab_size:
            raise ValueError("output_vocab_size must be between 1 and vocab_size")
        self.input_layer = InputLayer(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_attn_blocks):
            attention = GroupQueryAttention(
                embed_dim,
                num_heads,
                num_kv_groups,
                head_dim,
                rope_theta,
                rope_factor,
            )
            self.transformer_blocks.append(
                TransformerBlock(embed_dim, hidden_dim, attention)
            )

        self.final_norm = RMSNorm(embed_dim)
        # Apertus has separate token embedding and output weights.
        self.output_layer = OutputLayer(embed_dim, output_vocab_size or vocab_size)

    def checkpoint_names(self):
        """Map our readable parameter names to the official V1.5 checkpoint names."""
        names = {
            "input_layer.embedding.weight": "model.language_model.embed_tokens.weight",
            "final_norm.weight": "model.language_model.norm.weight",
            "output_layer.output_layer.weight": "lm_head.weight",
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

        Optional mask: (batch, tokens), 1 for real tokens and 0 for padding.
        Each call recomputes the entire sequence; there is no KV cache.
        """
        if x.ndim != 2 or not 0 < x.shape[1] <= self.context_len:
            raise ValueError(
                "Expected token IDs of shape (batch, tokens), with 1 <= tokens <= context_len"
            )
        if mask is not None:
            if mask.shape != x.shape:
                raise ValueError(
                    "mask must have the same (batch, tokens) shape as the input"
                )
            mask = mask.to(device=x.device, dtype=torch.bool)

        x = self.input_layer(x)
        for block in self.transformer_blocks:
            x = block(x, mask)
        return self.final_norm(x)

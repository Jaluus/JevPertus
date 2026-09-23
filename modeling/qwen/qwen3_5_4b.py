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
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Qwen3.5 learns an offset from 1, rather than the scale directly.
        self.weight = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (normalized * (1 + self.weight.float())).to(x.dtype)


class GatedRMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # DeltaNet's output norm uses a direct scale and a SiLU gate.
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x, gate):
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        normalized = normalized.to(x.dtype) * self.weight
        return (normalized * F.silu(gate.float())).to(x.dtype)


class RoPE(nn.Module):
    def __init__(self, head_dim, theta=10_000_000, partial_rotary_factor=0.25):
        super().__init__()
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        if not 0 < self.rotary_dim <= head_dim or self.rotary_dim % 2:
            raise ValueError(
                "The rotary dimension must be positive, even, and <= head_dim"
            )
        self.theta = theta

    def forward(self, x):
        # x: (batch, heads, tokens, head_dim). For text, all MRoPE axes
        # have the same positions, reducing to ordinary partial RoPE.
        positions = torch.arange(x.shape[2], device=x.device, dtype=torch.float32)
        frequencies = torch.arange(
            0, self.rotary_dim, 2, device=x.device, dtype=torch.float32
        )
        inv_freq = self.theta ** (-frequencies / self.rotary_dim)
        angles = positions[:, None] * inv_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

        rotating, unchanged = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        first, second = rotating.chunk(2, dim=-1)
        rotated = rotating * cos + torch.cat((-second, first), dim=-1) * sin
        return torch.cat((rotated, unchanged), dim=-1)


class GroupQueryAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_groups,
        head_dim,
        rope_theta=10_000_000,
        partial_rotary_factor=0.25,
    ):
        super().__init__()
        if num_heads <= 0 or num_kv_groups <= 0 or num_heads % num_kv_groups:
            raise ValueError("num_heads must be a positive multiple of num_kv_groups")
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.group_size = num_heads // num_kv_groups
        self.hidden_dim = num_heads * head_dim

        # Each query head projects both a query and an output gate.
        self.w_q = nn.Linear(embed_dim, 2 * self.hidden_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, num_kv_groups * head_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, num_kv_groups * head_dim, bias=False)
        self.w_o = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = RoPE(head_dim, rope_theta, partial_rotary_factor)

    def forward(self, x, mask=None):
        batch_size, num_tokens, _ = x.shape
        # W_q is of shape (embed_dim, 2 * num_heads * head_dim), producing both a query and a gate.
        # x is of shape (batch_size, num_tokens, embed_dim)
        # So the output of W_q(x) is of shape (batch_size, num_tokens, 2 * num_heads * head_dim)
        # The chunk operation splits this into two tensors of shape (batch_size, num_tokens, num_heads * head_dim)
        q, gate = (
            self.w_q(x)
            .view(batch_size, num_tokens, self.num_heads, 2 * self.head_dim)
            .chunk(2, -1)
        )
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
        # Fully padded query rows should produce zero attention, not NaNs.
        has_keys = allowed.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~has_keys, 0)
        weights = F.softmax(scores, dim=-1).masked_fill(~allowed, 0).to(q.dtype)
        context = (
            (weights @ v)
            .transpose(1, 2)
            .reshape(batch_size, num_tokens, self.hidden_dim)
        )

        context = (
            context * gate.reshape(batch_size, num_tokens, self.hidden_dim).sigmoid()
        )
        return self.w_o(context)


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        conv_kernel_size=4,
    ):
        super().__init__()
        if num_value_heads % num_key_heads:
            raise ValueError("num_value_heads must be divisible by num_key_heads")
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.key_dim = num_key_heads * key_head_dim
        self.value_dim = num_value_heads * value_head_dim
        conv_dim = 2 * self.key_dim + self.value_dim

        self.w_qkv = nn.Linear(embed_dim, conv_dim, bias=False)
        self.w_z = nn.Linear(embed_dim, self.value_dim, bias=False)
        self.w_beta = nn.Linear(embed_dim, num_value_heads, bias=False)
        self.w_decay = nn.Linear(embed_dim, num_value_heads, bias=False)
        self.conv = nn.Conv1d(
            conv_dim,
            conv_dim,
            conv_kernel_size,
            groups=conv_dim,
            padding=conv_kernel_size - 1,
            bias=False,
        )
        self.A_log = nn.Parameter(torch.empty(num_value_heads).uniform_(0.01, 16).log())
        self.dt_bias = nn.Parameter(torch.ones(num_value_heads))
        self.norm = GatedRMSNorm(value_head_dim)
        self.w_o = nn.Linear(self.value_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        batch_size, num_tokens, _ = x.shape
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # A depthwise causal convolution mixes each Q/K/V channel with its past.
        qkv = self.w_qkv(x).transpose(1, 2)
        qkv = F.silu(self.conv(qkv)[..., :num_tokens]).transpose(1, 2)
        q, k, v = qkv.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
        q = q.reshape(
            batch_size, num_tokens, self.num_key_heads, self.key_head_dim
        ).float()
        k = k.reshape(
            batch_size, num_tokens, self.num_key_heads, self.key_head_dim
        ).float()
        v = v.reshape(
            batch_size, num_tokens, self.num_value_heads, self.value_head_dim
        ).float()

        # L2 normalization here is different from the full-attention RMSNorm.
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        repeats = self.num_value_heads // self.num_key_heads
        q = q.repeat_interleave(repeats, dim=2) / math.sqrt(self.key_head_dim)
        k = k.repeat_interleave(repeats, dim=2)

        beta = self.w_beta(x).sigmoid().float()
        log_decay = -self.A_log.float().exp() * F.softplus(
            self.w_decay(x).float() + self.dt_bias.float()
        )
        decay = log_decay.exp()
        state = torch.zeros(
            batch_size,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
            device=x.device,
            dtype=torch.float32,
        )
        outputs = []
        for t in range(num_tokens):
            # Decay old memory, then write only the error in its value prediction.
            state = state * decay[:, t, :, None, None]
            key = k[:, t].unsqueeze(-1)
            prediction = (state * key).sum(dim=-2)
            delta = beta[:, t, :, None] * (v[:, t] - prediction)
            state = state + key * delta.unsqueeze(-2)
            outputs.append((state * q[:, t].unsqueeze(-1)).sum(dim=-2))

        context = torch.stack(outputs, dim=1).to(qkv.dtype)
        gate = self.w_z(x).reshape(
            batch_size, num_tokens, self.num_value_heads, self.value_head_dim
        )
        context = self.norm(context, gate).reshape(
            batch_size, num_tokens, self.value_dim
        )
        return self.w_o(context)


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


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


class Qwen3_5Model(nn.Module):
    def __init__(
        self,
        vocab_size=248320,
        embed_dim=2560,
        hidden_dim=9216,
        context_len=262144,
        num_heads=16,
        num_kv_groups=4,
        head_dim=256,
        num_attn_blocks=32,
        full_attention_interval=4,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        conv_kernel_size=4,
        rope_theta=10_000_000,
        partial_rotary_factor=0.25,
    ):
        super().__init__()
        if full_attention_interval < 1:
            raise ValueError("full_attention_interval must be positive")
        self.context_len = context_len
        self.input_layer = InputLayer(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList()
        for layer in range(num_attn_blocks):
            if (layer + 1) % full_attention_interval == 0:
                attention = GroupQueryAttention(
                    embed_dim,
                    num_heads,
                    num_kv_groups,
                    head_dim,
                    rope_theta,
                    partial_rotary_factor,
                )
            else:
                attention = GatedDeltaNet(
                    embed_dim,
                    linear_num_key_heads,
                    linear_num_value_heads,
                    linear_key_head_dim,
                    linear_value_head_dim,
                    conv_kernel_size,
                )
            self.transformer_blocks.append(
                TransformerBlock(embed_dim, hidden_dim, attention)
            )

        self.final_norm = RMSNorm(embed_dim)
        self.output_layer = OutputLayer(embed_dim, vocab_size)
        self.output_layer.output_layer.weight = self.input_layer.embedding.weight

    def checkpoint_names(self):
        """Map our readable parameter names to the official checkpoint names."""
        names = {
            "input_layer.embedding.weight": "model.language_model.embed_tokens.weight",
            "final_norm.weight": "model.language_model.norm.weight",
        }
        common = {
            "rms_norm1.weight": "input_layernorm.weight",
            "rms_norm2.weight": "post_attention_layernorm.weight",
            "ff.gate.weight": "mlp.gate_proj.weight",
            "ff.up.weight": "mlp.up_proj.weight",
            "ff.down.weight": "mlp.down_proj.weight",
        }
        linear = {
            "w_qkv.weight": "in_proj_qkv.weight",
            "w_z.weight": "in_proj_z.weight",
            "w_beta.weight": "in_proj_b.weight",
            "w_decay.weight": "in_proj_a.weight",
            "conv.weight": "conv1d.weight",
            "A_log": "A_log",
            "dt_bias": "dt_bias",
            "norm.weight": "norm.weight",
            "w_o.weight": "out_proj.weight",
        }
        full = {
            "w_q.weight": "q_proj.weight",
            "w_k.weight": "k_proj.weight",
            "w_v.weight": "v_proj.weight",
            "w_o.weight": "o_proj.weight",
            "q_norm.weight": "q_norm.weight",
            "k_norm.weight": "k_norm.weight",
        }
        for i, block in enumerate(self.transformer_blocks):
            ours, official = (
                f"transformer_blocks.{i}.",
                f"model.language_model.layers.{i}.",
            )
            names.update({ours + k: official + v for k, v in common.items()})
            is_linear = isinstance(block.attention, GatedDeltaNet)
            prefix = "linear_attn." if is_linear else "self_attn."
            names.update(
                {
                    ours + "attention." + k: official + prefix + v
                    for k, v in (linear if is_linear else full).items()
                }
            )
        # The output projection shares the embedding parameter; load it only once.
        return names

    @classmethod
    def from_pretrained(
        cls,
        model_id="Qwen/Qwen3.5-4B",
        *,
        device="cpu",
        dtype=torch.bfloat16,
        cache_dir=None,
        revision="main",
    ):
        """Load an official unquantized checkpoint from Hugging Face or a folder.

        Requires `pip install huggingface_hub safetensors`. Only text tensors are
        materialized, though downloaded shards also contain vision/MTP weights.
        The 4B text weights occupy about 8.4 GB in bfloat16, before activations.
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
        interval = c["full_attention_interval"]
        schedule = (
            [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(c["num_hidden_layers"])
            ]
            if interval > 0
            else []
        )
        if (
            config.get("quantization_config")
            or c.get("quantization_config")
            or c.get("model_type") != "qwen3_5_text"
            or not c.get(
                "tie_word_embeddings", config.get("tie_word_embeddings", False)
            )
            or c.get("attention_bias", False)
            or not c.get("attn_output_gate", True)
            or c["hidden_act"] != "silu"
            or c["rms_norm_eps"] != 1e-6
            or rope["rope_type"] != "default"
            or not schedule
            or c.get("layer_types", schedule) != schedule
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
                head_dim=c["head_dim"],
                num_attn_blocks=c["num_hidden_layers"],
                full_attention_interval=interval,
                linear_num_key_heads=c["linear_num_key_heads"],
                linear_num_value_heads=c["linear_num_value_heads"],
                linear_key_head_dim=c["linear_key_head_dim"],
                linear_value_head_dim=c["linear_value_head_dim"],
                conv_kernel_size=c["linear_conv_kernel_dim"],
                rope_theta=rope["rope_theta"],
                partial_rotary_factor=rope["partial_rotary_factor"],
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

        model.output_layer.output_layer.weight = model.input_layer.embedding.weight
        if any(p.is_meta for p in model.parameters()):
            raise RuntimeError("Some model parameters were not loaded")
        return model.eval()

    def forward(self, x, mask=None):
        """Token IDs (batch, tokens) -> logits (batch, tokens, vocab_size)."""
        return self.output_layer(self.partial_forward(x, mask))

    def partial_forward(self, x, mask=None):
        """Token IDs (batch, tokens) -> final normalized hidden states.

        Optional mask: (batch, tokens), 1 for real tokens and 0 for padding.
        Use contiguous left/right padding; arbitrary attention masks and packed
        sequences are not supported by this simple recurrent implementation.
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

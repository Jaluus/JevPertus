"""Self-contained PyTorch Apertus 8B; randomly initialized, without a weight loader.

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
    def __init__(self, head_dim, theta=12_000_000):
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be positive and even")
        self.rotary_dim = head_dim
        self.theta = theta

    def forward(self, x):
        # x: (batch, heads, tokens, head_dim). Apertus rotates the full head.
        positions = torch.arange(x.shape[2], device=x.device, dtype=torch.float32)
        frequencies = torch.arange(
            0, self.rotary_dim, 2, device=x.device, dtype=torch.float32
        )
        inv_freq = self.theta ** (-frequencies / self.rotary_dim)
        # Apertus uses Llama-3 scaling: 8x extension of the 8192-token context,
        # with interpolation between the low/high frequency factors 1 and 4.
        wavelength = 2 * math.pi / inv_freq
        blend = ((8192 / wavelength - 1) / 3).clamp(0, 1)
        inv_freq = inv_freq * (blend + (1 - blend) / 8)
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
        self.rope = RoPE(head_dim, rope_theta)

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
    ):
        super().__init__()
        self.context_len = context_len
        self.input_layer = InputLayer(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_attn_blocks):
            attention = GroupQueryAttention(
                embed_dim,
                num_heads,
                num_kv_groups,
                head_dim,
                rope_theta,
            )
            self.transformer_blocks.append(
                TransformerBlock(embed_dim, hidden_dim, attention)
            )

        self.final_norm = RMSNorm(embed_dim)
        # Apertus has separate token embedding and output weights.
        self.output_layer = OutputLayer(embed_dim, vocab_size)

    def forward(self, x, mask=None):
        """Token IDs (batch, tokens) -> logits (batch, tokens, vocab_size)."""
        return self.output_layer(self.partial_forward(x, mask))

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

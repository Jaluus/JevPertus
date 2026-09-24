"""Implement the Apertus text backbone and its transformer components."""

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    """Apply rotary position embeddings with Apertus frequency scaling."""

    def __init__(
        self,
        head_dim,
        theta=12_000_000,
        factor=8.0,
    ):
        """Initialize rotary embedding settings and nonpersistent caches.

        Args:
            head_dim: Even number of dimensions in each attention head.
            theta: Base used to compute rotary inverse frequencies.
            factor: Scaling factor applied to low rotary frequencies.
        """
        super().__init__()

        self.rotary_dim = head_dim
        self.theta = theta
        self.factor = factor
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _scaled_inverse_frequencies(self, device):
        """Compute the scaled inverse frequencies for rotary dimension pairs.

        Args:
            device: Device on which to create the frequency tensor.

        Returns:
            Float32 inverse frequencies of shape (head_dim // 2,).
        """
        dimensions = torch.arange(
            0, self.rotary_dim, 2, device=device, dtype=torch.float32
        )
        inv_freq = self.theta ** (-dimensions / self.rotary_dim)

        # Apertus uses Llama-3 scaling of the original 8192-token context,
        # with interpolation between the low/high frequency factors 1 and 4.
        wavelength = 2 * torch.pi / inv_freq
        blend = ((8192 / wavelength - 1) / 3).clamp(0, 1)
        return inv_freq * (blend + (1 - blend) / self.factor)

    @torch.no_grad()
    def precompute(self, length, device, dtype):
        """Cache rotary sine and cosine tables outside the model state dict.

        Args:
            length: Number of token positions to cache.
            device: Device on which to create the tables.
            dtype: Dtype of the cached tables.
        """
        # Each token position gets one angle per pair of rotary dimensions.
        positions = torch.arange(length, device=device, dtype=torch.float32)
        inv_freq = self._scaled_inverse_frequencies(device)
        angles = torch.outer(positions, inv_freq)

        # Apertus pairs the first half of each head with the second half.
        angles = einops.repeat(angles, "t d -> t (halves d)", halves=2)
        self.cos_cached = angles.cos().to(dtype)
        self.sin_cached = angles.sin().to(dtype)

    def forward(self, x):
        """Rotate attention states using their token positions.

        Args:
            x: Tensor of shape (batch, heads, tokens, head_dim).

        Returns:
            Rotated states with the same shape, device, and dtype as x.
        """
        length = x.shape[2]
        if (
            self.cos_cached is None
            or self.cos_cached.shape[0] < length
            or self.cos_cached.device != x.device
            or self.cos_cached.dtype != x.dtype
        ):
            self.precompute(length, x.device, x.dtype)
        cos = self.cos_cached[:length]
        sin = self.sin_cached[:length]
        first, second = x.chunk(2, dim=-1)
        rotated_half = torch.cat((-second, first), dim=-1)
        return x * cos + rotated_half * sin


class GroupQueryAttention(nn.Module):
    """Apply causal grouped-query attention with query and key normalization."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_groups,
        head_dim,
        rope_theta=12_000_000,
        rope_factor=8.0,
    ):
        """Initialize attention projections, normalization, and rotary embeddings.

        Args:
            embed_dim: Width of input and output token states.
            num_heads: Number of query attention heads.
            num_kv_groups: Number of shared key and value heads.
            head_dim: Even number of dimensions per attention head.
            rope_theta: Base used for rotary inverse frequencies.
            rope_factor: Low-frequency rotary scaling factor.

        Raises:
            ValueError: Head counts are nonpositive or num_heads is not divisible by
                num_kv_groups.
        """
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
        """Compute causal attention over token states.

        Args:
            x: Token states of shape (batch, tokens, embed_dim).

        Returns:
            Updated token states with the same shape as x.
        """
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
    """Apply a learnable, elementwise xIELU activation."""

    def __init__(self):
        """Initialize the two learned activation parameters."""
        super().__init__()
        # Each layer learns two scalars, constrained through softplus.
        self.alpha_p = nn.Parameter(torch.tensor([0.8]).expm1().log())
        self.alpha_n = nn.Parameter(torch.tensor([0.3]).expm1().log())

    def forward(self, x):
        """Apply the positive and negative branches of xIELU.

        Args:
            x: Input tensor of any shape.

        Returns:
            Activated tensor with the same shape as x.
        """
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = 0.5 + F.softplus(self.alpha_n)
        positive = alpha_p * x.square() + 0.5 * x
        negative = alpha_n * (x.clamp(max=-1e-6).expm1() - x) + 0.5 * x
        return torch.where(x > 0, positive, negative)


class FeedForwardNetwork(nn.Module):
    """Transform token states through an xIELU feed-forward network."""

    def __init__(self, embed_dim, hidden_dim):
        """Initialize the expansion and contraction projections.

        Args:
            embed_dim: Width of input and output token states.
            hidden_dim: Width of the intermediate activation.
        """
        super().__init__()
        self.up = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.activation = XIELU()
        self.down = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        """Apply the feed-forward transformation to token states.

        Args:
            x: Token states of shape (batch, tokens, embed_dim).

        Returns:
            Updated token states with the same shape as x.
        """
        return self.down(self.activation(self.up(x)))


class TransformerBlock(nn.Module):
    """Apply pre-normalized attention and feed-forward residual updates."""

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        attention: GroupQueryAttention,
    ):
        """Initialize the residual block and its normalization layers.

        Args:
            embed_dim: Width of input and output token states.
            hidden_dim: Intermediate width of the feed-forward network.
            attention: Attention module compatible with embed_dim.
        """
        super().__init__()
        self.rms_norm1 = nn.RMSNorm(embed_dim, eps=1e-5)
        self.attention = attention
        self.rms_norm2 = nn.RMSNorm(embed_dim, eps=1e-5)
        self.ff = FeedForwardNetwork(embed_dim, hidden_dim)

    def forward(self, x):
        """Apply attention and feed-forward residual updates.

        Args:
            x: Token states of shape (batch, tokens, embed_dim).

        Returns:
            Updated token states with the same shape as x.
        """
        x = x + self.attention(self.rms_norm1(x))
        x = x + self.ff(self.rms_norm2(x))
        return x


class ApertusModel(nn.Module):
    """Provide causal text logits and normalized token hidden states.

    Attributes:
        context_len: Configured context length, stored as metadata without a
            length check.
        vocab_size: Input vocabulary size and final padded logit width.
    """

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
        """Initialize the text backbone and vocabulary projection.

        Args:
            vocab_size: Number of input token embeddings and final padded output
                logits.
            embed_dim: Width of token embeddings and hidden states.
            hidden_dim: Intermediate feed-forward width.
            context_len: Configured context length, stored as metadata.
            num_heads: Number of query attention heads per block.
            num_kv_groups: Number of shared key and value heads per block.
            head_dim: Even number of dimensions per attention head.
            num_attn_blocks: Number of transformer blocks.
            rope_theta: Base used for rotary inverse frequencies.
            rope_factor: Low-frequency rotary scaling factor.
            output_vocab_size: Optional output projection width, at most vocab_size.
                A falsy value uses vocab_size.

        Raises:
            ValueError: An attention block receives invalid head counts.
        """
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

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute vocabulary logits for each input token.

        Args:
            x: Token IDs of shape (batch, tokens) on the model device.
            mask: Optional validity mask of shape (batch, tokens). Only right
                padding is supported; the mask does not affect attention.

        Returns:
            Logits of shape (batch, tokens, vocab_size). Input-only vocabulary
            entries receive the minimum finite value of the logit dtype. Callers
            must ignore padded token outputs.
        """
        logits = self.output_layer(self.partial_forward(x, mask))
        # V1.5's input-only image/audio IDs have no output-head rows.
        return F.pad(
            logits,
            (0, self.vocab_size - logits.shape[-1]),
            value=torch.finfo(logits.dtype).min,
        )

    def partial_forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute final normalized hidden states for every input token.

        Only right-padded inputs are supported: causal attention keeps padding
        from affecting earlier real tokens. Callers must ignore padded outputs.
        Each call recomputes the entire sequence without a key/value cache.

        Args:
            x: Token IDs of shape (batch, tokens) on the model device.
            mask: Optional validity mask of shape (batch, tokens), true for real
                tokens. It is converted to boolean but is not applied to attention.

        Returns:
            Hidden states of shape (batch, tokens, embed_dim).
        """

        if mask is not None:
            mask = mask.to(device=x.device, dtype=torch.bool)

        x = self.input_layer(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.final_norm(x)

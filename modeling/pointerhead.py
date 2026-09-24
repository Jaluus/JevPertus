"""Score answer options using projected decision and option states."""

import torch.nn as nn


class PointerHead(nn.Module):
    """Compute scaled dot-product scores for a question's options."""

    def __init__(self, hidden_dim, pointer_dim=256):
        """Initialize the query and key projections.

        Args:
            hidden_dim: Width of the input hidden states.
            pointer_dim: Width of the query and key projections.
        """
        super().__init__()
        self.config = dict(hidden_dim=hidden_dim, pointer_dim=pointer_dim)
        self.query = nn.Linear(hidden_dim, pointer_dim)
        self.key = nn.Linear(hidden_dim, pointer_dim)
        self.scale = pointer_dim**-0.5

    def forward(self, decide, options):
        """Score each option against the decision state.

        Args:
            decide: Decision tensor of shape (hidden_dim,).
            options: Option tensor of shape (num_options, hidden_dim).

        Returns:
            Unnormalized logits of shape (num_options,). Inputs must match the
            projection weights in device and dtype.
        """
        return (self.key(options) @ self.query(decide)) * self.scale

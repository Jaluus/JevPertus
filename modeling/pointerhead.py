import torch
import torch.nn as nn


class PointerHead(nn.Module):
    def __init__(self, hidden_dim, pointer_dim=256):
        super().__init__()
        self.config = dict(hidden_dim=hidden_dim, pointer_dim=pointer_dim)
        self.query = nn.Linear(hidden_dim, pointer_dim)
        self.key = nn.Linear(hidden_dim, pointer_dim)
        self.scale = pointer_dim**-0.5

    def forward(self, decide, options):
        return (self.key(options) @ self.query(decide)) * self.scale

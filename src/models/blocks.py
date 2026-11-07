import torch
import torch.nn as nn


def group_count(channels):
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.norm1 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.activation = nn.GELU()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                nn.GroupNorm(group_count(out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, hidden):
        updated = self.activation(self.norm1(self.conv1(hidden)))
        updated = self.norm2(self.conv2(updated))
        return self.activation(updated + self.shortcut(hidden))


class AttentionResidual(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        if channels % heads != 0:
            raise RuntimeError("attention channels must be divisible by the number of heads")
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, tokens):
        updated, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return self.norm(tokens + updated)

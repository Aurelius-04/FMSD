import math

import torch
import torch.nn as nn

from src.models.blocks import AttentionResidual


class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=kernel // 2, bias=True)
                for kernel in kernels
            ]
        )
        self.projection = nn.Linear(out_channels * len(kernels), out_channels)
        self.activation = nn.GELU()

    def forward(self, sequence):
        parts = [self.activation(branch(sequence)) for branch in self.branches]
        fused = torch.cat(parts, dim=1).transpose(1, 2)
        return self.projection(fused).transpose(1, 2)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, channels, length):
        super().__init__()
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, channels, 2, dtype=torch.float32) * (-math.log(10000.0) / channels))
        encoding = torch.zeros(length, channels, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, sequence):
        return sequence + self.encoding[: sequence.size(1)].unsqueeze(0).to(dtype=sequence.dtype)


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        kernels = tuple(model_cfg["temporal_kernels"])
        for kernel in kernels:
            if kernel % 2 == 0:
                raise RuntimeError("temporal kernels must be odd")
        channels = model_cfg["temporal_channels"]
        latent = model_cfg["latent_dim"]
        blocks = []
        current = in_channels
        for _ in range(model_cfg["temporal_depth"]):
            blocks.append(MultiScaleTemporalBlock(current, channels, kernels))
            current = channels
        self.blocks = nn.ModuleList(blocks)
        self.channel_projection = nn.Linear(channels, latent)
        self.position = TemporalPositionalEncoding(latent, model_cfg["window_length"])
        self.attention_layers = nn.ModuleList(
            [AttentionResidual(latent, model_cfg["temporal_heads"]) for _ in range(model_cfg["temporal_layers"])]
        )

    def forward(self, sequence):
        hidden = sequence.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        tokens = self.channel_projection(hidden.transpose(1, 2))
        tokens = self.position(tokens)
        for layer in self.attention_layers:
            tokens = layer(tokens)
        return tokens.mean(dim=1)

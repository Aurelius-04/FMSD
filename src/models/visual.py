import torch.nn as nn

from src.models.blocks import ResidualBlock


class CanopyResidualEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        channels = list(model_cfg["visual_channels"])
        block_counts = list(model_cfg["visual_blocks"])
        if len(channels) != 3 or len(block_counts) != 3:
            raise RuntimeError("the canopy encoder uses three residual stages")
        stages = []
        in_channels = 3
        for out_channels, block_count in zip(channels, block_counts):
            stage = []
            for block_index in range(block_count):
                stride = 2 if block_index == 0 else 1
                stage.append(ResidualBlock(in_channels, out_channels, stride))
                in_channels = out_channels
            stages.append(nn.Sequential(*stage))
        self.stages = nn.ModuleList(stages)
        self.projection = nn.Linear(channels[-1], model_cfg["latent_dim"])

    def forward(self, canopy):
        hidden = canopy
        for stage in self.stages:
            hidden = stage(hidden)
        pooled = hidden.mean(dim=(2, 3))
        return self.projection(pooled)

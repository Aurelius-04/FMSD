import torch
import torch.nn as nn

from src.models.blocks import AttentionResidual
from src.models.temporal import MultiScaleTemporalEncoder
from src.models.visual import CanopyResidualEncoder


class StructuredProjection(nn.Module):
    def __init__(self, in_channels, hidden, dropout):
        super().__init__()
        self.first = nn.Linear(in_channels, hidden)
        self.second = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        hidden = self.activation(self.first(features))
        hidden = self.second(hidden)
        return self.dropout(self.activation(self.norm(hidden)))


class HeterogeneousMultimodalSensorEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        latent = model_cfg["latent_dim"]
        self.eps = float(model_cfg["epsilon"])
        self.environmental_encoder = MultiScaleTemporalEncoder(model_cfg["environmental_channels"], cfg)
        self.soil_encoder = MultiScaleTemporalEncoder(model_cfg["soil_channels"], cfg)
        self.canopy_encoder = CanopyResidualEncoder(cfg)
        self.structured_encoder = StructuredProjection(
            model_cfg["structured_channels"],
            model_cfg["structured_hidden"],
            model_cfg["structured_dropout"],
        )
        if model_cfg["structured_hidden"] != latent:
            raise RuntimeError("structured projection width must equal the fused latent dimension")
        self.modality_embed = nn.Parameter(torch.zeros(4, latent))
        self.cross_modal_layers = nn.ModuleList(
            [AttentionResidual(latent, model_cfg["cross_modal_heads"]) for _ in range(model_cfg["cross_modal_layers"])]
        )
        self.interaction_matrix = nn.Parameter(torch.eye(latent))
        self.modality_logit = nn.Parameter(torch.zeros(4))
        self.interaction_scale = nn.Parameter(torch.tensor(1.0))
        self.joint_projection = nn.Linear(latent * 4, latent)
        self.fusion_norm = nn.LayerNorm(latent)
        nn.init.normal_(self.modality_embed, std=0.02)

    def normalize_unit_variance(self, features):
        centered = features - features.mean(dim=-1, keepdim=True)
        variance = torch.mean(torch.square(centered), dim=-1, keepdim=True)
        return features / torch.sqrt(variance + self.eps)

    def fuse(self, tokens):
        transformed = torch.matmul(tokens, self.interaction_matrix.transpose(0, 1))
        affinity = torch.matmul(tokens, transformed.transpose(1, 2))
        norms = torch.linalg.vector_norm(tokens, ord=2, dim=-1, keepdim=True)
        interaction = affinity / (norms * norms.transpose(1, 2) + self.eps)
        off_diagonal = interaction.sum(dim=-1) - interaction.diagonal(dim1=-2, dim2=-1)
        gates = torch.softmax(self.modality_logit + self.interaction_scale * off_diagonal, dim=-1)
        gated = torch.sum(gates.unsqueeze(-1) * tokens, dim=1)
        flat = tokens.reshape(tokens.shape[0], -1)
        return self.fusion_norm(gated + self.joint_projection(flat))

    def forward(self, environmental, soil, canopy, structured):
        environmental_token = self.normalize_unit_variance(self.environmental_encoder(environmental))
        soil_token = self.normalize_unit_variance(self.soil_encoder(soil))
        canopy_token = self.normalize_unit_variance(self.canopy_encoder(canopy))
        structured_token = self.normalize_unit_variance(self.structured_encoder(structured))
        tokens = torch.stack((environmental_token, soil_token, canopy_token, structured_token), dim=1)
        tokens = tokens + self.modality_embed.unsqueeze(0)
        for layer in self.cross_modal_layers:
            tokens = layer(tokens)
        return self.fuse(tokens)

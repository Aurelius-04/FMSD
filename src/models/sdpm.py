import torch
import torch.nn as nn


class TaskHead(nn.Module):
    def __init__(self, in_channels, hidden):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, latent):
        return self.network(latent).squeeze(-1)


class SensorToDecisionModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        decision_cfg = cfg["decision"]
        latent = model_cfg["latent_dim"]
        layer = nn.TransformerEncoderLayer(
            d_model=latent,
            nhead=model_cfg["decision_heads"],
            dim_feedforward=model_cfg["decision_feedforward"],
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=model_cfg["decision_layers"],
            enable_nested_tensor=False,
        )
        self.yield_head = TaskHead(latent, model_cfg["head_hidden"])
        self.cost_head = TaskHead(latent, model_cfg["head_hidden"])
        self.return_head = TaskHead(latent, model_cfg["head_hidden"])
        self.return_mix = float(decision_cfg["omega_return"])
        self.margin_reference = float(decision_cfg["theta_base"])
        self.margin_slope = float(decision_cfg["kappa"])
        self.cost_epsilon = float(decision_cfg["cost_epsilon"])

    def forward(self, representation, price):
        latent = self.encoder(representation.unsqueeze(1)).squeeze(1)
        predicted_yield = self.yield_head(latent)
        predicted_cost = self.cost_head(latent)
        direct_return = self.return_head(latent)
        accounting_return = predicted_yield * price - predicted_cost
        net_return = self.return_mix * direct_return + (1.0 - self.return_mix) * accounting_return
        margin = net_return / predicted_cost.clamp_min(self.cost_epsilon)
        decision_score = torch.sigmoid(self.margin_slope * (margin - self.margin_reference))
        return {
            "yield": predicted_yield,
            "cost": predicted_cost,
            "direct_return": direct_return,
            "accounting_return": accounting_return,
            "net_return": net_return,
            "margin": margin,
            "decision_score": decision_score,
        }

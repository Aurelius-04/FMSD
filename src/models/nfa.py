import torch
import torch.nn as nn


class ClientScoringNetwork(nn.Module):
    def __init__(self, descriptor_dim, hidden):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.scorer = nn.Linear(hidden + 3, 1)

    def forward(self, descriptor, direction, shift, reliability):
        hidden = self.encoder(descriptor)
        features = torch.cat(
            (
                hidden,
                direction.unsqueeze(-1),
                -shift.unsqueeze(-1),
                reliability.unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.scorer(features).squeeze(-1)


def log_reliability(loss, gradient_variance, completeness, cfg):
    return (
        -float(cfg["nfa"]["alpha_loss"]) * loss
        -float(cfg["nfa"]["alpha_variance"]) * gradient_variance
        +float(cfg["nfa"]["alpha_completeness"]) * completeness
    )


def distribution_shift(mu, diag, mu_reference, diag_reference, beta):
    mean_term = torch.linalg.vector_norm(mu - mu_reference, ord=2)
    cov_term = torch.linalg.vector_norm(diag - diag_reference, ord=2)
    return mean_term + float(beta) * cov_term


def cosine_alignment(update, reference, eps):
    denom = torch.linalg.vector_norm(update) * torch.linalg.vector_norm(reference) + float(eps)
    return torch.dot(update, reference) / denom


def update_reference(reference, observations, momentum):
    stacked = torch.stack(observations, dim=0)
    return (1.0 - float(momentum)) * reference + float(momentum) * stacked.mean(dim=0)


def fixed_aggregation_weights(direction, shift, reliability_log, cfg):
    eps = float(cfg["nfa"]["epsilon"])
    normalized_shift = shift / (torch.sum(shift) + eps)
    score = (
        float(cfg["nfa"]["omega_direction"]) * direction
        - float(cfg["nfa"]["omega_shift"]) * normalized_shift
        + float(cfg["nfa"]["omega_reliability"]) * reliability_log
    )
    weights = torch.softmax(score / float(cfg["nfa"]["temperature"]), dim=0)
    return weights, score


def learned_aggregation_weights(network, descriptor, direction, shift, reliability, cfg):
    score = network(descriptor, direction, shift, reliability)
    weights = torch.softmax(score / float(cfg["nfa"]["temperature"]), dim=0)
    return weights, score

from pathlib import Path

import torch


def parameter_state(model):
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}


def load_parameter_state(model, state):
    named = dict(model.named_parameters())
    if set(named) != set(state):
        raise RuntimeError("parameter state does not match the model")
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def flatten_state(state, names):
    columns = [state[name].detach().reshape(-1).cpu().to(dtype=torch.float64) for name in names]
    return torch.cat(columns, dim=0)


def add_weighted_states(base, deltas, weights):
    weight_values = [float(weight) for weight in weights]
    merged = {}
    for name, value in base.items():
        accumulated = value.clone()
        for weight, delta in zip(weight_values, deltas):
            accumulated.add_(delta[name], alpha=weight)
        merged[name] = accumulated
    return merged


def differentiable_weighted_states(base, deltas, weights, device):
    merged = {}
    for name, value in base.items():
        accumulated = value.detach().to(device=device, dtype=torch.float32)
        for index, delta in enumerate(deltas):
            step = delta[name].detach().to(device=device, dtype=torch.float32)
            accumulated = accumulated + weights[index] * step
        merged[name] = accumulated
    return merged


def save_checkpoint(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)

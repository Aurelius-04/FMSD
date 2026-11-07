import numpy as np
import torch

from src.data.batch import prepare_batch
from src.data.schema import CLIENT_NAMES
from src.losses.multitask import multitask_loss
from src.utils.metrics import cycle_metrics, regression_metrics


def recommendation_from_score(score, cfg):
    if score > float(cfg["decision"]["expand_threshold"]):
        return "Expand Inputs / Investment"
    if score < float(cfg["decision"]["reduce_threshold"]):
        return "Reduce Risk / Cut Inputs"
    return "Maintain Current Inputs"


def _concatenate(rows):
    return {key: np.concatenate(values, axis=0) for key, values in rows.items()}


def collect_predictions(model, clients, split, cfg, device):
    model.eval()
    rows = {
        "net_true": [],
        "net_pred": [],
        "yield_true": [],
        "yield_pred": [],
        "cost_true": [],
        "cost_pred": [],
        "decision_score": [],
        "cycle_id": [],
        "client_index": [],
    }
    with torch.inference_mode():
        for client in clients:
            loader = client[f"{split}_loader"]
            for batch in loader:
                prepared = prepare_batch(batch, cfg, train=False, device=device)
                prediction = model(prepared)
                rows["net_true"].append(prepared["net_return"].detach().cpu().numpy())
                rows["net_pred"].append(prediction["net_return"].detach().cpu().numpy())
                rows["yield_true"].append(prepared["yield"].detach().cpu().numpy())
                rows["yield_pred"].append(prediction["yield"].detach().cpu().numpy())
                rows["cost_true"].append(prepared["cost"].detach().cpu().numpy())
                rows["cost_pred"].append(prediction["cost"].detach().cpu().numpy())
                rows["decision_score"].append(prediction["decision_score"].detach().cpu().numpy())
                rows["cycle_id"].append(batch["cycle_id"].detach().cpu().numpy())
                rows["client_index"].append(batch["client_index"].detach().cpu().numpy())
    return _concatenate(rows)


def validation_loss(model, clients, cfg, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for client in clients:
            for batch in client["val_loader"]:
                prepared = prepare_batch(batch, cfg, train=False, device=device)
                prediction = model(prepared)
                loss, _ = multitask_loss(prediction, prepared, cfg)
                windows = int(prepared["yield"].shape[0])
                total += float(loss) * windows
                count += windows
    if count == 0:
        raise RuntimeError("validation split is empty")
    return total / count


def summarize_predictions(collected, cfg):
    pooled = regression_metrics(collected["net_true"], collected["net_pred"])
    cycles = cycle_metrics(collected["net_true"], collected["net_pred"], collected["cycle_id"])
    by_client = {}
    for client_index, name in enumerate(CLIENT_NAMES):
        mask = collected["client_index"] == client_index
        if not np.any(mask):
            continue
        by_client[name] = regression_metrics(collected["net_true"][mask], collected["net_pred"][mask])
    recommendations = {
        "Expand Inputs / Investment": 0,
        "Maintain Current Inputs": 0,
        "Reduce Risk / Cut Inputs": 0,
    }
    cycle_rows = []
    for cycle_id in np.unique(collected["cycle_id"]):
        mask = collected["cycle_id"] == cycle_id
        score = float(np.mean(collected["decision_score"][mask]))
        action = recommendation_from_score(score, cfg)
        recommendations[action] += 1
        client_index = int(collected["client_index"][mask][0])
        cycle_rows.append(
            {
                "cycle_id": int(cycle_id),
                "client": CLIENT_NAMES[client_index],
                "net_return": float(collected["net_true"][mask][0]),
                "prediction": float(np.mean(collected["net_pred"][mask])),
                "decision_score": score,
                "recommendation": action,
            }
        )
    return {
        "pooled": pooled,
        "cycle": cycles,
        "clients": by_client,
        "yield_mae": regression_metrics(collected["yield_true"], collected["yield_pred"])["mae"],
        "cost_mae": regression_metrics(collected["cost_true"], collected["cost_pred"])["mae"],
        "recommendations": recommendations,
        "cycles": cycle_rows,
    }

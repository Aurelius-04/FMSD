import torch.nn.functional as F


def multitask_loss(prediction, batch, cfg):
    weights = cfg["loss"]
    loss_yield = F.smooth_l1_loss(prediction["yield"], batch["yield"])
    loss_cost = F.smooth_l1_loss(prediction["cost"], batch["cost"])
    loss_return = F.smooth_l1_loss(prediction["direct_return"], batch["net_return"])
    consistency = (prediction["direct_return"] - prediction["accounting_return"]).abs().mean()
    total = (
        float(weights["lambda_yield"]) * loss_yield
        + float(weights["lambda_cost"]) * loss_cost
        + float(weights["lambda_return"]) * loss_return
        + float(weights["lambda_economic"]) * consistency
    )
    parts = {
        "yield": float(loss_yield.detach()),
        "cost": float(loss_cost.detach()),
        "return": float(loss_return.detach()),
        "economic": float(consistency.detach()),
    }
    return total, parts

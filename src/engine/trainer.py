import json
import math
from pathlib import Path

import torch
from torch.func import functional_call

from src.data.batch import prepare_batch
from src.data.dataset import build_clients
from src.engine.evaluate import collect_predictions, summarize_predictions, validation_loss
from src.losses.multitask import multitask_loss
from src.models.fmsd_net import FMSDNet
from src.models.nfa import (
    ClientScoringNetwork,
    cosine_alignment,
    distribution_shift,
    fixed_aggregation_weights,
    learned_aggregation_weights,
    log_reliability,
    update_reference,
)
from src.utils.seed import set_seed
from src.utils.serialize import json_ready
from src.utils.state import (
    add_weighted_states,
    differentiable_weighted_states,
    flatten_state,
    load_parameter_state,
    parameter_state,
    save_checkpoint,
)


def _flatten_grads(model):
    columns = []
    for param in model.parameters():
        if param.grad is None:
            columns.append(torch.zeros(param.numel(), dtype=torch.float64))
        else:
            columns.append(param.grad.detach().reshape(-1).cpu().to(dtype=torch.float64))
    return torch.cat(columns, dim=0)


def _local_update(model, loader, optimizer, cfg, device, observe):
    model.train()
    loss_sum = 0.0
    seen = 0
    grad_sum = None
    square_sum = 0.0
    steps = 0
    for batch in loader:
        prepared = prepare_batch(batch, cfg, train=True, device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(prepared)
        loss, _ = multitask_loss(prediction, prepared, cfg)
        if not torch.isfinite(loss):
            raise RuntimeError("local loss is not finite")
        loss.backward()
        if observe:
            windows = int(prepared["yield"].shape[0])
            loss_sum += float(loss.detach()) * windows
            seen += windows
            grad_vec = _flatten_grads(model)
            if grad_sum is None:
                grad_sum = torch.zeros_like(grad_vec)
            grad_sum += grad_vec
            square_sum += float(torch.dot(grad_vec, grad_vec))
            steps += 1
        optimizer.step()
    if not observe:
        return None
    if steps == 0 or seen == 0:
        raise RuntimeError("local loader produced no windows")
    mean_grad = grad_sum / steps
    variance = square_sum / steps - float(torch.dot(mean_grad, mean_grad))
    return loss_sum / seen, max(variance, 0.0)


def _representation_statistics(model, loader, cfg, device):
    model.eval()
    features = []
    observed = []
    with torch.inference_mode():
        for batch in loader:
            prepared = prepare_batch(batch, cfg, train=False, device=device)
            prediction = model(prepared)
            features.append(prediction["representation"].detach().cpu())
            observed.append(batch["modality_observed"].detach().cpu())
    stacked = torch.cat(features, dim=0)
    mu = stacked.mean(dim=0)
    diag = stacked.var(dim=0, unbiased=False)
    completeness = float(torch.cat(observed, dim=0).float().mean())
    return mu, diag, completeness


def _train_client(model, client, global_state, cfg, device, names):
    load_parameter_state(model, global_state)
    if cfg["optimizer"]["name"].lower() != "adam":
        raise RuntimeError("local optimization uses Adam")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )
    observed = None
    for epoch in range(int(cfg["train"]["local_epochs"])):
        result = _local_update(
            model,
            client["train_loader"],
            optimizer,
            cfg,
            device,
            epoch == int(cfg["train"]["local_epochs"]) - 1,
        )
        if result is not None:
            observed = result
    mu, diag, completeness = _representation_statistics(model, client["train_loader"], cfg, device)
    local_state = parameter_state(model)
    delta = {name: local_state[name] - global_state[name] for name in names}
    return {
        "loss": observed[0],
        "gradient_variance": observed[1],
        "completeness": completeness,
        "mu": mu,
        "diag": diag,
        "delta": delta,
        "delta_vector": flatten_state(delta, names),
        "fraction": float(client["record_fraction"]),
    }


def _call_state(model, parameters):
    state = {name: tensor for name, tensor in model.named_buffers()}
    state.update(parameters)
    return state


def _scoring_meta_update(model, scoring, global_state, deltas, weights, clients, cfg, device):
    model.eval()
    prepared_batches = []
    for client in clients:
        for batch in client["val_loader"]:
            prepared_batches.append(prepare_batch(batch, cfg, train=False, device=device))
    if not prepared_batches:
        raise RuntimeError("validation records are required for the learned scoring update")
    parameters = differentiable_weighted_states(global_state, deltas, weights, device)
    total = None
    for prepared in prepared_batches:
        prediction = functional_call(model, _call_state(model, parameters), (prepared,))
        loss, _ = multitask_loss(prediction, prepared, cfg)
        total = loss if total is None else total + loss
    total = total / len(prepared_batches)
    total.backward()
    meta_lr = float(cfg["nfa"]["meta_lr"])
    with torch.no_grad():
        for param in scoring.parameters():
            if param.grad is not None:
                param.sub_(meta_lr * param.grad)
        scoring.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)


def _communication(param_count, latent, client_count, rounds):
    model_bytes = param_count * 4
    auxiliary_floats = 4 + latent + latent
    auxiliary_bytes = auxiliary_floats * 4
    per_round = client_count * (model_bytes + model_bytes + auxiliary_bytes)
    total_bytes = per_round * rounds
    return {
        "parameter_count": int(param_count),
        "model_megabytes": model_bytes / 1e6,
        "auxiliary_floats": int(auxiliary_floats),
        "auxiliary_kilobytes": auxiliary_bytes / 1e3,
        "per_round_megabytes": per_round / 1e6,
        "total_gigabytes": total_bytes / 1e9,
        "rounds": int(rounds),
    }


def run_federated(cfg, device):
    set_seed(cfg["seed"])
    clients = build_clients(cfg, device)
    model = FMSDNet(cfg).to(device)
    names = [name for name, _ in model.named_parameters()]
    param_count = sum(param.numel() for param in model.parameters())
    latent = int(cfg["model"]["latent_dim"])
    aggregation = cfg["aggregation"]
    scoring = None
    if aggregation == "nfa_learned":
        scoring = ClientScoringNetwork(4, int(cfg["nfa"]["scoring_hidden"])).to(device)
    elif aggregation != "nfa_fixed":
        raise RuntimeError("aggregation must be nfa_fixed or nfa_learned")
    global_state = parameter_state(model)
    mu_g = torch.zeros(latent, dtype=torch.float32)
    diag_g = torch.zeros(latent, dtype=torch.float32)
    history = torch.zeros(param_count, dtype=torch.float64)
    rounds = int(cfg["train"]["communication_rounds"])
    val_history = []
    weight_history = []
    train_history = []
    converged = False
    executed = 0
    best_val = None
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"parameters {param_count} aggregation {aggregation} clients {len(clients)}")
    for round_index in range(1, rounds + 1):
        reports = []
        for client in clients:
            report = _train_client(model, client, global_state, cfg, device, names)
            reports.append(report)
            print(
                f"round {round_index}/{rounds} {client['name']} "
                f"train_loss {report['loss']:.6f} grad_var {report['gradient_variance']:.6e} "
                f"completeness {report['completeness']:.4f}"
            )
        mu_g = update_reference(mu_g, [report["mu"] for report in reports], cfg["nfa"]["momentum"])
        diag_g = update_reference(diag_g, [report["diag"] for report in reports], cfg["nfa"]["momentum"])
        direction = torch.stack(
            [
                cosine_alignment(report["delta_vector"], history, cfg["nfa"]["epsilon"])
                for report in reports
            ]
        )
        shift = torch.stack(
            [
                distribution_shift(report["mu"], report["diag"], mu_g, diag_g, cfg["nfa"]["beta"])
                for report in reports
            ]
        ).to(dtype=torch.float64)
        reliability_log = torch.tensor(
            [
                log_reliability(report["loss"], report["gradient_variance"], report["completeness"], cfg)
                for report in reports
            ],
            dtype=torch.float64,
        )
        deltas = [report["delta"] for report in reports]
        if aggregation == "nfa_fixed":
            weights, scores = fixed_aggregation_weights(direction, shift, reliability_log, cfg)
        else:
            floor = float(cfg["nfa"]["reliability_floor"])
            reliability = torch.exp(reliability_log.clamp(min=math.log(floor), max=20.0)).float().to(device)
            descriptor = torch.tensor(
                [
                    [report["loss"], report["gradient_variance"], report["completeness"], report["fraction"]]
                    for report in reports
                ],
                dtype=torch.float32,
            ).to(device)
            weights, scores = learned_aggregation_weights(
                scoring,
                descriptor,
                direction.float().to(device),
                shift.float().to(device),
                reliability,
                cfg,
            )
            _scoring_meta_update(model, scoring, global_state, deltas, weights, clients, cfg, device)
        if not torch.isfinite(scores).all() or not torch.isfinite(weights).all():
            raise RuntimeError("aggregation scores are not finite")
        numeric_weights = weights.detach().cpu().to(dtype=torch.float64)
        global_state = add_weighted_states(global_state, deltas, numeric_weights)
        load_parameter_state(model, global_state)
        history = torch.zeros_like(history)
        for weight, report in zip(numeric_weights.tolist(), reports):
            history = history + weight * report["delta_vector"]
        val_loss = validation_loss(model, clients, cfg, device)
        val_history.append(val_loss)
        weight_row = {client["name"]: float(weight) for client, weight in zip(clients, numeric_weights.tolist())}
        weight_history.append(weight_row)
        train_history.append({client["name"]: report["loss"] for client, report in zip(clients, reports)})
        executed = round_index
        payload = {
            "parameters": global_state,
            "scoring": None if scoring is None else {k: v.detach().cpu() for k, v in scoring.state_dict().items()},
            "mu_g": mu_g.cpu(),
            "diag_g": diag_g.cpu(),
            "round": round_index,
            "val_loss": val_loss,
            "aggregation": aggregation,
        }
        save_checkpoint(output_dir / "last.pt", payload)
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output_dir / "best.pt", payload)
        joined = " ".join(f"{name}:{weight:.4f}" for name, weight in weight_row.items())
        print(f"round {round_index}/{rounds} weights {joined}")
        print(f"round {round_index}/{rounds} val_loss {val_loss:.6f}")
        window = int(cfg["train"]["convergence_window"])
        if len(val_history) > window:
            previous = val_history[-1 - window]
            relative = abs(val_history[-1] - previous) / max(abs(previous), 1e-12)
            if relative < float(cfg["train"]["convergence_tolerance"]):
                converged = True
                print(f"round {round_index}/{rounds} convergence criterion met")
                break
    collected = collect_predictions(model, clients, "test", cfg, device)
    summary = summarize_predictions(collected, cfg)
    if not math.isfinite(summary["pooled"]["mae"]):
        raise RuntimeError("test metrics are not finite")
    metrics = {
        "aggregation": aggregation,
        "parameter_count": int(param_count),
        "rounds_executed": executed,
        "rounds_budget": rounds,
        "converged": converged,
        "best_val_loss": best_val,
        "validation_loss": val_history,
        "train_loss": train_history,
        "aggregation_weights": weight_history,
        "communication": _communication(param_count, latent, len(clients), executed),
        "test": summary,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(metrics), handle, indent=2)
        handle.write("\n")
    pooled = summary["pooled"]
    cycle = summary["cycle"]
    print(
        "test net_return "
        f"mae {pooled['mae']:.6f} rmse {pooled['rmse']:.6f} "
        f"mape {pooled['mape']:.6f} r2 {pooled['r2']:.6f}"
    )
    print(
        "test cycle "
        f"mae {cycle['mae']:.6f} rmse {cycle['rmse']:.6f} r2 {cycle['r2']:.6f} "
        f"cycles {cycle['cycles']}"
    )
    print(f"wrote {output_dir / 'metrics.json'}")
    return metrics

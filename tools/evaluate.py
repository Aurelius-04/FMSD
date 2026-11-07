import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataset import build_clients
from src.engine.evaluate import collect_predictions, summarize_predictions
from src.models.fmsd_net import FMSDNet
from src.utils.device import resolve_device
from src.utils.seed import set_seed
from src.utils.serialize import json_ready
from src.utils.state import load_parameter_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "outputs" / "last.pt"))
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def resolve_under_root(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.device:
        cfg["device"] = args.device
    cfg["data_root"] = str(resolve_under_root(ROOT, cfg["data_root"]))
    output_dir = resolve_under_root(ROOT, cfg["output_dir"])
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    set_seed(cfg["seed"])
    device, device_name = resolve_device(cfg["device"])
    print(f"device {device_name}")
    clients = build_clients(cfg, device)
    model = FMSDNet(cfg).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_parameter_state(model, payload["parameters"])
    collected = collect_predictions(model, clients, args.split, cfg, device)
    summary = summarize_predictions(collected, cfg)
    destination = output_dir / f"{args.split}_metrics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, indent=2)
        handle.write("\n")
    pooled = summary["pooled"]
    cycle = summary["cycle"]
    print(
        f"{args.split} net_return "
        f"mae {pooled['mae']:.6f} rmse {pooled['rmse']:.6f} "
        f"mape {pooled['mape']:.6f} r2 {pooled['r2']:.6f}"
    )
    print(
        f"{args.split} cycle "
        f"mae {cycle['mae']:.6f} rmse {cycle['rmse']:.6f} r2 {cycle['r2']:.6f} "
        f"cycles {cycle['cycles']}"
    )
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()

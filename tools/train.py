import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.engine.trainer import run_federated
from src.utils.device import resolve_device
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--aggregation", type=str, default=None, choices=["nfa_fixed", "nfa_learned"])
    return parser.parse_args()


def resolve_under_root(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.device:
        cfg["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.rounds is not None:
        cfg["train"]["communication_rounds"] = args.rounds
    if args.local_epochs is not None:
        cfg["train"]["local_epochs"] = args.local_epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.aggregation:
        cfg["aggregation"] = args.aggregation
    cfg["data_root"] = resolve_under_root(ROOT, cfg["data_root"])
    cfg["output_dir"] = resolve_under_root(ROOT, cfg["output_dir"])
    set_seed(cfg["seed"])
    device, device_name = resolve_device(cfg["device"])
    print(f"device {device_name}")
    run_federated(cfg, device)


if __name__ == "__main__":
    main()

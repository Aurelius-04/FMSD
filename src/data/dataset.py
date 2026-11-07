from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocess import apply_scaler, channel_scaler, repair_windows
from src.data.schema import (
    CLIENT_NAMES,
    ENVIRONMENTAL_CHANNELS,
    IMAGE_SIZE,
    SOIL_CHANNELS,
    STRUCTURED_CHANNELS,
    WINDOW_LENGTH,
)


class GreenhouseWindowDataset(Dataset):
    def __init__(self, arrays, client_index):
        self.environmental = torch.from_numpy(arrays["environmental"])
        self.soil = torch.from_numpy(arrays["soil"])
        self.canopy = torch.from_numpy(arrays["canopy"])
        self.structured = torch.from_numpy(arrays["structured"])
        self.price = torch.from_numpy(arrays["price"])
        self.target_yield = torch.from_numpy(arrays["yield"])
        self.cost = torch.from_numpy(arrays["cost"])
        self.net_return = torch.from_numpy(arrays["net_return"])
        self.cycle_id = torch.from_numpy(arrays["cycle_id"].astype(np.int64))
        self.modality_observed = torch.from_numpy(arrays["modality_observed"])
        self.client_index = int(client_index)

    def __len__(self):
        return int(self.environmental.shape[0])

    def __getitem__(self, index):
        return {
            "environmental": self.environmental[index],
            "soil": self.soil[index],
            "canopy": self.canopy[index],
            "structured": self.structured[index],
            "price": self.price[index],
            "yield": self.target_yield[index],
            "cost": self.cost[index],
            "net_return": self.net_return[index],
            "cycle_id": self.cycle_id[index],
            "modality_observed": self.modality_observed[index],
            "client_index": torch.tensor(self.client_index, dtype=torch.long),
        }


def _load_client(path):
    with np.load(path) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _split_mask(season, split_season):
    return season.astype(np.int64) == int(split_season)


def _take(array, mask):
    return array[mask]


def _scale_split(train_array, val_array, test_array):
    center, spread = channel_scaler(train_array)
    return (
        apply_scaler(train_array, center, spread),
        apply_scaler(val_array, center, spread),
        apply_scaler(test_array, center, spread),
    )


def _validate_raw_shapes(raw, cfg):
    if raw["environmental"].shape[1:] != (cfg["model"]["window_length"], cfg["model"]["environmental_channels"]):
        raise RuntimeError("environmental tensor shape does not match the configuration")
    if raw["soil"].shape[1:] != (cfg["model"]["window_length"], cfg["model"]["soil_channels"]):
        raise RuntimeError("soil tensor shape does not match the configuration")
    if raw["structured"].shape[1:] != (cfg["model"]["structured_channels"],):
        raise RuntimeError("structured tensor shape does not match the configuration")
    image_size = cfg["model"]["image_size"]
    if raw["canopy"].shape[1:] != (3, image_size, image_size):
        raise RuntimeError("canopy tensor shape does not match the configuration")
    if len(ENVIRONMENTAL_CHANNELS) != cfg["model"]["environmental_channels"]:
        raise RuntimeError("environmental channel count does not match the schema")
    if len(SOIL_CHANNELS) != cfg["model"]["soil_channels"]:
        raise RuntimeError("soil channel count does not match the schema")
    if len(STRUCTURED_CHANNELS) != cfg["model"]["structured_channels"]:
        raise RuntimeError("structured channel count does not match the schema")
    if WINDOW_LENGTH != cfg["model"]["window_length"] or IMAGE_SIZE != cfg["model"]["image_size"]:
        raise RuntimeError("dataset geometry does not match the model configuration")


def ensure_dataset(cfg):
    root = Path(cfg["data_root"])
    missing = [name for name in CLIENT_NAMES if not (root / f"{name}.npz").exists()]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"missing client records under {root}: {joined}")
    return root


def _make_loader(dataset, batch_size, shuffle, cfg, device, seed):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=False,
    )


def build_clients(cfg, device):
    if tuple(cfg["clients"]) != CLIENT_NAMES:
        raise RuntimeError("client order must match the dataset schema")
    root = ensure_dataset(cfg)
    batch_size = cfg["train"]["batch_size"]
    clients = []
    train_counts = []
    prepared = []
    for client_index, name in enumerate(CLIENT_NAMES):
        raw = _load_client(root / f"{name}.npz")
        _validate_raw_shapes(raw, cfg)
        environmental = repair_windows(raw["environmental"].astype(np.float32), cfg)
        soil = repair_windows(raw["soil"].astype(np.float32), cfg)
        canopy = raw["canopy"].astype(np.float32) / 255.0
        structured = raw["structured"].astype(np.float32)
        season = raw["season"].astype(np.int64)
        train_mask = _split_mask(season, cfg["seasons"]["train"])
        val_mask = _split_mask(season, cfg["seasons"]["val"])
        test_mask = _split_mask(season, cfg["seasons"]["test"])
        if not train_mask.any() or not val_mask.any() or not test_mask.any():
            raise RuntimeError(f"client {name} is missing a season split")
        environmental = _scale_split(environmental[train_mask], environmental[val_mask], environmental[test_mask])
        soil = _scale_split(soil[train_mask], soil[val_mask], soil[test_mask])
        structured = _scale_split(structured[train_mask], structured[val_mask], structured[test_mask])
        splits = {}
        for split_name, mask, env_i, soil_i, struct_i in (
            ("train", train_mask, environmental[0], soil[0], structured[0]),
            ("val", val_mask, environmental[1], soil[1], structured[1]),
            ("test", test_mask, environmental[2], soil[2], structured[2]),
        ):
            splits[split_name] = {
                "environmental": env_i,
                "soil": soil_i,
                "canopy": canopy[mask],
                "structured": struct_i,
                "price": raw["price"][mask].astype(np.float32),
                "yield": raw["yield"][mask].astype(np.float32),
                "cost": raw["cost"][mask].astype(np.float32),
                "net_return": raw["net_return"][mask].astype(np.float32),
                "cycle_id": raw["cycle_id"][mask].astype(np.int64),
                "modality_observed": raw["modality_observed"][mask].astype(np.float32),
            }
        prepared.append({"name": name, "index": client_index, "splits": splits})
        train_counts.append(int(train_mask.sum()))
    total_train = float(sum(train_counts))
    for item, train_count in zip(prepared, train_counts):
        loaders = {}
        for split_name, shuffle in (("train", True), ("val", False), ("test", False)):
            dataset = GreenhouseWindowDataset(item["splits"][split_name], item["index"])
            loaders[split_name] = _make_loader(
                dataset,
                batch_size,
                shuffle,
                cfg,
                device,
                cfg["seed"] + item["index"],
            )
        clients.append(
            {
                "name": item["name"],
                "index": item["index"],
                "train_loader": loaders["train"],
                "val_loader": loaders["val"],
                "test_loader": loaders["test"],
                "record_fraction": train_count / total_train,
                "train_count": train_count,
            }
        )
    return clients

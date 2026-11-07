import math

import torch
import torch.nn.functional as F


def augment_canopy(images, cfg):
    batch = images.shape[0]
    augmented = images
    if cfg["augmentation"]["horizontal_flip"]:
        flip = torch.rand(batch, device=images.device) < 0.5
        if bool(flip.any()):
            augmented = augmented.clone()
            augmented[flip] = torch.flip(images[flip], dims=[-1])
    degrees = cfg["augmentation"]["rotation_degrees"]
    if degrees and degrees > 0:
        angle = (torch.rand(batch, device=images.device) * 2.0 - 1.0) * degrees
        radians = angle * math.pi / 180.0
        cos_t = torch.cos(radians)
        sin_t = torch.sin(radians)
        affine = images.new_zeros(batch, 2, 3)
        affine[:, 0, 0] = cos_t
        affine[:, 0, 1] = -sin_t
        affine[:, 1, 0] = sin_t
        affine[:, 1, 1] = cos_t
        grid = F.affine_grid(affine, augmented.shape, align_corners=False)
        augmented = F.grid_sample(
            augmented,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )
    brightness = cfg["augmentation"]["brightness"]
    if brightness and brightness > 0:
        scale = 1.0 + (torch.rand(batch, 1, 1, 1, device=images.device, dtype=images.dtype) * 2.0 - 1.0) * brightness
        augmented = (augmented * scale).clamp(0.0, 1.0)
    return augmented


def normalize_imagenet(images, cfg):
    mean = images.new_tensor(cfg["imagenet"]["mean"]).view(1, 3, 1, 1)
    std = images.new_tensor(cfg["imagenet"]["std"]).view(1, 3, 1, 1)
    return (images - mean) / std


def prepare_batch(batch, cfg, train, device):
    environmental = batch["environmental"].to(device)
    soil = batch["soil"].to(device)
    canopy = batch["canopy"].to(device)
    structured = batch["structured"].to(device)
    if train:
        noise = float(cfg["augmentation"]["sensor_noise_std"])
        if noise > 0:
            environmental = environmental + torch.randn_like(environmental) * noise
            soil = soil + torch.randn_like(soil) * noise
        canopy = augment_canopy(canopy, cfg)
    canopy = normalize_imagenet(canopy, cfg)
    return {
        "environmental": environmental,
        "soil": soil,
        "canopy": canopy,
        "structured": structured,
        "price": batch["price"].to(device),
        "yield": batch["yield"].to(device),
        "cost": batch["cost"].to(device),
        "net_return": batch["net_return"].to(device),
        "cycle_id": batch["cycle_id"],
        "modality_observed": batch["modality_observed"],
        "client_index": batch["client_index"],
    }

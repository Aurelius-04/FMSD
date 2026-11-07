import torch


def resolve_device(requested):
    name = "auto" if requested is None else str(requested).strip().lower()
    cuda_ready = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_ready = bool(mps_backend is not None and mps_backend.is_available())
    if name in ("auto", ""):
        if cuda_ready:
            return torch.device("cuda"), "cuda"
        if mps_ready:
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"
    if name.startswith("cuda"):
        if cuda_ready:
            return torch.device(name), name
        if mps_ready:
            return torch.device("mps"), "mps (cuda unavailable)"
        return torch.device("cpu"), "cpu (cuda unavailable)"
    if name == "mps":
        if mps_ready:
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu (mps unavailable)"
    return torch.device(name), name

from __future__ import annotations

import torch


def resolve_torch_device(requested: str | torch.device | None = "cuda") -> torch.device:
    """Return a usable torch device for local or remote runs."""
    if isinstance(requested, torch.device):
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested

    value = str(requested or "cuda")
    if value.startswith("cuda") and torch.cuda.is_available():
        return torch.device(value)
    return torch.device("cpu")

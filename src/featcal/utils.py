from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_json(data: dict[str, Any], path: str | Path, *, indent: int | None = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, sort_keys=False)
        f.write("\n")


def save_report(report: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, output_dir / "metrics.json", indent=2)
    save_json(report, output_dir / "run.log", indent=None)


def count_parameters(model: nn.Module) -> dict[str, float | int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "all_params": total,
        "trainable_percentage": trainable / total if total else 0.0,
    }


def dataloader_worker_count(default: int = 8) -> int:
    cpu_count = os.cpu_count() or 1
    return max(0, min(default, cpu_count))


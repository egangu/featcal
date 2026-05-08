from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformers import CLIPVisionModel

from .models import load_vision_model


def task_arithmetic_merge(
    *,
    base_model_id: str,
    expert_model_ids: Mapping[str, str],
    scaling_factor: float,
    cache_dir: str | Path | None = None,
) -> CLIPVisionModel:
    """Construct a Task Arithmetic merged CLIP vision model.

    The merged state is

        theta_0 + scaling_factor * sum_i(theta_i - theta_0).

    Experts are loaded one at a time to keep the demo usable on ordinary
    workstations.
    """

    base_model = load_vision_model(base_model_id, cache_dir=cache_dir)
    base_state = {
        key: value.detach().cpu().clone()
        for key, value in base_model.state_dict().items()
    }
    merged_state: dict[str, Tensor] = {
        key: value.clone() for key, value in base_state.items()
    }

    for task, expert_id in tqdm(
        expert_model_ids.items(),
        desc="Task Arithmetic: loading experts",
    ):
        expert = load_vision_model(expert_id, cache_dir=cache_dir)
        expert_state = expert.state_dict()
        for key, base_value in base_state.items():
            if not torch.is_floating_point(base_value):
                continue
            merged_state[key].add_(
                expert_state[key].detach().cpu() - base_value,
                alpha=float(scaling_factor),
            )
        del expert, expert_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base_model.load_state_dict(merged_state)
    base_model.eval()
    return base_model


def save_vision_model(model: CLIPVisionModel, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)


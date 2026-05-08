from __future__ import annotations

import gc
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPVisionModel

from .data import iter_limited_image_batches, load_task_dataset, make_clip_loader
from .models import load_vision_model
from .solver import merge_layer_parameters_with_featcal_stats, modules_to_calibrate


@dataclass
class FeatCalConfig:
    num_calibration_examples: int = 256
    calibration_batch_size: int = 16
    lambda_ratio: float = 0.05
    anchor_blend_rho: float = 2.2
    teacher_interp_alpha: float = 0.25
    covariance_eps: float = 1e-8
    calibrate_bias: bool = True
    calibrate_layernorm: bool = True
    weight_transpose: bool = True


def _extract_hidden_state(outputs) -> Tensor:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _clip_first_layer_input(model: CLIPVisionModel, images: Tensor) -> Tensor:
    vision = model.vision_model
    hidden = vision.embeddings(images)
    return vision.pre_layrnorm(hidden)


def _clip_layers(model: CLIPVisionModel):
    return model.vision_model.encoder.layers


def _forward_clip_layer(layer: nn.Module, hidden_state: Tensor) -> Tensor:
    outputs = layer(
        hidden_state,
        attention_mask=None,
        causal_attention_mask=None,
    )
    return _extract_hidden_state(outputs)


def _token_weights_for_module(
    module_name: str,
    x: Tensor,
    *,
    layer_idx: int,
    num_layers: int,
) -> Tensor:
    weights = torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)
    if (
        x.dim() == 3
        and x.shape[1] > 1
        and layer_idx == num_layers - 1
        and not module_name.endswith("k_proj")
        and not module_name.endswith("v_proj")
    ):
        weights[:, 1:] = 0.01
    return weights


@torch.no_grad()
def collect_layer_statistics(
    *,
    student_layer: nn.Module,
    expert_layer: nn.Module,
    student_inputs: list[Tensor],
    expert_inputs: list[Tensor],
    layer_idx: int,
    num_layers: int,
    teacher_interp_alpha: float,
) -> tuple[dict[str, dict[str, Tensor | str]], list[Tensor]]:
    if len(student_inputs) != len(expert_inputs):
        raise ValueError("Student and expert input lists must have equal length.")

    student_modules = modules_to_calibrate(student_layer)
    expert_modules = modules_to_calibrate(expert_layer)
    if list(student_modules) != list(expert_modules):
        raise KeyError(
            "Student and expert module names do not align: "
            f"{list(student_modules)} vs {list(expert_modules)}"
        )

    student_cache: dict[str, Tensor] = {}
    expert_cache: dict[str, Tensor] = {}
    accumulators: dict[str, dict[str, Tensor | str]] = {}
    next_expert_inputs: list[Tensor] = []
    handles = []

    def build_hook(cache: dict[str, Tensor], name: str):
        def hook(module: nn.Module, inputs: tuple):
            del module
            cache[name] = inputs[0].detach()

        return hook

    for name, module in student_modules.items():
        handles.append(module.register_forward_pre_hook(build_hook(student_cache, name)))
    for name, module in expert_modules.items():
        handles.append(module.register_forward_pre_hook(build_hook(expert_cache, name)))

    student_device = next(student_layer.parameters()).device
    expert_device = next(expert_layer.parameters()).device
    alpha = float(teacher_interp_alpha)

    try:
        for student_input, expert_input in zip(student_inputs, expert_inputs):
            student_cache.clear()
            expert_cache.clear()
            student_input = student_input.to(student_device)
            expert_input = expert_input.to(expert_device)
            student_outputs = _forward_clip_layer(student_layer, student_input)
            expert_outputs = _forward_clip_layer(expert_layer, expert_input)
            next_expert_inputs.append(expert_outputs.detach().cpu())

            for module_name in student_modules:
                xs = student_cache[module_name]
                xe = expert_cache[module_name]
                if xs.shape != xe.shape:
                    raise RuntimeError(
                        f"FeatCal feature shape mismatch in {module_name}: "
                        f"student={tuple(xs.shape)}, expert={tuple(xe.shape)}"
                    )

                token_weights = _token_weights_for_module(
                    module_name,
                    xs,
                    layer_idx=layer_idx,
                    num_layers=num_layers,
                )
                flat_w = token_weights.reshape(-1).to(device=xs.device, dtype=xs.dtype)
                flat_w_col = flat_w.unsqueeze(-1)
                weight_sum = flat_w.sum().clamp_min(torch.finfo(xs.dtype).eps)
                student_module = student_modules[module_name]
                expert_module = expert_modules[module_name]

                if isinstance(student_module, nn.Linear):
                    if not isinstance(expert_module, nn.Linear):
                        raise TypeError(f"Expected expert {module_name} to be Linear.")
                    flat_xs = xs.reshape(-1, xs.shape[-1])
                    flat_xe = xe.reshape(-1, xe.shape[-1])
                    flat_xt = alpha * flat_xe + (1.0 - alpha) * flat_xs
                    update = {
                        "kind": "linear",
                        "sum_xs": (flat_xs * flat_w_col).sum(dim=0),
                        "sum_xt": (flat_xt * flat_w_col).sum(dim=0),
                        "sum_xsxs": flat_xs.transpose(0, 1)
                        @ (flat_xs * flat_w_col),
                        "sum_xsxt": flat_xs.transpose(0, 1)
                        @ (flat_xt * flat_w_col),
                        "sum_w": weight_sum,
                    }
                elif isinstance(student_module, nn.LayerNorm):
                    if not isinstance(expert_module, nn.LayerNorm):
                        raise TypeError(f"Expected expert {module_name} to be LayerNorm.")
                    zs = F.layer_norm(
                        xs,
                        normalized_shape=student_module.normalized_shape,
                        weight=None,
                        bias=None,
                        eps=student_module.eps,
                    ).detach()
                    flat_zs = zs.reshape(-1, zs.shape[-1])
                    update = {
                        "kind": "layernorm",
                        "sum_zs": (flat_zs * flat_w_col).sum(dim=0),
                        "sum_zs2": (flat_zs.pow(2) * flat_w_col).sum(dim=0),
                        "sum_w": weight_sum,
                    }
                else:
                    raise TypeError(f"Unsupported module type {type(student_module)!r}.")

                if module_name not in accumulators:
                    accumulators[module_name] = update
                else:
                    for key, value in update.items():
                        if key == "kind":
                            continue
                        accumulators[module_name][key] = (
                            accumulators[module_name][key] + value
                        )
    finally:
        for handle in handles:
            handle.remove()

    stats: dict[str, dict[str, Tensor | str]] = {}
    for module_name, acc in accumulators.items():
        sum_w = acc["sum_w"].detach().cpu()
        if acc["kind"] == "linear":
            stats[module_name] = {
                "kind": "linear",
                "mean_student": (acc["sum_xs"] / sum_w).detach().cpu(),
                "mean_teacher": (acc["sum_xt"] / sum_w).detach().cpu(),
                "cov_ss": (acc["sum_xsxs"] / sum_w).detach().cpu(),
                "cross_st": (acc["sum_xsxt"] / sum_w).detach().cpu(),
            }
        elif acc["kind"] == "layernorm":
            stats[module_name] = {
                "kind": "layernorm",
                "mean_student": (acc["sum_zs"] / sum_w).detach().cpu(),
                "energy_student": (acc["sum_zs2"] / sum_w).detach().cpu(),
            }
        else:
            raise KeyError(f"Unknown stat kind {acc['kind']!r}.")
    return stats, next_expert_inputs


class CLIPFeatCalibrator:
    def __init__(
        self,
        config: FeatCalConfig,
        *,
        processor: CLIPProcessor,
        device: torch.device,
        cache_dir: str | Path | None = None,
        datasets_cache_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.processor = processor
        self.device = device
        self.cache_dir = cache_dir
        self.datasets_cache_dir = datasets_cache_dir

    @torch.no_grad()
    def prepare_first_layer_inputs(
        self,
        *,
        merged_model: CLIPVisionModel,
        expert_models: Mapping[str, CLIPVisionModel],
        tasks: Sequence[str],
    ) -> tuple[dict[str, list[Tensor]], dict[str, list[Tensor]]]:
        student_inputs: dict[str, list[Tensor]] = {}
        expert_inputs: dict[str, list[Tensor]] = {}
        merged_model.to(self.device).eval()

        for task in tqdm(tasks, desc="FeatCal: computing first-layer inputs"):
            dataset = load_task_dataset(task, "train", cache_dir=self.datasets_cache_dir)
            loader = make_clip_loader(
                dataset,
                self.processor,
                batch_size=self.config.calibration_batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=(self.device.type == "cuda"),
            )
            image_batches = iter_limited_image_batches(
                loader,
                self.config.num_calibration_examples,
            )
            expert = expert_models[task].to(self.device).eval()
            task_student_inputs = []
            task_expert_inputs = []
            for images in image_batches:
                images = images.to(self.device)
                task_student_inputs.append(
                    _clip_first_layer_input(merged_model, images).detach().cpu()
                )
                task_expert_inputs.append(
                    _clip_first_layer_input(expert, images).detach().cpu()
                )
            student_inputs[task] = task_student_inputs
            expert_inputs[task] = task_expert_inputs
            expert.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return student_inputs, expert_inputs

    @torch.no_grad()
    def calibrate(
        self,
        *,
        merged_model: CLIPVisionModel,
        base_model: CLIPVisionModel,
        expert_models: Mapping[str, CLIPVisionModel],
        tasks: Sequence[str],
    ) -> CLIPVisionModel:
        merged_model.to(self.device).eval()
        base_model.to("cpu").eval()
        for expert in expert_models.values():
            expert.to("cpu").eval()

        student_inputs, expert_inputs = self.prepare_first_layer_inputs(
            merged_model=merged_model,
            expert_models=expert_models,
            tasks=tasks,
        )

        merged_layers = _clip_layers(merged_model)
        base_layers = _clip_layers(base_model)
        expert_layers = {task: _clip_layers(expert) for task, expert in expert_models.items()}
        num_layers = len(merged_layers)

        for layer_idx in tqdm(range(num_layers), desc="FeatCal: merging layers"):
            merged_layer = merged_layers[layer_idx].to(self.device)
            merged_layer_state = {
                k: v.detach().clone() for k, v in merged_layer.state_dict().items()
            }
            base_layer_state = {
                k: v.detach().clone() for k, v in base_layers[layer_idx].state_dict().items()
            }
            expert_param_dict: dict[str, list[Tensor]] = defaultdict(list)
            task_stats = []
            next_expert_inputs: dict[str, list[Tensor]] = {}

            for task in tasks:
                expert_layer = expert_layers[task][layer_idx].to(self.device)
                for param_name, param_value in expert_layer.state_dict().items():
                    expert_param_dict[param_name].append(param_value.detach().clone())
                stats, next_inputs = collect_layer_statistics(
                    student_layer=merged_layer,
                    expert_layer=expert_layer,
                    student_inputs=student_inputs[task],
                    expert_inputs=expert_inputs[task],
                    layer_idx=layer_idx,
                    num_layers=num_layers,
                    teacher_interp_alpha=self.config.teacher_interp_alpha,
                )
                task_stats.append(stats)
                next_expert_inputs[task] = next_inputs
                expert_layer.to("cpu")

            merged_layer_params = merge_layer_parameters_with_featcal_stats(
                expert_param_dict=expert_param_dict,
                expert_featcal_stats=task_stats,
                weight_transpose=self.config.weight_transpose,
                ridge_lambda=self.config.lambda_ratio,
                merged_params=merged_layer_state,
                base_params=base_layer_state,
                calibrate_bias=self.config.calibrate_bias,
                calibrate_layernorm=self.config.calibrate_layernorm,
                anchor_blend_rho=self.config.anchor_blend_rho,
                covariance_eps=self.config.covariance_eps,
            )
            merged_layer.load_state_dict(merged_layer_params, strict=False)
            expert_inputs = next_expert_inputs

            if layer_idx < num_layers - 1:
                student_inputs = {
                    task: self._forward_student_batches(merged_layer, inputs)
                    for task, inputs in student_inputs.items()
                }

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        merged_model.to(self.device).eval()
        return merged_model

    def _forward_student_batches(
        self,
        layer: nn.Module,
        batches: list[Tensor],
    ) -> list[Tensor]:
        outputs = []
        for batch in batches:
            hidden = _forward_clip_layer(layer, batch.to(self.device))
            outputs.append(hidden.detach().cpu())
        return outputs


def load_expert_models(
    expert_model_ids: Mapping[str, str],
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, CLIPVisionModel]:
    return {
        task: load_vision_model(model_id, cache_dir=cache_dir)
        for task, model_id in tqdm(
            expert_model_ids.items(),
            desc="Loading task experts",
        )
    }


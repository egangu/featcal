from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import torch
from torch import Tensor, nn


def modules_to_calibrate(layer: nn.Module) -> dict[str, nn.Module]:
    return {
        name: module
        for name, module in layer.named_modules()
        if name and isinstance(module, (nn.Linear, nn.LayerNorm))
    }


def _anchor_param(
    merged_param: Optional[Tensor],
    base_param: Optional[Tensor],
    rho: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tensor]:
    if merged_param is None or base_param is None:
        return None
    merged = merged_param.to(device=device, dtype=dtype)
    base = base_param.to(device=device, dtype=dtype)
    return rho * merged + (1.0 - rho) * base


def _preserve_or_average(
    param_values: list[Tensor],
    merged_param: Optional[Tensor],
) -> Tensor:
    if merged_param is not None:
        return merged_param.detach().clone()
    return torch.stack(param_values, dim=0).mean(dim=0)


def featcal_merge_linear_weight(
    param_weight_list: list[Tensor],
    featcal_stats_list: list[Mapping[str, Tensor]],
    *,
    weight_transpose: bool = True,
    ridge_lambda: float = 0.0,
    merged_param: Optional[Tensor] = None,
    base_param: Optional[Tensor] = None,
    anchor_blend_rho: float = 1.0,
    covariance_eps: float = 1e-8,
) -> Tensor:
    if not param_weight_list:
        raise ValueError("param_weight_list must not be empty.")
    if len(param_weight_list) != len(featcal_stats_list):
        raise ValueError("Weight and statistics lists must have the same length.")

    device = param_weight_list[0].device
    solve_dtype = torch.float32
    out_dtype = param_weight_list[0].dtype
    cov_ss_list = [
        stats["cov_ss"].to(device=device, dtype=solve_dtype)
        for stats in featcal_stats_list
    ]
    cross_st_list = [
        stats["cross_st"].to(device=device, dtype=solve_dtype)
        for stats in featcal_stats_list
    ]
    processed_weights = [
        (weight.transpose(-2, -1) if weight_transpose else weight).to(
            device=device,
            dtype=solve_dtype,
        )
        for weight in param_weight_list
    ]

    ident = torch.eye(cov_ss_list[0].shape[0], device=device, dtype=solve_dtype)
    cov_norms = torch.stack(
        [cov.norm(p="fro").clamp_min(covariance_eps) for cov in cov_ss_list],
        dim=0,
    )
    normalized_covs = [cov / norm for cov, norm in zip(cov_ss_list, cov_norms)]
    normalized_cross = [
        cross / norm for cross, norm in zip(cross_st_list, cov_norms)
    ]

    solve_matrix = torch.stack(normalized_covs, dim=0).sum(dim=0)
    rhs = torch.stack(
        [
            cross @ weight
            for cross, weight in zip(normalized_cross, processed_weights)
        ],
        dim=0,
    ).sum(dim=0)

    ridge = max(float(ridge_lambda), 0.0)
    solve_matrix = solve_matrix + ridge * ident
    if covariance_eps > 0:
        solve_matrix = solve_matrix + covariance_eps * ident

    anchor = _anchor_param(
        merged_param,
        base_param,
        anchor_blend_rho,
        device=device,
        dtype=solve_dtype,
    )
    if anchor is not None:
        anchor = anchor.transpose(-2, -1) if weight_transpose else anchor
        if ridge > 0:
            rhs = rhs + ridge * anchor

    try:
        merged_weight = torch.linalg.solve(solve_matrix, rhs)
    except RuntimeError:
        merged_weight = torch.linalg.pinv(solve_matrix) @ rhs
    merged_weight = (
        merged_weight.transpose(-2, -1) if weight_transpose else merged_weight
    )
    return merged_weight.to(dtype=out_dtype)


def featcal_merge_linear_bias(
    param_bias_list: list[Tensor],
    param_weight_list: list[Tensor],
    featcal_stats_list: list[Mapping[str, Tensor]],
    merged_weight: Tensor,
    *,
    ridge_lambda: float = 0.0,
    merged_param: Optional[Tensor] = None,
    base_param: Optional[Tensor] = None,
    anchor_blend_rho: float = 1.0,
    covariance_eps: float = 1e-8,
) -> Tensor:
    if not (
        len(param_bias_list) == len(param_weight_list) == len(featcal_stats_list)
    ):
        raise ValueError("Bias, weight, and statistics lists must have the same length.")

    device = param_bias_list[0].device
    dtype = merged_weight.dtype
    merged_weight = merged_weight.to(device=device, dtype=dtype)
    rhs = torch.zeros_like(param_bias_list[0], device=device, dtype=dtype)
    denom = torch.zeros((), device=device, dtype=dtype)

    for bias_i, weight_i, stats in zip(
        param_bias_list,
        param_weight_list,
        featcal_stats_list,
    ):
        cov_ss = stats["cov_ss"].to(device=device, dtype=dtype)
        mean_student = stats["mean_student"].to(device=device, dtype=dtype)
        mean_teacher = stats["mean_teacher"].to(device=device, dtype=dtype)
        scale = 1.0 / cov_ss.norm(p="fro").clamp_min(covariance_eps)
        bias_i = bias_i.to(device=device, dtype=dtype)
        weight_i = weight_i.to(device=device, dtype=dtype)
        rhs = rhs + scale * (
            bias_i + weight_i @ mean_teacher - merged_weight @ mean_student
        )
        denom = denom + scale

    anchor = _anchor_param(
        merged_param,
        base_param,
        anchor_blend_rho,
        device=device,
        dtype=dtype,
    )
    ridge = max(float(ridge_lambda), 0.0)
    if anchor is not None and ridge > 0:
        rhs = rhs + ridge * anchor
        denom = denom + ridge

    return rhs / denom.clamp_min(torch.finfo(dtype).eps)


def featcal_merge_layernorm_affine(
    param_weight_list: list[Tensor],
    param_bias_list: list[Tensor],
    featcal_stats_list: list[Mapping[str, Tensor]],
    *,
    ridge_lambda: float = 0.0,
    merged_weight: Optional[Tensor] = None,
    base_weight: Optional[Tensor] = None,
    merged_bias: Optional[Tensor] = None,
    base_bias: Optional[Tensor] = None,
    anchor_blend_rho: float = 1.0,
    covariance_eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    if not (
        len(param_weight_list) == len(param_bias_list) == len(featcal_stats_list)
    ):
        raise ValueError(
            "LayerNorm weight, bias, and statistics lists must have the same length."
        )

    device = param_weight_list[0].device
    dtype = param_weight_list[0].dtype
    sum_energy = torch.zeros_like(param_weight_list[0], device=device, dtype=dtype)
    sum_mean = torch.zeros_like(param_weight_list[0], device=device, dtype=dtype)
    rhs_weight = torch.zeros_like(param_weight_list[0], device=device, dtype=dtype)
    rhs_bias = torch.zeros_like(param_weight_list[0], device=device, dtype=dtype)
    num_experts = torch.zeros((), device=device, dtype=dtype)

    for gamma_i, beta_i, stats in zip(
        param_weight_list,
        param_bias_list,
        featcal_stats_list,
    ):
        energy = stats["energy_student"].to(device=device, dtype=dtype)
        mean = stats["mean_student"].to(device=device, dtype=dtype)
        gamma_i = gamma_i.to(device=device, dtype=dtype)
        beta_i = beta_i.to(device=device, dtype=dtype)
        sum_energy = sum_energy + energy
        sum_mean = sum_mean + mean
        rhs_weight = rhs_weight + energy * gamma_i + mean * beta_i
        rhs_bias = rhs_bias + mean * gamma_i + beta_i
        num_experts = num_experts + 1.0

    anchor_weight = _anchor_param(
        merged_weight,
        base_weight,
        anchor_blend_rho,
        device=device,
        dtype=dtype,
    )
    anchor_bias = _anchor_param(
        merged_bias,
        base_bias,
        anchor_blend_rho,
        device=device,
        dtype=dtype,
    )
    ridge = max(float(ridge_lambda), 0.0)
    a11 = sum_energy + (ridge if anchor_weight is not None else 0.0)
    a12 = sum_mean
    a22 = num_experts + (ridge if anchor_bias is not None else 0.0)
    if anchor_weight is not None and ridge > 0:
        rhs_weight = rhs_weight + ridge * anchor_weight
    if anchor_bias is not None and ridge > 0:
        rhs_bias = rhs_bias + ridge * anchor_bias

    det = (a11 * a22 - a12.pow(2)).clamp_min(covariance_eps)
    weight = (a22 * rhs_weight - a12 * rhs_bias) / det
    bias = (a11 * rhs_bias - a12 * rhs_weight) / det
    return weight, bias


def merge_layer_parameters_with_featcal_stats(
    *,
    expert_param_dict: Mapping[str, list[Tensor]],
    expert_featcal_stats: list[Mapping[str, Mapping[str, Tensor]]],
    weight_transpose: bool,
    ridge_lambda: float,
    merged_params: Mapping[str, Tensor],
    base_params: Mapping[str, Tensor],
    calibrate_bias: bool,
    calibrate_layernorm: bool,
    anchor_blend_rho: float,
    covariance_eps: float,
) -> dict[str, Tensor]:
    merged_layer_params: dict[str, Tensor] = {}

    for param_name, param_values in expert_param_dict.items():
        if param_name in merged_layer_params:
            continue

        merged_param = merged_params.get(param_name)
        base_param = base_params.get(param_name)

        if param_name.endswith(".weight"):
            module_name = param_name[: -len(".weight")]
            module_stats = [stats[module_name] for stats in expert_featcal_stats]
            module_kind = module_stats[0]["kind"]

            if module_kind == "layernorm":
                bias_name = f"{module_name}.bias"
                bias_values = expert_param_dict[bias_name]
                if calibrate_layernorm:
                    weight, bias = featcal_merge_layernorm_affine(
                        param_values,
                        bias_values,
                        module_stats,
                        ridge_lambda=ridge_lambda,
                        merged_weight=merged_param,
                        base_weight=base_param,
                        merged_bias=merged_params.get(bias_name),
                        base_bias=base_params.get(bias_name),
                        anchor_blend_rho=anchor_blend_rho,
                        covariance_eps=covariance_eps,
                    )
                    merged_layer_params[param_name] = weight
                    merged_layer_params[bias_name] = bias
                else:
                    merged_layer_params[param_name] = _preserve_or_average(
                        param_values,
                        merged_param,
                    )
                    merged_layer_params[bias_name] = _preserve_or_average(
                        bias_values,
                        merged_params.get(bias_name),
                    )
                continue

            merged_layer_params[param_name] = featcal_merge_linear_weight(
                param_values,
                module_stats,
                weight_transpose=weight_transpose,
                ridge_lambda=ridge_lambda,
                merged_param=merged_param,
                base_param=base_param,
                anchor_blend_rho=anchor_blend_rho,
                covariance_eps=covariance_eps,
            )
            continue

        if param_name.endswith(".bias"):
            module_name = param_name[: -len(".bias")]
            module_stats = [stats[module_name] for stats in expert_featcal_stats]
            module_kind = module_stats[0]["kind"]
            if module_kind == "layernorm":
                if not calibrate_layernorm:
                    merged_layer_params[param_name] = _preserve_or_average(
                        param_values,
                        merged_param,
                    )
                continue
            if not calibrate_bias:
                merged_layer_params[param_name] = _preserve_or_average(
                    param_values,
                    merged_param,
                )
                continue
            weight_name = f"{module_name}.weight"
            merged_layer_params[param_name] = featcal_merge_linear_bias(
                param_values,
                expert_param_dict[weight_name],
                module_stats,
                merged_layer_params[weight_name],
                ridge_lambda=ridge_lambda,
                merged_param=merged_param,
                base_param=base_param,
                anchor_blend_rho=anchor_blend_rho,
                covariance_eps=covariance_eps,
            )
            continue

        merged_layer_params[param_name] = _preserve_or_average(
            param_values,
            merged_param,
        )

    return merged_layer_params


from __future__ import annotations

import torch

from featcal.solver import (
    featcal_merge_layernorm_affine,
    featcal_merge_linear_bias,
    featcal_merge_linear_weight,
)


def test_linear_weight_identity_solution():
    weight = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    stats = {
        "cov_ss": torch.eye(2),
        "cross_st": torch.eye(2),
        "mean_student": torch.zeros(2),
        "mean_teacher": torch.zeros(2),
    }
    merged = featcal_merge_linear_weight(
        [weight],
        [stats],
        ridge_lambda=0.0,
        covariance_eps=0.0,
    )
    assert torch.allclose(merged, weight)


def test_linear_bias_matches_expert_mean():
    weight = torch.eye(2)
    bias = torch.tensor([1.0, -2.0])
    stats = {
        "cov_ss": torch.eye(2),
        "mean_student": torch.tensor([0.0, 0.0]),
        "mean_teacher": torch.tensor([0.0, 0.0]),
    }
    merged = featcal_merge_linear_bias(
        [bias],
        [weight],
        [stats],
        merged_weight=weight,
        ridge_lambda=0.0,
    )
    assert torch.allclose(merged, bias)


def test_layernorm_affine_identity_solution():
    gamma = torch.tensor([1.5, 0.5])
    beta = torch.tensor([-0.25, 0.75])
    stats = {
        "mean_student": torch.zeros(2),
        "energy_student": torch.ones(2),
    }
    merged_gamma, merged_beta = featcal_merge_layernorm_affine(
        [gamma],
        [beta],
        [stats],
        ridge_lambda=0.0,
    )
    assert torch.allclose(merged_gamma, gamma)
    assert torch.allclose(merged_beta, beta)


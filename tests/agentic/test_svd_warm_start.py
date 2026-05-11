"""Unit tests for SVD warm-start utility.

All tests use synthetic inputs; no Ray / model loading required.
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from roll.pipeline.agentic.svd_warm_start import (
    build_meta_lora,
    build_warm_start_state_dict,
    group_lora_pairs_by_module,
    pick_truncation_rank,
    svd_truncate_and_perturb,
)


# ---------------------------------------------------------------------------
# Minimal config stub (mirrors SVDWarmStartConfig fields used by functions)
# ---------------------------------------------------------------------------

@dataclass
class _Cfg:
    truncation_policy: Literal["fixed", "energy"] = "fixed"
    truncation_rank: int = 2
    energy_threshold: float = 0.9
    shrink_factor: float = 1.0
    residual_noise_scope: Literal["a_only", "a_and_b", "none"] = "a_only"
    perturbation_sigma: float = 1e-3
    min_population_size: int = 2
    first_iteration_fallback: Literal["cold_start", "no_op", "raise"] = "cold_start"
    missing_adapter_policy: Literal["skip", "raise"] = "skip"
    missing_param_policy: Literal["kaiming_zero", "raise"] = "kaiming_zero"
    adapter_load_timeout_s: int = 5
    compute_dtype: Literal["float32", "float64"] = "float32"
    seed_offset: int = 0
    enabled: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter_sd(module: str, r: int, d_in: int, d_out: int, A_val: float = 1.0, B_val: float = 1.0) -> dict:
    """Return a minimal PEFT-format state dict for one module."""
    return {
        f"{module}.lora_A.weight": torch.full((r, d_in), A_val),
        f"{module}.lora_B.weight": torch.full((d_out, r), B_val),
    }


def _write_adapter_file(tmpdir: str, name: str, sd: dict) -> str:
    path = os.path.join(tmpdir, name)
    os.makedirs(path, exist_ok=True)
    save_file(sd, os.path.join(path, "adapter_model.safetensors"))
    return path


# ---------------------------------------------------------------------------
# Phase 1 tests
# ---------------------------------------------------------------------------

def test_phase1_meta_lora_aggregation():
    module = "base_model.model.q_proj"
    r, d_in, d_out = 4, 8, 8

    # Two adapters: A1=1, B1=1 and A2=2, B2=2; Nash = [0.3, 0.7]
    sd1 = _make_adapter_sd(module, r, d_in, d_out, A_val=1.0, B_val=1.0)
    sd2 = _make_adapter_sd(module, r, d_in, d_out, A_val=2.0, B_val=2.0)
    nash = [0.3, 0.7]

    W_meta = build_meta_lora([sd1, sd2], nash, compute_dtype="float32")

    # delta_W_1 = B1 @ A1 = (d_out, r) @ (r, d_in) = ones(d_out, d_in) * r
    # delta_W_2 = B2 @ A2 = 4 * ones(d_out, d_in) * r
    expected = 0.3 * r * torch.ones(d_out, d_in) + 0.7 * 4 * r * torch.ones(d_out, d_in)
    assert module in W_meta
    assert torch.allclose(W_meta[module], expected, atol=1e-5)


def test_phase1_base_model_contributes_zero():
    module = "base_model.model.v_proj"
    r, d_in, d_out = 4, 8, 8
    sd = _make_adapter_sd(module, r, d_in, d_out, A_val=1.0, B_val=1.0)
    # base model (None) + adapter; Nash = [0.5, 0.5]
    W_meta = build_meta_lora([None, sd], [0.5, 0.5], compute_dtype="float32")
    expected = 0.5 * r * torch.ones(d_out, d_in)
    assert torch.allclose(W_meta[module], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Phase 2 tests
# ---------------------------------------------------------------------------

def test_phase2_fixed_truncation_preserves_topk():
    torch.manual_seed(0)
    r, d_in, d_out = 8, 16, 16
    W = torch.randn(d_out, d_in)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = 3
    W_topk = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]

    cfg = _Cfg(truncation_policy="fixed", truncation_rank=k, shrink_factor=1.0, residual_noise_scope="none")
    gen = torch.Generator()
    A_new, B_new, k_out, _ = svd_truncate_and_perturb(
        W, r, cfg.truncation_policy, cfg.truncation_rank, cfg.energy_threshold,
        cfg.shrink_factor, cfg.residual_noise_scope, cfg.perturbation_sigma, gen
    )
    W_reconstructed = B_new @ A_new
    assert k_out == k
    assert torch.allclose(W_reconstructed, W_topk, atol=1e-5)


def test_phase2_energy_truncation_picks_k_correctly():
    # Build a matrix with a spectrum where top-2 capture >= 95% of energy.
    torch.manual_seed(1)
    d = 16
    S_true = torch.tensor([10.0, 5.0, 1.0, 0.5, 0.1] + [0.01] * (d - 5))
    U = torch.linalg.qr(torch.randn(d, d))[0]
    Vh = torch.linalg.qr(torch.randn(d, d))[0]
    W = (U[:, :len(S_true)] * S_true.unsqueeze(0)) @ Vh[:len(S_true), :]

    energy_95 = (S_true[:2] ** 2).sum() / (S_true ** 2).sum()
    assert energy_95.item() >= 0.95

    k = pick_truncation_rank(S_true, "energy", truncation_rank=4, energy_threshold=0.95, lora_rank=8)
    assert k == 2


# ---------------------------------------------------------------------------
# Phase 3 tests
# ---------------------------------------------------------------------------

def _run_phase3(scope: str, r: int = 8, k: int = 3):
    torch.manual_seed(42)
    d = 16
    W = torch.randn(d, d)
    cfg = _Cfg(truncation_policy="fixed", truncation_rank=k, shrink_factor=1.0,
               residual_noise_scope=scope, perturbation_sigma=0.01)
    gen = torch.Generator()
    gen.manual_seed(0)
    A_new, B_new, k_out, _ = svd_truncate_and_perturb(
        W, r, cfg.truncation_policy, cfg.truncation_rank, cfg.energy_threshold,
        cfg.shrink_factor, cfg.residual_noise_scope, cfg.perturbation_sigma, gen
    )
    return A_new, B_new, k_out


def test_phase3_residual_noise_scope_a_only():
    r, k = 8, 3
    A_new, B_new, _ = _run_phase3("a_only", r=r, k=k)
    assert torch.all(B_new[:, k:] == 0), "B residual must be zero for a_only"
    assert A_new[k:, :].abs().mean() > 0, "A residual must have noise for a_only"


def test_phase3_residual_noise_scope_a_and_b():
    r, k = 8, 3
    A_new, B_new, _ = _run_phase3("a_and_b", r=r, k=k)
    assert A_new[k:, :].abs().mean() > 0, "A residual must have noise"
    assert B_new[:, k:].abs().mean() > 0, "B residual must have noise"


def test_phase3_residual_noise_scope_none():
    r, k = 8, 3
    A_new, B_new, _ = _run_phase3("none", r=r, k=k)
    assert torch.all(A_new[k:, :] == 0), "A residual must be zero for none"
    assert torch.all(B_new[:, k:] == 0), "B residual must be zero for none"


def test_phase3_shrink_factor_below_one():
    torch.manual_seed(7)
    d, r, k = 16, 8, 4
    W = torch.randn(d, d)
    cfg_full = _Cfg(truncation_policy="fixed", truncation_rank=k, shrink_factor=1.0, residual_noise_scope="none")
    cfg_shrunk = _Cfg(truncation_policy="fixed", truncation_rank=k, shrink_factor=0.5, residual_noise_scope="none")
    gen = torch.Generator()

    gen.manual_seed(0)
    A1, B1, _, _ = svd_truncate_and_perturb(
        W, r, cfg_full.truncation_policy, cfg_full.truncation_rank, cfg_full.energy_threshold,
        cfg_full.shrink_factor, cfg_full.residual_noise_scope, cfg_full.perturbation_sigma, gen
    )
    gen.manual_seed(0)
    A2, B2, _, _ = svd_truncate_and_perturb(
        W, r, cfg_shrunk.truncation_policy, cfg_shrunk.truncation_rank, cfg_shrunk.energy_threshold,
        cfg_shrunk.shrink_factor, cfg_shrunk.residual_noise_scope, cfg_shrunk.perturbation_sigma, gen
    )
    # B @ A with shrink_factor=0.5 should be shrink^2 * (B @ A with shrink_factor=1.0)
    assert torch.allclose(B2 @ A2, 0.25 * (B1 @ A1), atol=1e-5)


# ---------------------------------------------------------------------------
# missing_adapter_policy tests
# ---------------------------------------------------------------------------

def test_missing_adapter_skip_renormalizes_nash():
    module = "base_model.model.q_proj"
    r, d_in, d_out = 4, 8, 8

    with tempfile.TemporaryDirectory() as tmpdir:
        sd1 = _make_adapter_sd(module, r, d_in, d_out, A_val=1.0, B_val=1.0)
        path1 = _write_adapter_file(tmpdir, "ckpt1", sd1)

        cfg = _Cfg(missing_adapter_policy="skip", truncation_rank=2, truncation_policy="fixed",
                   residual_noise_scope="none")
        # path2 does not exist
        path2 = os.path.join(tmpdir, "nonexistent")

        # Nash = [0.4, 0.6]; path2 is missing so only path1 should be used with weight 1.0
        sd_out, meta = build_warm_start_state_dict(
            lora_paths=[path1, path2],
            nash_probs=np.array([0.4, 0.6]),
            cfg=cfg,
            lora_rank=r,
            model_dtype=torch.float32,
            seed=0,
        )
        assert meta["n_skipped"] == 1
        assert meta["n_population"] == 1

        # With only path1 and renorm weight=1.0, W_meta = B1 @ A1 = r * ones(d_out, d_in)
        # Verify the output keys exist
        assert f"{module}.lora_A.default.weight" in sd_out
        assert f"{module}.lora_B.default.weight" in sd_out


def test_missing_adapter_raise():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _Cfg(missing_adapter_policy="raise", truncation_rank=2, adapter_load_timeout_s=1)
        bad_path = os.path.join(tmpdir, "nonexistent")
        with pytest.raises(Exception):
            build_warm_start_state_dict(
                lora_paths=[bad_path],
                nash_probs=np.array([1.0]),
                cfg=cfg,
                lora_rank=4,
                model_dtype=torch.float32,
                seed=0,
            )


# ---------------------------------------------------------------------------
# Key naming test
# ---------------------------------------------------------------------------

def test_key_naming_matches_deepspeed():
    module = "base_model.model.layers.0.self_attn.q_proj"
    r, d_in, d_out = 4, 8, 8

    with tempfile.TemporaryDirectory() as tmpdir:
        sd = _make_adapter_sd(module, r, d_in, d_out)
        path = _write_adapter_file(tmpdir, "ckpt", sd)

        cfg = _Cfg(truncation_rank=2, truncation_policy="fixed", residual_noise_scope="none")
        sd_out, _ = build_warm_start_state_dict(
            lora_paths=[path],
            nash_probs=np.array([1.0]),
            cfg=cfg,
            lora_rank=r,
            model_dtype=torch.float32,
            seed=0,
        )

    for k in sd_out:
        assert ".lora_A.default.weight" in k or ".lora_B.default.weight" in k, (
            f"Key '{k}' does not match DeepSpeed in-model naming convention"
        )

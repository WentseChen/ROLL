"""Unit tests for roll.pipeline.agentic.svd_warm_start.

Pure CPU, synthetic tensors. No Ray, no DeepSpeed, no model loading.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from safetensors.torch import save_file

from roll.pipeline.agentic import svd_warm_start as sws


# ---------- Test config (duck-typed; mirrors SVDWarmStartConfig surface) ---------- #

@dataclass
class _TestCfg:
    enabled: bool = True
    truncation_policy: Literal["fixed", "energy"] = "fixed"
    truncation_rank: int = 4
    energy_threshold: float = 0.9
    shrink_factor: float = 1.0
    residual_noise_scope: Literal["a_only", "a_and_b", "none"] = "a_only"
    perturbation_sigma: float = 1e-3
    min_population_size: int = 2
    first_iteration_fallback: str = "cold_start"
    missing_adapter_policy: Literal["skip", "raise"] = "skip"
    missing_param_policy: str = "kaiming_zero"
    adapter_load_timeout_s: int = 5
    compute_dtype: Literal["float32", "float64"] = "float32"
    seed_offset: int = 0


def _make_adapter_sd(d_in=16, d_out=8, r=4, scale=1.0, seed=0):
    """Synthetic PEFT-format adapter state dict for a single module 'foo'."""
    g = torch.Generator().manual_seed(seed)
    return {
        "base_model.model.foo.lora_A.weight": torch.randn(r, d_in, generator=g) * scale,
        "base_model.model.foo.lora_B.weight": torch.randn(d_out, r, generator=g) * scale,
    }


def _save_adapter_to_dir(sd, tmpdir):
    path = os.path.join(tmpdir, "adapter_model.safetensors")
    save_file(sd, path)
    return tmpdir


# ---------- Phase 1: meta-LoRA aggregation ---------- #

def test_phase1_meta_lora_aggregation():
    sd1 = _make_adapter_sd(seed=1)
    sd2 = _make_adapter_sd(seed=2)
    probs = [0.7, 0.3]
    W = sws.build_meta_lora([sd1, sd2], probs, compute_dtype="float32")
    expected = (
        0.7 * (sd1["base_model.model.foo.lora_B.weight"] @ sd1["base_model.model.foo.lora_A.weight"])
        + 0.3 * (sd2["base_model.model.foo.lora_B.weight"] @ sd2["base_model.model.foo.lora_A.weight"])
    )
    torch.testing.assert_close(W["base_model.model.foo"], expected, rtol=1e-5, atol=1e-5)


def test_phase1_skips_none_entries():
    sd = _make_adapter_sd(seed=3)
    probs = [0.4, 0.6]  # base placeholder (None) + one adapter
    W = sws.build_meta_lora([None, sd], probs, compute_dtype="float32")
    expected = 0.6 * (sd["base_model.model.foo.lora_B.weight"] @ sd["base_model.model.foo.lora_A.weight"])
    torch.testing.assert_close(W["base_model.model.foo"], expected, rtol=1e-5, atol=1e-5)


# ---------- Phase 2: truncation policies ---------- #

def test_phase2_fixed_truncation_preserves_topk():
    cfg = _TestCfg(truncation_policy="fixed", truncation_rank=3, residual_noise_scope="none", perturbation_sigma=0.0)
    W = torch.randn(8, 16, generator=torch.Generator().manual_seed(42))
    A_new, B_new, k, _ = sws.svd_truncate_and_perturb(
        W, lora_rank=8,
        truncation_policy=cfg.truncation_policy, truncation_rank=cfg.truncation_rank,
        energy_threshold=cfg.energy_threshold, shrink_factor=cfg.shrink_factor,
        residual_noise_scope=cfg.residual_noise_scope, perturbation_sigma=cfg.perturbation_sigma,
        generator=torch.Generator().manual_seed(0), output_dtype=torch.float32,
    )
    assert k == 3
    reconstructed = B_new[:, :k].to(torch.float64) @ A_new[:k, :].to(torch.float64)
    U, S, Vh = torch.linalg.svd(W.to(torch.float64), full_matrices=False)
    expected = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
    torch.testing.assert_close(reconstructed, expected, rtol=1e-4, atol=1e-4)


def test_phase2_energy_truncation_picks_k_correctly():
    # Singular values 1.0, 0.5, 0.3, 0.2; energies 1, 0.25, 0.09, 0.04; total = 1.38
    # Cumfrac: 0.725, 0.906, 0.971, 1.0. With threshold 0.9 → k=2.
    sv = torch.tensor([1.0, 0.5, 0.3, 0.2])
    k, retained = sws.pick_truncation_rank(sv, "energy", 0, 0.9, lora_rank=8)
    assert k == 2
    assert 0.9 <= retained < 0.95


def test_phase2_energy_clipped_to_lora_rank_minus_one():
    # Degenerate spectrum: many equal singular values → need many k to reach threshold.
    sv = torch.ones(20)
    k, _ = sws.pick_truncation_rank(sv, "energy", 0, 0.999, lora_rank=4)
    assert k == 3  # lora_rank - 1


# ---------- Phase 3: shrink-and-perturb ---------- #

def test_phase3_residual_noise_scope_a_only():
    cfg_sigma = 0.05
    W = torch.randn(8, 16, generator=torch.Generator().manual_seed(7))
    A_new, B_new, k, _ = sws.svd_truncate_and_perturb(
        W, lora_rank=8, truncation_policy="fixed", truncation_rank=3, energy_threshold=0.9,
        shrink_factor=1.0, residual_noise_scope="a_only", perturbation_sigma=cfg_sigma,
        generator=torch.Generator().manual_seed(0), output_dtype=torch.float32,
    )
    assert torch.all(B_new[:, k:] == 0)
    residual_a = A_new[k:, :]
    assert residual_a.std().item() == pytest.approx(cfg_sigma, rel=0.3)


def test_phase3_residual_noise_scope_a_and_b():
    cfg_sigma = 0.05
    W = torch.randn(8, 16, generator=torch.Generator().manual_seed(7))
    A_new, B_new, k, _ = sws.svd_truncate_and_perturb(
        W, lora_rank=8, truncation_policy="fixed", truncation_rank=3, energy_threshold=0.9,
        shrink_factor=1.0, residual_noise_scope="a_and_b", perturbation_sigma=cfg_sigma,
        generator=torch.Generator().manual_seed(0), output_dtype=torch.float32,
    )
    assert A_new[k:, :].std().item() == pytest.approx(cfg_sigma, rel=0.3)
    assert B_new[:, k:].std().item() == pytest.approx(cfg_sigma, rel=0.3)


def test_phase3_residual_noise_scope_none():
    W = torch.randn(8, 16, generator=torch.Generator().manual_seed(7))
    A_new, B_new, k, _ = sws.svd_truncate_and_perturb(
        W, lora_rank=8, truncation_policy="fixed", truncation_rank=3, energy_threshold=0.9,
        shrink_factor=1.0, residual_noise_scope="none", perturbation_sigma=0.05,
        generator=torch.Generator().manual_seed(0), output_dtype=torch.float32,
    )
    assert torch.all(A_new[k:, :] == 0)
    assert torch.all(B_new[:, k:] == 0)


def test_phase3_shrink_factor_below_one():
    shrink = 0.5
    W = torch.randn(8, 16, generator=torch.Generator().manual_seed(11))
    # Run twice: shrink=1.0 then shrink=0.5; principal slice should scale by `shrink`.
    A1, B1, k1, _ = sws.svd_truncate_and_perturb(
        W, lora_rank=8, truncation_policy="fixed", truncation_rank=3, energy_threshold=0.9,
        shrink_factor=1.0, residual_noise_scope="none", perturbation_sigma=0.0,
        generator=torch.Generator().manual_seed(0), output_dtype=torch.float64,
    )
    A2, B2, k2, _ = sws.svd_truncate_and_perturb(
        W, lora_rank=8, truncation_policy="fixed", truncation_rank=3, energy_threshold=0.9,
        shrink_factor=shrink, residual_noise_scope="none", perturbation_sigma=0.0,
        generator=torch.Generator().manual_seed(0), output_dtype=torch.float64,
    )
    assert k1 == k2 == 3
    torch.testing.assert_close(A2[:k2, :], A1[:k1, :] * shrink, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(B2[:, :k2], B1[:, :k1] * shrink, rtol=1e-6, atol=1e-6)


# ---------- Missing-adapter handling ---------- #

def test_missing_adapter_skip_renormalizes_nash():
    sd1 = _make_adapter_sd(seed=10)
    with tempfile.TemporaryDirectory() as a_dir, tempfile.TemporaryDirectory() as bad_parent:
        _save_adapter_to_dir(sd1, a_dir)
        nonexistent = os.path.join(bad_parent, "missing_subdir")
        cfg = _TestCfg(missing_adapter_policy="skip", min_population_size=2, adapter_load_timeout_s=1, truncation_rank=4)
        state, meta = sws.build_warm_start_state_dict(
            lora_paths=[a_dir, nonexistent],
            nash_probs=[0.6, 0.4],
            cfg=cfg,
            lora_rank=8,
            model_dtype=torch.float32,
            seed=0,
        )
        # After renormalization, a_dir gets pi=1.0; expected W = B1 @ A1
        expected_W = sd1["base_model.model.foo.lora_B.weight"].to(torch.float64) @ sd1["base_model.model.foo.lora_A.weight"].to(torch.float64)
        A_new = state["base_model.model.foo.lora_A.default.weight"].to(torch.float64)
        B_new = state["base_model.model.foo.lora_B.default.weight"].to(torch.float64)
        k = int(meta["k_mean"])  # synthetic has 1 module so mean == actual k
        reconstructed = B_new[:, :k] @ A_new[:k, :]
        U, S, Vh = torch.linalg.svd(expected_W, full_matrices=False)
        topk = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
        torch.testing.assert_close(reconstructed, topk, rtol=1e-4, atol=1e-4)
        assert meta["n_adapters_skipped"] == 1
        assert meta["n_adapters_loaded"] == 1


def test_missing_adapter_raise():
    sd1 = _make_adapter_sd(seed=10)
    with tempfile.TemporaryDirectory() as a_dir, tempfile.TemporaryDirectory() as bad_parent:
        _save_adapter_to_dir(sd1, a_dir)
        nonexistent = os.path.join(bad_parent, "missing_subdir")
        cfg = _TestCfg(missing_adapter_policy="raise", adapter_load_timeout_s=1)
        with pytest.raises(Exception):
            sws.build_warm_start_state_dict(
                lora_paths=[a_dir, nonexistent], nash_probs=[0.5, 0.5], cfg=cfg,
                lora_rank=4, model_dtype=torch.float32, seed=0,
            )


# ---------- Key naming round-trip ---------- #

def test_key_naming_matches_deepspeed():
    sd1 = _make_adapter_sd(seed=20)
    with tempfile.TemporaryDirectory() as a_dir, tempfile.TemporaryDirectory() as b_dir:
        _save_adapter_to_dir(sd1, a_dir)
        _save_adapter_to_dir(_make_adapter_sd(seed=21), b_dir)
        cfg = _TestCfg(adapter_load_timeout_s=1)
        state, _ = sws.build_warm_start_state_dict(
            lora_paths=[a_dir, b_dir], nash_probs=[0.5, 0.5], cfg=cfg,
            lora_rank=4, model_dtype=torch.float32, seed=0,
        )
        keys = sorted(state.keys())
        assert keys == [
            "base_model.model.foo.lora_A.default.weight",
            "base_model.model.foo.lora_B.default.weight",
        ], f"unexpected keys: {keys}"


# ---------- NaN/Inf guards ---------- #

def test_nan_input_adapter_raises():
    sd = _make_adapter_sd(seed=40)
    sd["base_model.model.foo.lora_A.weight"][0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        sws.build_meta_lora([sd], [1.0], compute_dtype="float32")


def test_nan_W_meta_raises_in_svd():
    W = torch.full((4, 8), float("inf"))
    with pytest.raises(RuntimeError, match="non-finite"):
        sws.svd_truncate_and_perturb(
            W, lora_rank=4, truncation_policy="fixed", truncation_rank=2, energy_threshold=0.9,
            shrink_factor=1.0, residual_noise_scope="none", perturbation_sigma=0.0,
            generator=torch.Generator().manual_seed(0), output_dtype=torch.float32,
        )


def test_zero_W_meta_warns_no_crash(caplog):
    W = torch.zeros(4, 8)
    A_new, B_new, k, energy = sws.svd_truncate_and_perturb(
        W, lora_rank=4, truncation_policy="fixed", truncation_rank=2, energy_threshold=0.9,
        shrink_factor=1.0, residual_noise_scope="a_only", perturbation_sigma=0.01,
        generator=torch.Generator().manual_seed(0), output_dtype=torch.float32,
    )
    assert torch.all(A_new[:k, :] == 0)  # sqrt(0)=0
    assert torch.all(B_new[:, :k] == 0)
    assert energy == 0.0


# ---------- Seed reproducibility ---------- #

def test_seed_reproducibility():
    sd1 = _make_adapter_sd(seed=30)
    with tempfile.TemporaryDirectory() as a_dir, tempfile.TemporaryDirectory() as b_dir:
        _save_adapter_to_dir(sd1, a_dir)
        _save_adapter_to_dir(_make_adapter_sd(seed=31), b_dir)
        cfg = _TestCfg(residual_noise_scope="a_and_b", perturbation_sigma=0.1, adapter_load_timeout_s=1)
        s1, _ = sws.build_warm_start_state_dict(
            lora_paths=[a_dir, b_dir], nash_probs=[0.5, 0.5], cfg=cfg,
            lora_rank=4, model_dtype=torch.float32, seed=123,
        )
        s2, _ = sws.build_warm_start_state_dict(
            lora_paths=[a_dir, b_dir], nash_probs=[0.5, 0.5], cfg=cfg,
            lora_rank=4, model_dtype=torch.float32, seed=123,
        )
        for k in s1:
            torch.testing.assert_close(s1[k], s2[k])

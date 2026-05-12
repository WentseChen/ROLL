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
    log_spectrum_metrics: bool = False


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
    """shrink_factor scales the EFFECTIVE principal contribution (B @ A) by exactly
    shrink_factor — matches classical Shrink-and-Perturb W := lambda*W + epsilon.
    Per-factor scaling would give shrink_factor^2, which is wrong.
    """
    shrink = 0.5
    W = torch.randn(8, 16, generator=torch.Generator().manual_seed(11))
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
    reconstructed1 = B1[:, :k1] @ A1[:k1, :]
    reconstructed2 = B2[:, :k2] @ A2[:k2, :]
    torch.testing.assert_close(reconstructed2, shrink * reconstructed1, rtol=1e-6, atol=1e-6)


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


def test_zero_W_meta_warns_no_crash(monkeypatch):
    # ROLL's logger doesn't propagate to root; intercept logger.warning directly.
    seen = []
    monkeypatch.setattr(sws.logger, "warning", lambda msg, *a, **kw: seen.append(str(msg)))
    W = torch.zeros(4, 8)
    A_new, B_new, k, energy = sws.svd_truncate_and_perturb(
        W, lora_rank=4, truncation_policy="fixed", truncation_rank=2, energy_threshold=0.9,
        shrink_factor=1.0, residual_noise_scope="a_only", perturbation_sigma=0.01,
        generator=torch.Generator().manual_seed(0), output_dtype=torch.float32,
    )
    assert torch.all(A_new[:k, :] == 0)
    assert torch.all(B_new[:, :k] == 0)
    assert energy == 0.0
    assert any("zero-energy" in m for m in seen), f"expected zero-energy warning, got: {seen!r}"


# ---------- Spectrum metrics (Roy-Vetterli r_eff, PR, log_vol, subspace_pres) ---------- #

def test_spectrum_r_eff_uniform_equals_rank():
    """Uniform spectrum (all sigmas equal) → r_eff_pre == rank.

    With k<r_eff_pre, ratio = r_eff_post / r_eff_pre = 4/10 = 0.4 (fraction of
    effective rank preserved, not internal utilization).
    """
    S = torch.ones(10)
    Vh_top = torch.eye(10, 32)[:4]  # k=4
    m = sws._spectrum_module_metrics(S, Vh_top, k=4, module_path="m", population_adapters=[])
    assert m["r_eff_pre"] == pytest.approx(10.0, rel=1e-6)
    assert m["r_eff_post"] == pytest.approx(4.0, rel=1e-6)  # top-4 also uniform
    assert m["r_eff_ratio"] == pytest.approx(0.4, rel=1e-6)


def test_spectrum_r_eff_ratio_full_preservation_when_k_geq_pre():
    """When k >= r_eff_pre, no truncation needed → r_eff_ratio = 1.0."""
    S = torch.tensor([1.0, 1.0, 1.0, 1.0])  # uniform rank-4 spectrum
    Vh_top = torch.eye(4, 16)
    m = sws._spectrum_module_metrics(S, Vh_top, k=4, module_path="m", population_adapters=[])
    assert m["r_eff_pre"] == pytest.approx(4.0, rel=1e-6)
    assert m["r_eff_post"] == pytest.approx(4.0, rel=1e-6)
    assert m["r_eff_ratio"] == pytest.approx(1.0, rel=1e-6)


def test_spectrum_r_eff_peaky_equals_one():
    """sigma_1 dominates → r_eff_pre ≈ 1; truncation preserves it fully (ratio≈1)."""
    S = torch.tensor([100.0, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    Vh_top = torch.eye(8, 16)[:4]
    m = sws._spectrum_module_metrics(S, Vh_top, k=4, module_path="m", population_adapters=[])
    assert m["r_eff_pre"] < 1.01  # essentially rank-1
    assert m["r_eff_ratio"] == pytest.approx(1.0, abs=1e-2)  # nothing to lose
    assert m["pr_ratio"] == pytest.approx(1.0, abs=1e-3)


def test_spectrum_pr_uniform_equals_rank():
    """Uniform spectrum → PR == rank (energy-weighted soft rank)."""
    S = torch.ones(6)
    Vh_top = torch.eye(6, 16)[:3]
    m = sws._spectrum_module_metrics(S, Vh_top, k=3, module_path="m", population_adapters=[])
    # PR_full = (6)^2 / 6 = 6; PR_top = (3)^2 / 3 = 3; ratio = 0.5
    assert m["pr_ratio"] == pytest.approx(0.5, rel=1e-6)


def test_spectrum_log_vol_detects_collapse():
    """One sigma -> 0 makes log_vol diverge negatively, even if others are large."""
    S_healthy = torch.tensor([1.0, 1.0, 1.0, 1.0])
    Vh_top = torch.eye(4, 16)[:4]
    m_healthy = sws._spectrum_module_metrics(S_healthy, Vh_top, k=4, module_path="m", population_adapters=[])
    S_collapsed = torch.tensor([1.0, 1.0, 1.0, 1e-10])
    m_collapsed = sws._spectrum_module_metrics(S_collapsed, Vh_top, k=4, module_path="m", population_adapters=[])
    assert m_healthy["log_vol"] == pytest.approx(0.0, abs=1e-6)
    assert m_collapsed["log_vol"] < -20.0  # log(1e-10) = -23


def test_spectrum_subspace_pres_identity_pop():
    """When meta-LoRA row-space exactly contains a population member's row-space, pres=1."""
    d_in = 32
    # Member j has rank-2 row-space along axes 0 and 1.
    A_j = torch.zeros(4, d_in)
    A_j[0, 0] = 1.0
    A_j[1, 1] = 1.0
    sd_j = {"base_model.model.m.lora_A.weight": A_j, "base_model.model.m.lora_B.weight": torch.eye(8, 4)}
    # Meta row-space = first 4 axes (includes the member's first 2).
    Vh_top = torch.zeros(4, d_in)
    for i in range(4):
        Vh_top[i, i] = 1.0
    S = torch.tensor([1.0, 1.0, 1.0, 1.0])
    m = sws._spectrum_module_metrics(S, Vh_top, k=4, module_path="base_model.model.m", population_adapters=[sd_j])
    assert m["subspace_pres_min"] == pytest.approx(1.0, abs=1e-5)


def test_spectrum_subspace_pres_orthogonal_pop():
    """When meta and member row-spaces are orthogonal, pres=0."""
    d_in = 32
    # Member rank-2 along axes 10, 11.
    A_j = torch.zeros(4, d_in)
    A_j[0, 10] = 1.0
    A_j[1, 11] = 1.0
    sd_j = {"base_model.model.m.lora_A.weight": A_j, "base_model.model.m.lora_B.weight": torch.eye(8, 4)}
    # Meta row-space = first 4 axes (orthogonal to member).
    Vh_top = torch.zeros(4, d_in)
    for i in range(4):
        Vh_top[i, i] = 1.0
    S = torch.tensor([1.0, 1.0, 1.0, 1.0])
    m = sws._spectrum_module_metrics(S, Vh_top, k=4, module_path="base_model.model.m", population_adapters=[sd_j])
    assert m["subspace_pres_min"] == pytest.approx(0.0, abs=1e-5)


def test_spectrum_subspace_pres_orthogonal_member_larger_than_k():
    """Regression: r_j > k_eff orthogonal case. Earlier code dropped (r_j - k) silently-
    orthogonal directions and gave pres ≈ 0.29 instead of 0."""
    d_in = 32
    # Member rank-8 (r_j) along axes 10..17 (orthogonal to meta).
    A_j = torch.zeros(8, d_in)
    for i in range(8):
        A_j[i, 10 + i] = 1.0
    sd_j = {"base_model.model.m.lora_A.weight": A_j, "base_model.model.m.lora_B.weight": torch.eye(8, 8)}
    # Meta row-space = first 4 axes (k=4, r_j=8 → r_j > k).
    Vh_top = torch.zeros(4, d_in)
    for i in range(4):
        Vh_top[i, i] = 1.0
    S = torch.tensor([1.0, 1.0, 1.0, 1.0])
    m = sws._spectrum_module_metrics(S, Vh_top, k=4, module_path="base_model.model.m", population_adapters=[sd_j])
    assert m["subspace_pres_min"] == pytest.approx(0.0, abs=1e-5)


def test_spectrum_subspace_pres_contained_member_smaller_than_k():
    """Q_j ⊂ Q_meta (r_j < k). pres should be 1."""
    d_in = 32
    A_j = torch.zeros(4, d_in)
    A_j[0, 0] = 1.0
    A_j[1, 2] = 1.0  # member is rank-2 along axes 0, 2
    sd_j = {"base_model.model.m.lora_A.weight": A_j, "base_model.model.m.lora_B.weight": torch.eye(8, 4)}
    # Meta covers axes 0,1,2,3 (k=4 > r_j=2, member's rows live in meta)
    Vh_top = torch.zeros(4, d_in)
    for i in range(4):
        Vh_top[i, i] = 1.0
    S = torch.tensor([1.0, 1.0, 1.0, 1.0])
    m = sws._spectrum_module_metrics(S, Vh_top, k=4, module_path="base_model.model.m", population_adapters=[sd_j])
    assert m["subspace_pres_min"] == pytest.approx(1.0, abs=1e-5)


def test_spectrum_metrics_integration_via_build_warm_start():
    """End-to-end: log_spectrum_metrics=True surfaces spectrum/* keys in metadata."""
    sd1 = _make_adapter_sd(seed=50)
    with tempfile.TemporaryDirectory() as a_dir, tempfile.TemporaryDirectory() as b_dir:
        _save_adapter_to_dir(sd1, a_dir)
        _save_adapter_to_dir(_make_adapter_sd(seed=51), b_dir)
        cfg = _TestCfg(adapter_load_timeout_s=1, log_spectrum_metrics=True)
        _state, meta = sws.build_warm_start_state_dict(
            lora_paths=[a_dir, b_dir], nash_probs=[0.5, 0.5], cfg=cfg,
            lora_rank=4, model_dtype=torch.float32, seed=0,
        )
        for key in (
            "spectrum/r_eff_pre", "spectrum/r_eff_post", "spectrum/r_eff_ratio",
            "spectrum/pr_ratio", "spectrum/log_vol", "spectrum/subspace_pres_min",
        ):
            assert key in meta, f"missing {key} in metadata"
        # r_eff_ratio is fraction of effective rank preserved, in [0, 1+eps]
        assert 0.0 <= meta["spectrum/r_eff_ratio"] <= 1.05
        # Truncation discards some effective rank (r_eff_pre > k or close to k);
        # r_eff_post <= r_eff_pre by construction
        assert meta["spectrum/r_eff_post"] <= meta["spectrum/r_eff_pre"] + 1e-6
        # subspace_pres_min should be in [0, 1]
        assert 0.0 <= meta["spectrum/subspace_pres_min"] <= 1.0


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

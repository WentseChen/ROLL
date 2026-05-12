"""SVD-empowered PSRO warm start for LoRA adapters.

Phase 1 — Knowledge Condensation: Nash-weighted meta-LoRA aggregation
    W_meta^(m) = sum_i pi_i * (B_i^(m) @ A_i^(m))

Phase 2 — Meta-SVD Truncation: top-k by fixed-rank or energy threshold.

Phase 3 — Shrink-and-Perturb Refactorization: principal slice scaled by
    shrink_factor; residual slice filled with N(0, sigma^2) per
    residual_noise_scope.

All public functions are pure CPU / single-process; no Ray, no DeepSpeed.
The output state dict re-inserts the ".default." infix (peft active-adapter
name) so the actor can look up keys directly off model.module.named_parameters().
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional

import numpy as np
import torch
from safetensors.torch import load_file

from roll.utils.logging import get_logger

logger = get_logger()


def load_adapter_state_dict(ckpt_dir: Optional[str], max_wait_s: int) -> Optional[dict]:
    """Load <ckpt_dir>/adapter_model.safetensors with retry on in-flight uploads.

    Returns None when ckpt_dir is None (base-model placeholder).
    """
    if ckpt_dir is None:
        return None
    path = os.path.join(ckpt_dir, "adapter_model.safetensors")
    deadline = time.monotonic() + max_wait_s
    last_err: Optional[BaseException] = None
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            return load_file(path)
        except (FileNotFoundError, OSError, RuntimeError) as e:
            last_err = e
            wait = min(2 ** min(attempt, 4), 10)
            logger.warning(f"load_adapter_state_dict: attempt {attempt} failed for {path}: {e}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"load_adapter_state_dict: never succeeded for {path}: {last_err}")


def group_lora_pairs_by_module(state_dict: dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Parse PEFT on-disk LoRA keys into per-module (A, B) pairs.

    On-disk key format (after export_peft_adapter strips .default.):
        base_model.model.<path>.lora_A.weight
        base_model.model.<path>.lora_B.weight
    Returns {module_path: (A, B)} with A shape (r, d_in), B shape (d_out, r).
    """
    parts: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if key.endswith(".lora_A.weight"):
            module = key[: -len(".lora_A.weight")]
            parts.setdefault(module, {})["A"] = tensor
        elif key.endswith(".lora_B.weight"):
            module = key[: -len(".lora_B.weight")]
            parts.setdefault(module, {})["B"] = tensor
    return {m: (d["A"], d["B"]) for m, d in parts.items() if "A" in d and "B" in d}


def _resolve_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def build_meta_lora(
    adapters: list[Optional[dict]],
    nash_probs: list[float],
    compute_dtype: str = "float32",
) -> dict[str, torch.Tensor]:
    """Nash-weighted sum W_meta = sum_i pi_i * (B_i @ A_i) per module.

    None entries (base model) contribute Delta W = 0 but their pi *is consumed* —
    base mass dilutes the principal slice rather than being renormalized away.
    This matches the spec: W_meta = Sigma_i pi_i * (B_i @ A_i) over the full
    population including the base placeholder.

    NaN/Inf in any input adapter raises; SVD on NaN silently returns NaN and
    would otherwise corrupt the actor's live LoRA weights.
    """
    if len(adapters) != len(nash_probs):
        raise ValueError(f"adapters/nash_probs length mismatch: {len(adapters)} vs {len(nash_probs)}")
    dtype = _resolve_dtype(compute_dtype)
    W_meta: dict[str, torch.Tensor] = {}
    for idx, (prob, sd) in enumerate(zip(nash_probs, adapters)):
        if sd is None:
            continue
        for module, (A, B) in group_lora_pairs_by_module(sd).items():
            if not (torch.isfinite(A).all() and torch.isfinite(B).all()):
                raise RuntimeError(
                    f"svd_warm_start: non-finite values in adapter idx={idx} module={module}; "
                    f"refusing to propagate NaN/Inf into LoRA params."
                )
            contrib = float(prob) * (B.to(dtype) @ A.to(dtype))
            if module in W_meta:
                W_meta[module] = W_meta[module] + contrib
            else:
                W_meta[module] = contrib
    return W_meta


def pick_truncation_rank(
    singular_values: torch.Tensor,
    truncation_policy: str,
    truncation_rank: int,
    energy_threshold: float,
    lora_rank: int,
) -> tuple[int, float]:
    """Return (k, energy_retained) per the configured policy.

    Clipped to [1, lora_rank - 1] for both policies so the residual slice has
    at least one rank slot available for noise injection.
    """
    sv = singular_values.to(torch.float64)
    energy = (sv ** 2)
    total = float(energy.sum().item())
    upper = max(1, lora_rank - 1)

    # Also clip k to the number of singular values returned. For non-square LoRA
    # targets where min(d_out, d_in) < lora_rank-1 (rare but possible), this avoids
    # a shape mismatch when assigning B_new[:, :k] = B_top.
    upper = max(1, min(upper, int(sv.numel())))

    if truncation_policy == "fixed":
        k = max(1, min(truncation_rank, upper))
    elif truncation_policy == "energy":
        if total == 0.0:
            return 1, 0.0
        cumfrac = torch.cumsum(energy, dim=0) / total
        meets = (cumfrac >= energy_threshold).nonzero(as_tuple=True)[0]
        k = int(meets[0].item()) + 1 if len(meets) > 0 else int(len(sv))
        k = max(1, min(k, upper))
    else:
        raise ValueError(f"unknown truncation_policy: {truncation_policy}")

    retained = float(energy[:k].sum().item() / total) if total > 0 else 0.0
    return k, retained


def _truncate_with_aux(
    W_meta: torch.Tensor,
    lora_rank: int,
    truncation_policy: str,
    truncation_rank: int,
    energy_threshold: float,
    shrink_factor: float,
    residual_noise_scope: str,
    perturbation_sigma: float,
    generator: Optional[torch.Generator],
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int, float, torch.Tensor, torch.Tensor]:
    """Internal: same as svd_truncate_and_perturb, but also returns full S and top-k Vh
    (needed for spectrum metrics)."""
    d_out, d_in = W_meta.shape
    if not torch.isfinite(W_meta).all():
        raise RuntimeError("svd_warm_start._truncate_with_aux: non-finite W_meta.")
    U, S, Vh = torch.linalg.svd(W_meta, full_matrices=False)  # U:(d_out,p), S:(p,), Vh:(p,d_in)
    if not (torch.isfinite(S).all() and torch.isfinite(U).all() and torch.isfinite(Vh).all()):
        raise RuntimeError("svd_warm_start._truncate_with_aux: SVD produced non-finite outputs.")
    if float(S.sum().item()) == 0.0:
        logger.warning(
            f"svd_warm_start: zero-energy spectrum for module of shape {tuple(W_meta.shape)}; "
            f"output will be pure noise (or zero if residual_noise_scope=='none')."
        )
    k, energy_retained = pick_truncation_rank(
        S, truncation_policy, truncation_rank, energy_threshold, lora_rank,
    )

    # Apply sqrt(shrink_factor) to each factor so the effective principal
    # contribution (B_top @ A_top) is scaled by shrink_factor exactly once —
    # matches classical Shrink-and-Perturb (W := lambda * W + epsilon).
    sqrt_S_top = torch.sqrt(S[:k].clamp(min=0.0))
    sqrt_shrink = math.sqrt(shrink_factor) if shrink_factor > 0 else 0.0
    B_top = U[:, :k] * sqrt_S_top.unsqueeze(0) * sqrt_shrink                # (d_out, k)
    A_top = sqrt_S_top.unsqueeze(1) * Vh[:k, :] * sqrt_shrink               # (k, d_in)

    A_new = torch.zeros((lora_rank, d_in), dtype=W_meta.dtype, device=W_meta.device)
    B_new = torch.zeros((d_out, lora_rank), dtype=W_meta.dtype, device=W_meta.device)
    A_new[:k, :] = A_top
    B_new[:, :k] = B_top

    if k < lora_rank and perturbation_sigma > 0 and residual_noise_scope != "none":
        residual_shape_a = (lora_rank - k, d_in)
        residual_shape_b = (d_out, lora_rank - k)
        if residual_noise_scope in ("a_only", "a_and_b"):
            A_new[k:, :] = torch.randn(residual_shape_a, generator=generator, dtype=W_meta.dtype) * perturbation_sigma
        if residual_noise_scope == "a_and_b":
            B_new[:, k:] = torch.randn(residual_shape_b, generator=generator, dtype=W_meta.dtype) * perturbation_sigma

    return A_new.to(output_dtype), B_new.to(output_dtype), k, energy_retained, S, Vh[:k, :]


def svd_truncate_and_perturb(
    W_meta: torch.Tensor,
    lora_rank: int,
    truncation_policy: str,
    truncation_rank: int,
    energy_threshold: float,
    shrink_factor: float,
    residual_noise_scope: str,
    perturbation_sigma: float,
    generator: Optional[torch.Generator],
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """Per-module SVD + Phase-3 refactorization.

    Returns (A_new, B_new, k, energy_retained) where
        A_new: (lora_rank, d_in)
        B_new: (d_out, lora_rank)
    """
    A_new, B_new, k, energy_retained, _S, _Vh_top = _truncate_with_aux(
        W_meta, lora_rank, truncation_policy, truncation_rank, energy_threshold,
        shrink_factor, residual_noise_scope, perturbation_sigma, generator, output_dtype,
    )
    return A_new, B_new, k, energy_retained


# ---------- Spectrum metrics (literature-grounded "volume of information preserved") ----------
#
# These supplement energy_retained, which is dominated by sigma_1 and hides whether
# truncation kept enough effective rank / population-direction diversity. References:
#   - Roy & Vetterli (2007), "The effective rank: A measure of effective dimensionality"
#   - Davis-Kahan / Wedin sin-Theta theorems
#   - Participation ratio from Anderson localization / Goldt et al. neural net dim


def _build_row_basis(A: torch.Tensor, rtol: float = 1e-6) -> torch.Tensor:
    """Orthonormal basis of row(A) as a (d_in, r_eff) matrix.

    A has shape (r, d_in); rows are not assumed orthonormal. The row-space is
    spanned by right singular vectors with non-zero singular values.
    Returns an empty (d_in, 0) tensor when A has no significant content.
    """
    if A.numel() == 0:
        return A.new_zeros((A.shape[-1], 0))
    _U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    if S.numel() == 0:
        return A.new_zeros((A.shape[-1], 0))
    s_max = float(S[0].item())
    if s_max == 0.0:
        return A.new_zeros((A.shape[-1], 0))
    tol = rtol * s_max
    r_eff = int((S > tol).sum().item())
    return Vh[:r_eff, :].T  # (d_in, r_eff)


def _spectrum_module_metrics(
    S_full: torch.Tensor,
    Vh_top: torch.Tensor,
    k: int,
    module_path: str,
    population_adapters: list[Optional[dict]],
) -> dict[str, float]:
    """Six "volume of information preserved" scalars for one module.

    See module-level docstring for the metric set and citations.

    Args:
        S_full: full singular spectrum of W_meta, shape (p,).
        Vh_top: top-k right singular vectors of W_meta, shape (k, d_in).
        k: truncation rank actually used.
        module_path: LoRA module key (for fetching per-member A_j).
        population_adapters: list of state_dicts (or None for base placeholder).

    Returns:
        Dict with keys:
          r_eff_pre, r_eff_post, r_eff_ratio, pr_ratio, log_vol, subspace_pres_min
    """
    eps = 1e-12
    sv = S_full.to(torch.float64).clamp(min=0.0)
    sv_pos = sv[sv > eps]
    if sv_pos.numel() == 0:
        # Degenerate: zero-energy meta-LoRA. Conventional defaults so downstream
        # aggregation doesn't NaN out.
        return {
            "r_eff_pre": 0.0,
            "r_eff_post": 0.0,
            "r_eff_ratio": 0.0,
            "pr_ratio": 0.0,
            "log_vol": 0.0,
            "subspace_pres_min": 1.0,
        }

    k_eff = min(int(k), int(sv_pos.numel()))

    # 1) Effective rank (Roy-Vetterli) on the full pre-truncation spectrum.
    p_full = sv_pos / sv_pos.sum()
    H_pre = -(p_full * torch.log(p_full.clamp(min=eps))).sum()
    r_eff_pre = float(torch.exp(H_pre))

    # 2) Effective rank on the post-truncation kept top-k.
    sv_top = sv_pos[:k_eff]
    p_top = sv_top / sv_top.sum().clamp(min=eps)
    H_post = -(p_top * torch.log(p_top.clamp(min=eps))).sum()
    r_eff_post = float(torch.exp(H_post))

    # 3) Primary headline: fraction of effective rank preserved across truncation.
    #    r_eff_post / r_eff_pre directly answers "how much of the original effective
    #    spectrum survived?" Goes to 1.0 only if r_eff_pre <= k (no truncation
    #    needed); otherwise drops proportionally to how much was discarded.
    r_eff_ratio = r_eff_post / max(r_eff_pre, eps)

    # 4) Participation ratio (L2-energy weighted soft rank). Disagreement with
    #    r_eff_ratio reveals sigma_1 dominance / peakiness.
    e_full = sv_pos.pow(2)
    e_top = sv_top.pow(2)
    pr_full = float((e_full.sum() ** 2) / e_full.pow(2).sum().clamp(min=eps))
    pr_top = float((e_top.sum() ** 2) / e_top.pow(2).sum().clamp(min=eps))
    pr_ratio = pr_top / max(pr_full, eps)

    # 5) Log-volume on kept dims. Catches multiplicative collapse (any sigma_i -> 0).
    log_vol = float(torch.log(sv_top.clamp(min=eps)).sum())

    # 6) Subspace preservation per population member (Davis-Kahan principal angles).
    #    For each loaded adapter j, compare row(top_k(W_meta)) to row(A_j).
    Q_meta = Vh_top.to(torch.float64).T  # (d_in, k_eff_basis); columns orthonormal
    # Truncate to k_eff columns to match the actual non-zero subspace dim.
    Q_meta = Q_meta[:, :k_eff] if Q_meta.shape[1] > k_eff else Q_meta
    pres_per_member: list[float] = []
    for sd in population_adapters:
        if sd is None:
            continue  # base placeholder contributes Delta W = 0; no row-space
        pairs = group_lora_pairs_by_module(sd)
        if module_path not in pairs:
            continue
        A_j, _B_j = pairs[module_path]
        Q_j = _build_row_basis(A_j.to(torch.float64))  # (d_in, r_j)
        if Q_j.shape[1] == 0:
            continue
        # Principal angles via SVD of cross-Gram: svd(Q_meta^T @ Q_j) -> cos(theta_i).
        # There are d_eff = min(k_eff, r_j) principal angles. When r_j > k_eff, the
        # remaining (r_j - k_eff) directions of Q_j are orthogonal to Q_meta by
        # construction and contribute sin^2 = 1 each to the Frobenius sin-norm.
        # Hence: ||sin Theta||_F^2 (full) = r_j - sum(cos^2) — covers both r_j <= k
        # and r_j > k uniformly. Normalize by r_j so pres in [0, 1]:
        #   pres = 1 iff Q_j is entirely contained in Q_meta (all cos = 1);
        #   pres = 0 iff Q_j is orthogonal to Q_meta (all cos = 0).
        M = Q_meta.T @ Q_j  # (k_eff, r_j)
        cos_vals = torch.linalg.svdvals(M).clamp(min=0.0, max=1.0)
        r_j = int(Q_j.shape[1])
        sin_sq_sum_full = max(0.0, float(r_j) - float((cos_vals ** 2).sum()))
        pres = 1.0 - (sin_sq_sum_full / max(r_j, 1)) ** 0.5
        pres_per_member.append(max(0.0, min(1.0, pres)))

    subspace_pres_min = float(min(pres_per_member)) if pres_per_member else 1.0

    return {
        "r_eff_pre": r_eff_pre,
        "r_eff_post": r_eff_post,
        "r_eff_ratio": r_eff_ratio,
        "pr_ratio": pr_ratio,
        "log_vol": log_vol,
        "subspace_pres_min": subspace_pres_min,
    }


def _renormalize(probs: list[float], keep_mask: list[bool]) -> list[float]:
    kept = [p for p, m in zip(probs, keep_mask) if m]
    total = sum(kept)
    if total <= 0:
        raise RuntimeError("missing_adapter_policy=skip: remaining Nash weights sum to 0 after skipping.")
    return [p / total if m else 0.0 for p, m in zip(probs, keep_mask)]


def _to_in_model_keys(module_path: str) -> tuple[str, str]:
    """Re-insert .default. infix for in-model named_parameters() lookup."""
    return (
        f"{module_path}.lora_A.default.weight",
        f"{module_path}.lora_B.default.weight",
    )


def build_warm_start_state_dict(
    lora_paths: list[Optional[str]],
    nash_probs,
    cfg,
    lora_rank: int,
    model_dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Top-level entry: load adapters, build meta-LoRA, SVD-truncate-perturb,
    pack into a state dict keyed by in-model named_parameters() names.

    cfg: SVDWarmStartConfig instance (duck-typed; accessed as attributes).
    Returns (state_dict, metadata).
    """
    nash_probs = list(nash_probs) if not isinstance(nash_probs, list) else nash_probs
    if len(lora_paths) != len(nash_probs):
        raise ValueError(f"lora_paths/nash_probs length mismatch: {len(lora_paths)} vs {len(nash_probs)}")

    # Load adapters with missing-policy handling.
    adapters: list[Optional[dict]] = []
    keep_mask: list[bool] = []
    n_skipped = 0
    for path in lora_paths:
        if path is None:
            adapters.append(None)
            keep_mask.append(True)  # base contributes 0 but its pi counts
            continue
        try:
            sd = load_adapter_state_dict(path, max_wait_s=cfg.adapter_load_timeout_s)
            adapters.append(sd)
            keep_mask.append(True)
        except Exception as e:
            if cfg.missing_adapter_policy == "raise":
                raise
            logger.warning(f"svd_warm_start: skipping missing/corrupt adapter {path}: {e}")
            adapters.append(None)
            keep_mask.append(False)
            n_skipped += 1

    effective_probs = _renormalize(nash_probs, keep_mask) if n_skipped > 0 else nash_probs

    W_meta = build_meta_lora(adapters, effective_probs, compute_dtype=cfg.compute_dtype)
    if not W_meta:
        raise RuntimeError("svd_warm_start: build_meta_lora produced no modules (all adapters skipped or empty).")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    log_spectrum = bool(getattr(cfg, "log_spectrum_metrics", False))
    spectrum_metric_keys = (
        "r_eff_pre", "r_eff_post", "r_eff_ratio", "pr_ratio", "log_vol", "subspace_pres_min",
    )

    state_dict: dict[str, torch.Tensor] = {}
    k_values: list[int] = []
    energies: list[float] = []
    spectrum_per_module: dict[str, list[float]] = {k: [] for k in spectrum_metric_keys}
    for module, W in W_meta.items():
        if log_spectrum:
            A_new, B_new, k, energy_retained, S_full, Vh_top = _truncate_with_aux(
                W,
                lora_rank=lora_rank,
                truncation_policy=cfg.truncation_policy,
                truncation_rank=cfg.truncation_rank,
                energy_threshold=cfg.energy_threshold,
                shrink_factor=cfg.shrink_factor,
                residual_noise_scope=cfg.residual_noise_scope,
                perturbation_sigma=cfg.perturbation_sigma,
                generator=generator,
                output_dtype=model_dtype,
            )
            module_metrics = _spectrum_module_metrics(
                S_full=S_full,
                Vh_top=Vh_top,
                k=k,
                module_path=module,
                population_adapters=adapters,
            )
            for mk in spectrum_metric_keys:
                spectrum_per_module[mk].append(module_metrics[mk])
        else:
            A_new, B_new, k, energy_retained = svd_truncate_and_perturb(
                W,
                lora_rank=lora_rank,
                truncation_policy=cfg.truncation_policy,
                truncation_rank=cfg.truncation_rank,
                energy_threshold=cfg.energy_threshold,
                shrink_factor=cfg.shrink_factor,
                residual_noise_scope=cfg.residual_noise_scope,
                perturbation_sigma=cfg.perturbation_sigma,
                generator=generator,
                output_dtype=model_dtype,
            )
        a_key, b_key = _to_in_model_keys(module)
        state_dict[a_key] = A_new
        state_dict[b_key] = B_new
        k_values.append(k)
        energies.append(energy_retained)

    n_loaded = sum(1 for sd in adapters if sd is not None)
    k_arr = np.asarray(k_values, dtype=np.float64)
    e_arr = np.asarray(energies, dtype=np.float64)
    metadata = {
        "n_modules": float(len(W_meta)),
        "n_adapters_loaded": float(n_loaded),
        "n_adapters_skipped": float(n_skipped),
        "n_population": float(len(lora_paths)),
        "k_mean": float(k_arr.mean()),
        "k_min": float(k_arr.min()),
        "k_max": float(k_arr.max()),
        "energy_retained_mean": float(e_arr.mean()),
        "energy_retained_min": float(e_arr.min()),
        "energy_retained_max": float(e_arr.max()),
    }
    if log_spectrum:
        for mk in spectrum_metric_keys:
            arr = np.asarray(spectrum_per_module[mk], dtype=np.float64)
            if arr.size == 0:
                continue
            metadata[f"spectrum/{mk}"] = float(arr.mean())
    return state_dict, metadata

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

    None entries (base model) contribute Delta W = 0 but their pi is consumed.
    Caller is responsible for renormalizing nash_probs after dropping any
    missing adapters; this function assumes the given (adapters, nash_probs)
    are already aligned and renormalized.
    """
    if len(adapters) != len(nash_probs):
        raise ValueError(f"adapters/nash_probs length mismatch: {len(adapters)} vs {len(nash_probs)}")
    dtype = _resolve_dtype(compute_dtype)
    W_meta: dict[str, torch.Tensor] = {}
    for prob, sd in zip(nash_probs, adapters):
        if sd is None:
            continue
        for module, (A, B) in group_lora_pairs_by_module(sd).items():
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
    d_out, d_in = W_meta.shape
    U, S, Vh = torch.linalg.svd(W_meta, full_matrices=False)  # U:(d_out,p), S:(p,), Vh:(p,d_in)
    k, energy_retained = pick_truncation_rank(
        S, truncation_policy, truncation_rank, energy_threshold, lora_rank,
    )

    sqrt_S_top = torch.sqrt(S[:k].clamp(min=0.0))
    B_top = U[:, :k] * sqrt_S_top.unsqueeze(0) * shrink_factor              # (d_out, k)
    A_top = sqrt_S_top.unsqueeze(1) * Vh[:k, :] * shrink_factor             # (k, d_in)

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

    return A_new.to(output_dtype), B_new.to(output_dtype), k, energy_retained


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

    state_dict: dict[str, torch.Tensor] = {}
    k_values: list[int] = []
    energies: list[float] = []
    for module, W in W_meta.items():
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
    return state_dict, metadata

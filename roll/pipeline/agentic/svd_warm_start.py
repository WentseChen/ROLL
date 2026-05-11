"""SVD warm-start for PSRO LoRA adapters.

Phase 1: Nash-weighted meta-LoRA aggregation.
Phase 2: SVD truncation (fixed rank or energy threshold).
Phase 3: Shrink-and-Perturb refactorization into (A_new, B_new).

All functions are pure CPU / single-process; no Ray or DeepSpeed dependencies.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from roll.utils.logging import get_logger

logger = get_logger()


def load_adapter_state_dict(ckpt_dir: Optional[str], max_wait_s: int) -> Optional[dict]:
    """Load adapter_model.safetensors from ckpt_dir with retry logic.

    Returns None for base model (ckpt_dir is None).
    """
    if ckpt_dir is None:
        return None

    from safetensors.torch import load_file

    path = os.path.join(ckpt_dir, "adapter_model.safetensors")
    deadline = time.monotonic() + max_wait_s
    last_err = None
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            return load_file(path)
        except (FileNotFoundError, OSError, Exception) as e:
            last_err = e
            wait = min(2 ** min(attempt, 4), 10)
            logger.warning(f"load_adapter_state_dict: attempt {attempt} failed for {path}: {e}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"load_adapter_state_dict: never succeeded for {path}: {last_err}")


def group_lora_pairs_by_module(state_dict: dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Parse PEFT adapter keys into per-module (A, B) pairs.

    PEFT key format: base_model.model.<path>.lora_A.weight / lora_B.weight
    Returns: {module_path: (A, B)} where A is (r, d_in), B is (d_out, r).
    """
    modules: dict[str, dict] = {}
    for k, v in state_dict.items():
        if "lora_A" in k and k.endswith(".weight"):
            module = k[: k.rindex(".lora_A.weight")]
            modules.setdefault(module, {})["A"] = v
        elif "lora_B" in k and k.endswith(".weight"):
            module = k[: k.rindex(".lora_B.weight")]
            modules.setdefault(module, {})["B"] = v
    return {m: (d["A"], d["B"]) for m, d in modules.items() if "A" in d and "B" in d}


def build_meta_lora(
    adapters: list[Optional[dict]],
    nash_probs: list[float],
    compute_dtype: str,
) -> dict[str, torch.Tensor]:
    """Compute Nash-weighted W_meta = Σ_i π_i · (B_i · A_i) per module.

    adapters: list of state_dicts (or None for base model, which contributes ΔW = 0).
    Returns: {module_path: W_meta tensor of shape (d_out, d_in)}.
    """
    dtype = torch.float64 if compute_dtype == "float64" else torch.float32
    W_meta: dict[str, torch.Tensor] = {}
    for prob, sd in zip(nash_probs, adapters):
        if sd is None:
            continue
        pairs = group_lora_pairs_by_module(sd)
        for module, (A, B) in pairs.items():
            delta = prob * (B.to(dtype) @ A.to(dtype))
            if module in W_meta:
                W_meta[module] = W_meta[module] + delta
            else:
                W_meta[module] = delta
    return W_meta


def pick_truncation_rank(singular_values: torch.Tensor, truncation_policy: str, truncation_rank: int, energy_threshold: float, lora_rank: int) -> int:
    """Return truncation rank k given singular values and config knobs."""
    if truncation_policy == "fixed":
        return truncation_rank
    # energy policy: smallest k s.t. top-k captures energy_threshold fraction
    total = (singular_values ** 2).sum().item()
    if total == 0:
        return 1
    cumsum = torch.cumsum(singular_values ** 2, dim=0)
    indices = (cumsum / total >= energy_threshold).nonzero(as_tuple=True)[0]
    k = int(indices[0].item()) + 1 if len(indices) > 0 else len(singular_values)
    return max(1, min(k, lora_rank - 1))


def svd_truncate_and_perturb(
    W_meta: torch.Tensor,
    lora_rank: int,
    truncation_policy: str,
    truncation_rank: int,
    energy_threshold: float,
    shrink_factor: float,
    residual_noise_scope: str,
    perturbation_sigma: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """SVD + Shrink-and-Perturb for one module.

    Returns: (A_new, B_new, k, energy_retained)
    A_new shape: (r, d_in), B_new shape: (d_out, r).
    """
    U, S, Vh = torch.linalg.svd(W_meta, full_matrices=False)
    k = pick_truncation_rank(S, truncation_policy, truncation_rank, energy_threshold, lora_rank)

    total_energy = (S ** 2).sum().item()
    retained_energy = (S[:k] ** 2).sum().item()
    energy_retained = retained_energy / total_energy if total_energy > 0 else 1.0

    d_out, d_in = W_meta.shape
    B_new = torch.zeros(d_out, lora_rank, dtype=W_meta.dtype)
    A_new = torch.zeros(lora_rank, d_in, dtype=W_meta.dtype)

    sqrt_S = torch.sqrt(S[:k])
    B_new[:, :k] = U[:, :k] * sqrt_S.unsqueeze(0) * shrink_factor
    A_new[:k, :] = sqrt_S.unsqueeze(1) * Vh[:k, :] * shrink_factor

    if residual_noise_scope != "none" and lora_rank > k:
        sigma = perturbation_sigma
        if residual_noise_scope in ("a_only", "a_and_b"):
            A_new[k:, :] = torch.normal(
                mean=torch.zeros(lora_rank - k, d_in, dtype=W_meta.dtype),
                std=sigma,
                generator=generator,
            )
        if residual_noise_scope == "a_and_b":
            B_new[:, k:] = torch.normal(
                mean=torch.zeros(d_out, lora_rank - k, dtype=W_meta.dtype),
                std=sigma,
                generator=generator,
            )

    return A_new, B_new, k, energy_retained


def build_warm_start_state_dict(
    lora_paths: list[Optional[str]],
    nash_probs: np.ndarray,
    cfg: "SVDWarmStartConfig",
    lora_rank: int,
    model_dtype: torch.dtype,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Top-level entry: produce warm-start LoRA state dict from FSP population.

    lora_paths: list of checkpoint dirs (None = base model).
    nash_probs: Nash mixed strategy, one weight per path.
    Returns: (state_dict, metadata)
      state_dict keys use DeepSpeed in-model naming:
        base_model.model.<path>.lora_{A,B}.default.weight
    """
    assert len(lora_paths) == len(nash_probs), "lora_paths and nash_probs must have same length"

    # Load adapters, handling missing ones per policy
    probs = list(nash_probs)
    adapters: list[Optional[dict]] = []
    valid_probs: list[float] = []
    n_skipped = 0

    for path, prob in zip(lora_paths, probs):
        try:
            sd = load_adapter_state_dict(path, cfg.adapter_load_timeout_s)
            adapters.append(sd)
            valid_probs.append(prob)
        except Exception as e:
            if cfg.missing_adapter_policy == "raise":
                raise
            logger.warning(f"svd_warm_start: skipping adapter {path}: {e}")
            n_skipped += 1

    if not adapters:
        raise RuntimeError("svd_warm_start: all adapters failed to load")

    # Renormalize Nash weights over remaining adapters
    total = sum(valid_probs)
    if total <= 0:
        raise RuntimeError("svd_warm_start: all Nash weights are zero after skipping")
    norm_probs = [p / total for p in valid_probs]

    compute_dtype = cfg.compute_dtype
    W_meta_map = build_meta_lora(adapters, norm_probs, compute_dtype)

    if not W_meta_map:
        raise RuntimeError("svd_warm_start: no LoRA modules found in population adapters")

    generator = torch.Generator()
    generator.manual_seed(seed + cfg.seed_offset)

    state_dict: dict[str, torch.Tensor] = {}
    k_per_module: list[int] = []
    energy_per_module: list[float] = []

    for module, W_meta in W_meta_map.items():
        A_new, B_new, k, energy = svd_truncate_and_perturb(
            W_meta=W_meta,
            lora_rank=lora_rank,
            truncation_policy=cfg.truncation_policy,
            truncation_rank=cfg.truncation_rank,
            energy_threshold=cfg.energy_threshold,
            shrink_factor=cfg.shrink_factor,
            residual_noise_scope=cfg.residual_noise_scope,
            perturbation_sigma=cfg.perturbation_sigma,
            generator=generator,
        )
        # Cast to model dtype for in-place copy into fp16/bf16 model params
        A_out = A_new.to(model_dtype)
        B_out = B_new.to(model_dtype)
        # DeepSpeed in-model naming: module + .lora_{A,B}.default.weight
        state_dict[f"{module}.lora_A.default.weight"] = A_out
        state_dict[f"{module}.lora_B.default.weight"] = B_out
        k_per_module.append(k)
        energy_per_module.append(energy)

    meta = {
        "n_population": len(adapters),
        "n_skipped": n_skipped,
        "n_modules": len(W_meta_map),
        "k_mean": float(np.mean(k_per_module)) if k_per_module else 0.0,
        "energy_retained_mean": float(np.mean(energy_per_module)) if energy_per_module else 0.0,
        "energy_retained_min": float(np.min(energy_per_module)) if energy_per_module else 0.0,
        "energy_retained_max": float(np.max(energy_per_module)) if energy_per_module else 0.0,
        "energy_retained_std": float(np.std(energy_per_module)) if energy_per_module else 0.0,
    }
    logger.info(
        f"svd_warm_start: applied (n_population={meta['n_population']}, n_skipped={meta['n_skipped']}, "
        f"n_modules={meta['n_modules']}, k_mean={meta['k_mean']:.1f}, "
        f"energy_retained mean={meta['energy_retained_mean']:.4f} "
        f"min={meta['energy_retained_min']:.4f} max={meta['energy_retained_max']:.4f} "
        f"std={meta['energy_retained_std']:.4f})"
    )
    return state_dict, meta

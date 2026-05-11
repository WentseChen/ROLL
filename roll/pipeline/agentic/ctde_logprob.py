"""Centralized critic utilities for CTDE-GRPO.

Builds a "privileged" DataProto where the original prompt is extended with global
state (opponent card + reasoning) before the response tokens, then aligns the
resulting log-probs back to the original sequence coordinate system.
"""
import re
from typing import Dict, Optional

import numpy as np
import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import CTDEConfig
from roll.utils.logging import get_logger

logger = get_logger()

CARD_NAMES = {0: "Jack", 1: "Queen", 2: "King"}


def format_global_context(global_state: dict, ctde_config: CTDEConfig) -> str:
    """Build the global-context string appended to the user turn for the teacher pass.

    Produces a natural paragraph so the privileged information reads coherently
    to the model rather than looking like a debug dump.
    """
    if not global_state:
        return ""

    card_part = ""
    reasoning_part = ""

    if ctde_config.use_opponent_card:
        opp_card_val = global_state.get("opp_card")
        if opp_card_val is not None:
            card_name = CARD_NAMES.get(int(opp_card_val), str(opp_card_val))
            card_part = f"your opponent is holding a {card_name}"

    if ctde_config.use_opponent_reasoning:
        opp_reasoning = global_state.get("opp_reasoning", "").strip()
        # Strip any residual <think>...</think> tags so the text reads naturally
        opp_reasoning = re.sub(r"</?think>", "", opp_reasoning).strip()
        if opp_reasoning:
            reasoning_part = f'Before acting, their private reasoning was: "{opp_reasoning}"'

    if not card_part and not reasoning_part:
        return ""

    # Combine into a single natural paragraph
    if card_part and reasoning_part:
        body = f"For this hand, {card_part}. {reasoning_part}"
    elif card_part:
        body = f"For this hand, {card_part}."
    else:
        body = reasoning_part

    return f"\n\n{body}\nKnowing this, what is your reasoning and action?"


def build_privileged_batch(
    batch: DataProto,
    global_states: np.ndarray,
    tokenizer: PreTrainedTokenizer,
    ctde_config: CTDEConfig,
    max_seq_len: int,
) -> DataProto:
    """Build a DataProto with global context tokens inserted after the first prompt.

    Token layout for each sample:
        [first_prompt_tokens | global_ctx_tokens | rest_of_sequence_tokens | PAD...]

    The response_mask is updated to reflect the shifted positions.
    Returns a new DataProto suitable for compute_log_probs().
    """
    input_ids = batch.batch["input_ids"]    # [B, seq_len]
    response_mask = batch.batch["response_mask"]  # [B, seq_len] bool/int
    attention_mask = batch.batch["attention_mask"]  # [B, seq_len]
    B, seq_len = input_ids.shape
    device = input_ids.device

    new_input_ids_list = []
    new_response_mask_list = []
    new_attention_mask_list = []

    for i in range(B):
        resp_mask_i = response_mask[i].bool()
        attn_i = attention_mask[i]
        non_pad_len = int(attn_i.sum().item())

        resp_positions = resp_mask_i.nonzero(as_tuple=True)[0]
        if len(resp_positions) == 0:
            # No response — pass through unchanged (padded to max_seq_len)
            new_input_ids_list.append(_pad_or_trunc(input_ids[i], max_seq_len, tokenizer.pad_token_id))
            new_response_mask_list.append(_pad_or_trunc(resp_mask_i.long(), max_seq_len, 0))
            new_attention_mask_list.append(_pad_or_trunc(attn_i, max_seq_len, 0))
            continue

        first_resp_idx = int(resp_positions[0].item())

        # Build global context tokens
        gs = global_states[i] if global_states is not None else {}
        ctx_text = format_global_context(gs, ctde_config) if gs else ""
        if ctx_text:
            ctx_ids = tokenizer.encode(ctx_text, add_special_tokens=False)
        else:
            ctx_ids = []
        K = len(ctx_ids)

        # Insert global context inside the user turn, right before the last <|im_end|>
        # in the prompt portion.  This produces a valid chat structure:
        #   <|im_start|>user\n{obs}\n\n[Global context...]\n<|im_end|>
        #   <|im_start|>assistant\n{original response}
        # Inserting at first_resp_idx (after <|im_start|>assistant\n) would place the
        # ctx inside the assistant turn, making the structure incoherent.
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        prompt_tokens = input_ids[i, :first_resp_idx]
        im_end_pos_in_prompt = (prompt_tokens == im_end_id).nonzero(as_tuple=True)[0]
        if len(im_end_pos_in_prompt) > 0:
            insert_pos = int(im_end_pos_in_prompt[-1].item())  # before last <|im_end|>
        else:
            insert_pos = first_resp_idx  # fallback for non-standard templates

        prefix = input_ids[i, :insert_pos]
        suffix_prompt = input_ids[i, insert_pos:first_resp_idx]
        response_tokens = input_ids[i, first_resp_idx:non_pad_len]

        ctx_tensor = torch.tensor(ctx_ids, dtype=torch.long, device=device)
        new_tokens = torch.cat([prefix, ctx_tensor, suffix_prompt, response_tokens], dim=0)
        new_non_pad_len = new_tokens.shape[0]

        # Truncate if exceeds max_seq_len
        if new_non_pad_len > max_seq_len:
            new_tokens = new_tokens[:max_seq_len]
            new_non_pad_len = max_seq_len

        # Shift response positions by K
        new_resp_mask = torch.zeros(max_seq_len, dtype=torch.long, device=device)
        shifted_positions = resp_positions + K
        valid = shifted_positions[shifted_positions < max_seq_len]
        new_resp_mask[valid] = 1

        # Attention mask: 1 for all non-pad tokens
        new_attn = torch.zeros(max_seq_len, dtype=torch.long, device=device)
        new_attn[:new_non_pad_len] = 1

        # Pad tokens to max_seq_len
        if new_non_pad_len < max_seq_len:
            pad = torch.full((max_seq_len - new_non_pad_len,), tokenizer.pad_token_id, dtype=torch.long, device=device)
            new_tokens = torch.cat([new_tokens, pad], dim=0)

        new_input_ids_list.append(new_tokens)
        new_response_mask_list.append(new_resp_mask)
        new_attention_mask_list.append(new_attn)

    new_input_ids = torch.stack(new_input_ids_list, dim=0)       # [B, max_seq_len]
    new_resp_mask_t = torch.stack(new_response_mask_list, dim=0)  # [B, max_seq_len]
    new_attn_t = torch.stack(new_attention_mask_list, dim=0)      # [B, max_seq_len]
    position_ids = new_attn_t.cumsum(dim=-1) - 1

    priv_batch = DataProto()
    priv_batch.batch = TensorDict(
        {
            "input_ids": new_input_ids,
            "attention_mask": new_attn_t,
            "position_ids": position_ids,
            "response_mask": new_resp_mask_t,
        },
        batch_size=[B],
    )
    return priv_batch


def align_teacher_logprobs(
    priv_log_probs: torch.Tensor,
    priv_response_mask: torch.Tensor,
    orig_response_mask: torch.Tensor,
) -> torch.Tensor:
    """Re-align teacher log-probs from the privileged sequence to the original coordinate system.

    Both priv_log_probs and the output use the [:, 1:] (next-token) convention:
        log_probs[i, t] = log P(token[t+1] | context)
        response_mask_shifted[t] = 1 iff token[t+1] is a response token

    Args:
        priv_log_probs:    [B, priv_len-1]
        priv_response_mask:[B, priv_len] (unshifted; 1 at response token positions)
        orig_response_mask:[B, orig_len] (unshifted; 1 at response token positions)

    Returns:
        teacher_lp: [B, orig_len-1] with teacher log-probs at response positions, 0 elsewhere
    """
    B = priv_log_probs.shape[0]
    orig_len = orig_response_mask.shape[1]
    teacher_lp = torch.zeros(B, orig_len - 1, dtype=priv_log_probs.dtype, device=priv_log_probs.device)

    for i in range(B):
        # Shifted masks: index t == 1 iff token[t+1] is a response token
        priv_shifted = priv_response_mask[i, 1:].bool()   # [priv_len-1]
        orig_shifted = orig_response_mask[i, 1:].bool()   # [orig_len-1]

        resp_lp = priv_log_probs[i][priv_shifted]          # [M]
        orig_positions = orig_shifted.nonzero(as_tuple=True)[0]  # [M']

        n = min(resp_lp.shape[0], orig_positions.shape[0])
        if n > 0:
            teacher_lp[i, orig_positions[:n]] = resp_lp[:n].float()

    return teacher_lp  # [B, orig_len-1]


def compute_ctde_token_bonus(
    teacher_log_probs: torch.Tensor,
    infer_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    bonus_weight: float,
) -> torch.Tensor:
    """Compute per-token CTDE bonus from teacher/student log-ratio, length-normalized.

    Per-token bonus is bonus_weight * (log π_teacher - log π_student) / n_response_tokens.
    Total per-trajectory contribution is bonus_weight * mean(log_ratio), independent of
    length — kills the length-hacking incentive while preserving per-token gradient direction.

    Args:
        teacher_log_probs: [B, seq_len-1]
        infer_log_probs:   [B, seq_len-1]
        response_mask:     [B, seq_len] (unshifted)
        bonus_weight:      scalar weight

    Returns:
        per_token_bonus:  [B, seq_len-1]
        log_ratio_mean:   [B] mean log-ratio per trajectory (for logging)
    """
    resp_shifted = response_mask[:, 1:].float()                        # [B, seq_len-1]
    log_ratio = (teacher_log_probs - infer_log_probs) * resp_shifted   # [B, seq_len-1]

    n_tokens = resp_shifted.sum(dim=-1, keepdim=True).clamp(min=1.0)   # [B, 1]
    per_token_bonus = bonus_weight * log_ratio / n_tokens              # [B, seq_len-1]

    log_ratio_mean = log_ratio.sum(dim=-1) / n_tokens.squeeze(-1)      # [B] for logging
    return per_token_bonus, log_ratio_mean


def _pad_or_trunc(t: torch.Tensor, length: int, pad_val: int) -> torch.Tensor:
    if t.shape[0] >= length:
        return t[:length]
    pad = torch.full((length - t.shape[0],), pad_val, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=0)

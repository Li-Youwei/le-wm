"""FAST token decode: token IDs → continuous action chunks.

Completes the inference pipeline:
    generate() → clean FAST tokens → fast_decode() → (H, action_dim) continuous actions

Usage:
    from fast_utils import load_fast_processor, fast_decode

    processor = load_fast_processor("data/fast_tokenizer")
    actions = fast_decode(token_ids, lengths, processor, time_horizon=10, action_dim=7)
    # actions: (B, H, action_dim) in [-1, 1] (normalized)
    # To get raw actions: actions * (q99 - q01) / 2 + (q99 + q01) / 2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor


def load_fast_processor(tokenizer_path: str | Path) -> AutoProcessor:
    """Load a trained FAST tokenizer/processor.

    Args:
        tokenizer_path: directory containing the saved FAST processor
            (output of preprocess_libero.py --save-tokenizer).

    Returns:
        FAST processor with encode/decode capability.
    """
    return AutoProcessor.from_pretrained(
        str(tokenizer_path), trust_remote_code=True,
    )


def fast_decode(
    token_ids: torch.Tensor,
    lengths: torch.Tensor,
    processor: AutoProcessor,
    time_horizon: int = 10,
    action_dim: int = 7,
) -> np.ndarray:
    """Decode FAST token IDs back to continuous action chunks.

    Pipeline: token IDs → BPE decode → inverse rounding → inverse DCT → actions.

    Args:
        token_ids: (B, max_len) FAST token IDs from generate() (PAD-stripped).
        lengths: (B,) real token count per sample.
        processor: FAST processor from load_fast_processor().
        time_horizon: action chunk length in raw steps (H=10 for LIBERO).
        action_dim: action dimensionality (7 for LIBERO: 6D EE + gripper).

    Returns:
        actions: (B, time_horizon, action_dim) continuous actions in normalized
            space (approx [-1, 1]). Apply inverse normalization using the
            action_low/action_high stats from preprocessing to get raw actions.
    """
    B = token_ids.size(0)
    token_lists = []
    for i in range(B):
        k = lengths[i].item()
        tokens_i = token_ids[i, :k].cpu().tolist()
        token_lists.append(tokens_i)

    return processor.decode(
        token_lists,
        time_horizon=time_horizon,
        action_dim=action_dim,
    )


def denormalize_actions(
    actions: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """Undo the [-1, 1] normalization applied during preprocessing.

    Args:
        actions: (B, H, D) normalized actions from fast_decode().
        action_low: (D,) per-dimension 1st percentile from preprocessing.
        action_high: (D,) per-dimension 99th percentile from preprocessing.

    Returns:
        (B, H, D) raw continuous actions in original scale.
    """
    mid = (action_high + action_low) / 2.0
    half_range = (action_high - action_low) / 2.0
    # Avoid division by zero for constant dimensions
    half_range = np.where(half_range < 1e-8, 1.0, half_range)
    return actions * half_range + mid

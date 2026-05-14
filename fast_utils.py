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

import logging
from pathlib import Path

import numpy as np
import torch
from scipy.fft import idct
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

# Rate-limit decode warnings: track count to avoid spamming logs
_decode_warn_count = 0
_DECODE_WARN_LIMIT = 20


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

    Pipeline: token IDs → BPE decode → ord() → pad/truncate to H*D → reshape
    → inverse quantize → inverse DCT → continuous actions.

    Handles the common case where BPE decode produces slightly more or fewer
    values than time_horizon * action_dim by padding with zeros or truncating.
    The processor's built-in decode falls back to all-zeros on any mismatch,
    which makes the robot freeze. This function instead preserves as much of
    the decoded signal as possible.

    Args:
        token_ids: (B, max_len) FAST token IDs from generate().
        lengths: (B,) real token count per sample.
        processor: FAST processor from load_fast_processor().
        time_horizon: action chunk length in raw steps (H=20 for LIBERO at 20Hz).
        action_dim: action dimensionality (7 for LIBERO: 6D EE + gripper).

    Returns:
        actions: (B, time_horizon, action_dim) continuous actions in normalized
            space (approx [-1, 1]).
    """
    global _decode_warn_count

    B = token_ids.size(0)
    expected_len = time_horizon * action_dim
    scale = getattr(processor, "scale", 10)
    min_token = getattr(processor, "min_token", 0)
    bpe_tokenizer = getattr(processor, "bpe_tokenizer", None)

    decoded_actions = []
    for i in range(B):
        k = lengths[i].item()
        tokens_i = token_ids[i, :k].cpu().tolist()

        try:
            # BPE decode → string of chars → ord() → quantized DCT coefficients
            decoded_str = bpe_tokenizer.decode(tokens_i)
            raw_values = np.array(list(map(ord, decoded_str))) + min_token
            actual_len = len(raw_values)

            # Pad or truncate to exactly H * D
            if actual_len != expected_len:
                if _decode_warn_count < _DECODE_WARN_LIMIT:
                    logger.warning(
                        "FAST decode: got %d values, expected %d (H=%d, D=%d). %s.",
                        actual_len, expected_len, time_horizon, action_dim,
                        "Truncating" if actual_len > expected_len else "Padding with zeros",
                    )
                    _decode_warn_count += 1
                    if _decode_warn_count == _DECODE_WARN_LIMIT:
                        logger.warning("FAST decode: suppressing further warnings.")

                if actual_len > expected_len:
                    raw_values = raw_values[:expected_len]
                else:
                    raw_values = np.pad(raw_values, (0, expected_len - actual_len))

            dct_coeff = raw_values.reshape(time_horizon, action_dim).astype(np.float64)
            actions_i = idct(dct_coeff / scale, axis=0, norm="ortho")

        except Exception:
            if _decode_warn_count < _DECODE_WARN_LIMIT:
                logger.warning("FAST decode: failed for sample %d, using zeros.", i)
                _decode_warn_count += 1
            actions_i = np.zeros((time_horizon, action_dim))

        decoded_actions.append(actions_i)

    return np.stack(decoded_actions)


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
    zero_range = half_range < 1e-8
    restored = actions * half_range + mid
    return np.where(zero_range, mid, restored)

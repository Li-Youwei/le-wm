"""fit_tokenizer_all4.py — Unified FAST tokenizer over all 4 LIBERO suites + per-task length audit.

Option B (per user decision): each task's action chunks are normalized to [-1, 1] using THAT
task's own (action_low, action_high) percentile bounds. The 40 per-task-normalized chunk
arrays are then concatenated and used to fit a single shared BPE vocabulary. Downstream
preprocessing (preprocess_libero.py --load-tokenizer ...) reuses this fitted tokenizer
while each per-task .h5 keeps its own (low, high) attrs.

Output:
- Fitted tokenizer at $TOKENIZER_DIR/ (default: /Data/lyw/fast_tokenizer_all4/)
- Audit JSON at $AUDIT_OUT (default: /Data/lyw/libero_processed_v5/audit_token_length.json)
- Stdout: per-task token-length stats + recommended max_action_tokens

The raw LIBERO HDF5 directory ($RAW_LIBERO_DIR, default /nas_data_new/caz/data_ssd/libero)
is opened READ-ONLY. We never write into it.

Usage (on the GPU server):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
        python fit_tokenizer_all4.py \\
            --raw-root /nas_data_new/caz/data_ssd/libero \\
            --tokenizer-out /Data/lyw/fast_tokenizer_all4 \\
            --audit-out /Data/lyw/libero_processed_v5/audit_token_length.json \\
            --chunk-size 20 --stride 1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

from preprocess_libero import (
    compute_action_stats,
    load_demo_keys,
    normalize_actions,
    tokenize_actions,
)

SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

logger = logging.getLogger("fit_tokenizer_all4")


# ---------------------------------------------------------------------------
# Slim action-chunk extractor (no images, no proprio)
# ---------------------------------------------------------------------------


def extract_action_chunks_only(
    f: h5py.File, demo_keys: list[str], chunk_size: int, stride: int
) -> np.ndarray:
    """Vectorized sliding-window extraction of anchor-relative (H, 7) action chunks.

    Mirrors preprocess_libero.extract_chunks but omits image / proprio outputs to
    keep memory bounded across 40 tasks. Returns (N_total, H, 7) physical-unit
    chunks (NOT normalized).
    """
    H = chunk_size
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    all_chunks: list[np.ndarray] = []

    for demo_i, dk in enumerate(demo_keys):
        actions_raw = f[f"data/{dk}/actions"][()]
        T = actions_raw.shape[0]
        if T < H + 1:
            continue
        n_chunks = (T - H - 1) // stride + 1
        last_obs_idx = (n_chunks - 1) * stride + H

        ee_pos = f[f"data/{dk}/obs/ee_pos"][: last_obs_idx + 1]
        ee_ori = f[f"data/{dk}/obs/ee_ori"][: last_obs_idx + 1]
        gripper_cmd = actions_raw[:last_obs_idx, 6]

        Rs = R.from_rotvec(ee_ori)
        starts_arr = np.arange(n_chunks) * stride
        k_p1 = np.arange(1, H + 1)
        target_idx = starts_arr[:, None] + k_p1[None, :]
        target_flat = target_idx.reshape(-1)
        anchor_flat = np.repeat(starts_arr, H)

        pos_deltas = (ee_pos[target_flat] - ee_pos[anchor_flat]).reshape(n_chunks, H, 3)
        R_target = Rs[target_flat]
        R_anchor = Rs[anchor_flat]
        rot_deltas = (R_target * R_anchor.inv()).as_rotvec().reshape(n_chunks, H, 3)
        gripper_idx = starts_arr[:, None] + np.arange(H)[None, :]
        gripper_values = gripper_cmd[gripper_idx.reshape(-1)].reshape(n_chunks, H, 1)

        chunks = np.concatenate(
            [pos_deltas, rot_deltas, gripper_values], axis=-1
        ).astype(np.float32)
        all_chunks.append(chunks)

    if not all_chunks:
        return np.zeros((0, H, 7), dtype=np.float32)
    return np.concatenate(all_chunks, axis=0)


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------


def token_length_stats(lengths: list[int]) -> dict[str, float | int]:
    """Compute per-task summary stats on FAST token lengths."""
    arr = np.asarray(lengths, dtype=np.int64)
    return {
        "n": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "std": float(arr.std()),
    }


def recommend_max_action_tokens(
    per_task_p99: list[float], current_setting: int = 80
) -> int:
    """Pick the smallest 16-multiple >= max p99 across tasks; floor at current_setting."""
    max_p99 = max(per_task_p99) if per_task_p99 else 0.0
    if max_p99 <= current_setting:
        return current_setting
    bump = int(np.ceil(max_p99 / 16.0) * 16)
    return max(bump, current_setting)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/nas_data_new/caz/data_ssd/libero"),
        help="READ-ONLY raw LIBERO root containing libero_{spatial,object,goal,10}/",
    )
    parser.add_argument(
        "--tokenizer-out",
        type=Path,
        default=Path("/Data/lyw/fast_tokenizer_all4"),
        help="Output dir for the fitted FAST tokenizer",
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=Path("/Data/lyw/libero_processed_v5/audit_token_length.json"),
        help="Output JSON for per-task token length audit",
    )
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--current-max-action-tokens",
        type=int,
        default=80,
        help="Current max_action_tokens (used as floor when recommending a bump)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.raw_root.is_dir():
        raise FileNotFoundError(f"raw_root not found or not a dir: {args.raw_root}")

    # -----------------------------------------------------------------------
    # Pass 1: per-task chunk extraction + per-task normalization (Option B)
    # -----------------------------------------------------------------------
    per_task_normalized: dict[str, np.ndarray] = {}  # task_key → (N, H, 7) in [-1, 1]
    per_task_low: dict[str, np.ndarray] = {}
    per_task_high: dict[str, np.ndarray] = {}

    for suite in SUITES:
        suite_dir = args.raw_root / suite
        if not suite_dir.is_dir():
            raise FileNotFoundError(f"Missing suite dir: {suite_dir}")
        hdf5_files = sorted(suite_dir.glob("*.hdf5"))
        if not hdf5_files:
            raise FileNotFoundError(f"No *.hdf5 under {suite_dir}")
        if len(hdf5_files) != 10:
            logger.warning(
                "Expected 10 hdf5 files in %s, found %d", suite, len(hdf5_files)
            )

        for h5_path in hdf5_files:
            task_key = f"{suite}::{h5_path.stem}"
            logger.info("=== %s (%s) ===", task_key, h5_path)

            with h5py.File(h5_path, "r") as f:  # READ-ONLY
                demo_keys = load_demo_keys(f)
                physical = extract_action_chunks_only(
                    f, demo_keys, args.chunk_size, args.stride
                )

            if physical.shape[0] == 0:
                logger.warning("  No chunks extracted for %s — skipping", task_key)
                continue

            low, high = compute_action_stats(
                [physical[i] for i in range(physical.shape[0])]
            )
            normed = normalize_actions(physical, low, high)
            per_task_normalized[task_key] = normed.astype(np.float32)
            per_task_low[task_key] = low
            per_task_high[task_key] = high
            logger.info(
                "  %d chunks, normalized to [-1, 1] using per-task percentile bounds",
                normed.shape[0],
            )

    if not per_task_normalized:
        raise RuntimeError("No task chunks collected — abort.")

    # -----------------------------------------------------------------------
    # Pass 2: concat → fit shared FAST tokenizer
    # -----------------------------------------------------------------------
    merged = np.concatenate(list(per_task_normalized.values()), axis=0)
    logger.info(
        "Merged action chunks for tokenizer fit: shape=%s (40 tasks combined)",
        merged.shape,
    )

    args.tokenizer_out.mkdir(parents=True, exist_ok=True)
    _, tokenizer = tokenize_actions(
        merged,
        fit=True,
        save_tokenizer_path=str(args.tokenizer_out),
        load_tokenizer_path=None,
    )
    logger.info("Saved unified tokenizer to %s", args.tokenizer_out)

    # -----------------------------------------------------------------------
    # Pass 3: per-task audit — re-encode each task with the fitted tokenizer
    # -----------------------------------------------------------------------
    per_task_stats: dict[str, dict[str, Any]] = {}
    per_task_p99: list[float] = []

    for task_key, normed in per_task_normalized.items():
        tokens_raw = tokenizer(normed)
        if isinstance(tokens_raw, dict) or hasattr(tokens_raw, "input_ids"):
            tokens_raw = tokens_raw["input_ids"]
        lengths = [len(t) for t in tokens_raw]
        stats = token_length_stats(lengths)
        per_task_stats[task_key] = stats
        per_task_p99.append(stats["p99"])
        logger.info(
            "  %s: n=%d min=%d mean=%.1f median=%.0f p95=%.1f p99=%.1f max=%d",
            task_key,
            stats["n"],
            stats["min"],
            stats["mean"],
            stats["median"],
            stats["p95"],
            stats["p99"],
            stats["max"],
        )

    recommended = recommend_max_action_tokens(
        per_task_p99, current_setting=args.current_max_action_tokens
    )
    overall_max = max(s["max"] for s in per_task_stats.values())
    overall_p99 = float(np.max(per_task_p99))

    logger.info("---")
    logger.info("Overall max token length: %d", overall_max)
    logger.info("Worst per-task p99 across 40 tasks: %.2f", overall_p99)
    logger.info(
        "Recommended max_action_tokens: %d (current=%d)",
        recommended,
        args.current_max_action_tokens,
    )
    if recommended > args.current_max_action_tokens:
        logger.warning(
            "max_action_tokens MUST be bumped to %d in config/train/data/libero.yaml AND "
            "preprocess_all4.sh — otherwise preprocess_libero.py will assert-fail.",
            recommended,
        )

    # -----------------------------------------------------------------------
    # Pass 4: write audit JSON
    # -----------------------------------------------------------------------
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "tokenizer_dir": str(args.tokenizer_out),
        "chunk_size": args.chunk_size,
        "stride": args.stride,
        "n_tasks": len(per_task_stats),
        "overall_max_token_length": overall_max,
        "overall_p99": overall_p99,
        "current_max_action_tokens": args.current_max_action_tokens,
        "recommended_max_action_tokens": recommended,
        "per_task": per_task_stats,
        "per_task_action_low": {
            k: per_task_low[k].tolist() for k in per_task_normalized
        },
        "per_task_action_high": {
            k: per_task_high[k].tolist() for k in per_task_normalized
        },
    }
    args.audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote audit JSON to %s", args.audit_out)


if __name__ == "__main__":
    main()

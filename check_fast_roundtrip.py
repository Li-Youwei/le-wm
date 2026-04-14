#!/usr/bin/env python3
"""check_fast_roundtrip.py — Sanity-check the FAST encode/decode round-trip.

Validates the full action codec pipeline on preprocessed LIBERO samples:

    raw actions (H, 7)                     # from original LIBERO HDF5
      ── normalize to [-1, 1] ──────►      # action_low / action_high
      ── FAST encode ──────────────►       # stored as fast_tokens
      ── FAST decode ──────────────►       # fast_utils.fast_decode
      ── denormalize ──────────────►
      reconstructed raw actions (H, 7)

We compare the reconstructed chunk against the raw ground-truth actions from
the *original* LIBERO HDF5 (via the source_file attr) rather than against the
already-normalized-and-clipped `continuous_actions` stored alongside the FAST
tokens. That way any error floor from 1st/99th-percentile clipping is visible
rather than hidden.

To isolate the two error sources, we also report:
  • The pure FAST round-trip error in normalized space
    (decoded_norm vs stored_norm — no clipping involved).
  • The clipping gap (stored_norm vs re-normalized raw GT) — should be ~0
    unless actions got clipped.
  • The raw-space error after denormalization (what the robot actually sees).

Usage:
    python check_fast_roundtrip.py \\
        --processed-dir /Data/lyw/libero_processed/libero_spatial/ \\
        --tokenizer /Data/lyw/fast_tokenizer \\
        [--num-samples 10] [--print-samples 3] [--seed 0] \\
        [--libero-root /path/to/raw/libero_spatial]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import torch

from fast_utils import denormalize_actions, fast_decode, load_fast_processor
from preprocess_libero import normalize_actions


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_processed_files(processed_dir: Path) -> list[Path]:
    """Find every .hdf5 / .h5 file under `processed_dir` (recursive)."""
    files = sorted(processed_dir.rglob("*.hdf5"))
    if not files:
        files = sorted(processed_dir.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5/.h5 files found under {processed_dir}")
    return files


def build_sample_index(files: list[Path]) -> list[tuple[Path, int]]:
    """Build a flat `(file_path, local_idx)` index across all processed files."""
    index: list[tuple[Path, int]] = []
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            if "fast_tokens" not in f:
                print(f"  [skip] {fpath.name}: no fast_tokens dataset")
                continue
            n = f["fast_tokens"].shape[0]
            index.extend((fpath, i) for i in range(n))
    return index


def resolve_source_file(stored_path: str, libero_root: Path | None) -> Path:
    """Resolve the original LIBERO HDF5 referenced by `source_file` attr.

    Preprocessing stores an absolute path; if the file has moved, fall back to
    `libero_root/<basename>` when provided.
    """
    src = Path(stored_path)
    if src.exists():
        return src
    if libero_root is not None:
        candidate = libero_root / src.name
        if candidate.exists():
            return candidate
    tried = [str(src)]
    if libero_root is not None:
        tried.append(str(libero_root / src.name))
    raise FileNotFoundError(
        "Original LIBERO HDF5 not found. Tried: "
        + ", ".join(tried)
        + ". Pass --libero-root to override."
    )


def load_raw_demo_actions(source_file: Path, demo_idx: int) -> np.ndarray:
    """Load raw (un-normalized) actions for a single demo from the original HDF5."""
    with h5py.File(source_file, "r") as f:
        demo_keys = sorted(
            (k for k in f["data"].keys() if k.startswith("demo_")),
            key=lambda k: int(k.split("_")[1]),
        )
        if not demo_keys:
            raise RuntimeError(f"No demo_* groups under data/ in {source_file}")
        if demo_idx >= len(demo_keys):
            raise IndexError(
                f"demo_idx={demo_idx} out of range (file has {len(demo_keys)} demos)"
            )
        dk = demo_keys[demo_idx]
        return np.asarray(f[f"data/{dk}/actions"][()], dtype=np.float64)  # (T, 7)


# ---------------------------------------------------------------------------
# Per-sample round-trip
# ---------------------------------------------------------------------------

def decode_one_sample(
    tokens_np: np.ndarray,
    token_len: int,
    processor,
    chunk_size: int,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """FAST tokens -> (normalized, raw) action chunks.

    Returns:
        decoded_norm: (H, action_dim) in approximately [-1, 1]
        decoded_raw : (H, action_dim) in original action units
    """
    token_tensor = torch.from_numpy(tokens_np.astype(np.int64)).unsqueeze(0)  # (1, k)
    length_tensor = torch.tensor([token_len], dtype=torch.long)
    decoded_norm_batch = fast_decode(
        token_tensor,
        length_tensor,
        processor,
        time_horizon=chunk_size,
        action_dim=action_dim,
    )  # (1, H, 7)
    decoded_raw_batch = denormalize_actions(
        decoded_norm_batch, action_low, action_high
    )
    return decoded_norm_batch[0], decoded_raw_batch[0]


def format_row(values: np.ndarray, width: int = 8, prec: int = 3) -> str:
    """Right-aligned fixed-width numeric row for tabular printing."""
    return " ".join(f"{v:>+{width}.{prec}f}" for v in values)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FAST round-trip sanity check on preprocessed LIBERO data."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("/Data/lyw/libero_processed/libero_spatial/"),
        help="Directory containing preprocessed HDF5 files (searched recursively).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("/Data/lyw/fast_tokenizer"),
        help="Directory containing the saved FAST tokenizer.",
    )
    parser.add_argument(
        "--libero-root",
        type=Path,
        default=None,
        help="Root dir containing the original LIBERO task HDF5 files — used "
             "only when the `source_file` attr no longer resolves.",
    )
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of random samples to check (default 10).")
    parser.add_argument("--print-samples", type=int, default=3,
                        help="Samples to print per-step per-dim detail for (default 3).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducible sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---- Step 1: Discover and index all preprocessed samples ----
    print(f"[Step 1] Scanning {args.processed_dir}")
    files = discover_processed_files(args.processed_dir)
    print(f"  Found {len(files)} processed HDF5 file(s)")
    index = build_sample_index(files)
    print(f"  Total samples indexed: {len(index)}")
    if len(index) < args.num_samples:
        raise RuntimeError(
            f"Only {len(index)} samples available, need at least {args.num_samples}."
        )

    picks = random.sample(range(len(index)), args.num_samples)

    # ---- Step 2: Load the FAST tokenizer ----
    print(f"\n[Step 2] Loading FAST tokenizer from {args.tokenizer}")
    processor = load_fast_processor(args.tokenizer)

    # ---- Step 3: Run round-trip on each sample ----
    per_dim_l1_raw: list[np.ndarray] = []        # (action_dim,)
    per_step_l1_raw: list[np.ndarray] = []       # (chunk_size,)
    per_sample_overall_l1: list[float] = []
    per_sample_max_err: list[float] = []
    per_sample_norm_l1: list[float] = []
    per_sample_clip_gap: list[float] = []

    for rank, pick in enumerate(picks):
        fpath, local_idx = index[pick]

        with h5py.File(fpath, "r") as f:
            chunk_size = int(f.attrs["chunk_size"])
            action_low = np.asarray(f.attrs["action_low"], dtype=np.float64)
            action_high = np.asarray(f.attrs["action_high"], dtype=np.float64)
            action_dim = int(f.attrs.get("action_dim", 7))
            source_file = str(f.attrs["source_file"])

            demo_idx = int(f["demo_idx"][local_idx])
            chunk_idx = int(f["chunk_idx"][local_idx])

            tokens_np = np.asarray(f["fast_tokens"][local_idx], dtype=np.int64)
            token_len = int(f["fast_length"][local_idx])
            if token_len != len(tokens_np):
                # vlen stores the real length; `fast_length` is a redundant scalar.
                print(f"  [warn] fast_length={token_len} disagrees with "
                      f"len(fast_tokens)={len(tokens_np)}; using token array length.")
                token_len = len(tokens_np)

            stored_norm = np.asarray(
                f["continuous_actions"][local_idx], dtype=np.float64
            )  # (H, 7) — already normalized + clipped

        # Original raw ground truth from the source LIBERO HDF5
        src_path = resolve_source_file(source_file, args.libero_root)
        raw_demo_actions = load_raw_demo_actions(src_path, demo_idx)
        start = chunk_idx * chunk_size
        gt_chunk_raw = raw_demo_actions[start : start + chunk_size]  # (H, 7)
        if gt_chunk_raw.shape[0] != chunk_size:
            raise RuntimeError(
                f"GT chunk slice is {gt_chunk_raw.shape}, expected "
                f"({chunk_size}, {action_dim}). demo={demo_idx} chunk={chunk_idx} "
                f"src={src_path}"
            )

        # FAST decode
        decoded_norm, decoded_raw = decode_one_sample(
            tokens_np, token_len, processor,
            chunk_size, action_dim, action_low, action_high,
        )

        # ---- Error decomposition ----
        # 1) Pure FAST codec error (normalized space): decoded_norm vs stored_norm
        norm_err = np.abs(decoded_norm - stored_norm)

        # 2) Clipping gap: stored_norm vs re-normalize(gt_raw)
        renorm_gt = normalize_actions(gt_chunk_raw, action_low, action_high)
        clip_gap = np.abs(renorm_gt - stored_norm)

        # 3) Raw-space total error: decoded_raw vs gt_chunk_raw
        raw_err = np.abs(decoded_raw - gt_chunk_raw)

        per_dim = raw_err.mean(axis=0)               # (action_dim,)
        per_step = raw_err.mean(axis=1)              # (chunk_size,)
        overall = float(raw_err.mean())
        max_err = float(raw_err.max())

        per_dim_l1_raw.append(per_dim)
        per_step_l1_raw.append(per_step)
        per_sample_overall_l1.append(overall)
        per_sample_max_err.append(max_err)
        per_sample_norm_l1.append(float(norm_err.mean()))
        per_sample_clip_gap.append(float(clip_gap.max()))

        # ---- Per-sample header ----
        print()
        print("=" * 80)
        print(f"[Sample {rank + 1}/{args.num_samples}]  {fpath.name}")
        print(f"  global_idx={pick}  local_idx={local_idx}  "
              f"demo={demo_idx}  chunk={chunk_idx}  token_len={token_len}")
        print(f"  chunk_size H={chunk_size}  action_dim={action_dim}")
        print(f"  source_file={src_path.name}")
        print(f"  raw-space     : mean |err|={overall:.5f}  max |err|={max_err:.5f}")
        print(f"  normalized    : mean |err|={norm_err.mean():.2e}  "
              f"max |err|={norm_err.max():.2e}   (pure FAST codec)")
        print(f"  clipping gap  : max |err|={clip_gap.max():.2e}   "
              f"(stored_norm vs renorm(gt); should be ~0)")
        print(f"  per-dim L1  (raw): {format_row(per_dim, width=7, prec=4)}")
        print(f"  per-step L1 (raw): {format_row(per_step, width=7, prec=4)}")

        # ---- Detailed per-step per-dim table for the first N samples ----
        if rank < args.print_samples:
            print()
            print("  Per-step per-dim detail (raw scale):")
            header_dims = "  ".join(f"{'d' + str(d):>8}" for d in range(action_dim))
            print(f"    step      {header_dims}")
            print(f"    " + "-" * (11 + len(header_dims)))
            for h in range(chunk_size):
                print(f"    {h:>3d}  P | {format_row(decoded_raw[h])}")
                print(f"        G | {format_row(gt_chunk_raw[h])}")
                print(f"        E | "
                      + " ".join(f"{v:>+8.3f}" for v in raw_err[h]))

    # ---- Step 4: Aggregate summary ----
    per_dim_arr = np.stack(per_dim_l1_raw, axis=0)     # (N, action_dim)
    per_step_arr = np.stack(per_step_l1_raw, axis=0)   # (N, chunk_size)

    print()
    print("=" * 80)
    print(f"[Aggregate over {args.num_samples} samples]")
    print(f"  Raw-space mean |err|: "
          f"mean={np.mean(per_sample_overall_l1):.5f}  "
          f"std={np.std(per_sample_overall_l1):.5f}  "
          f"min={np.min(per_sample_overall_l1):.5f}  "
          f"max={np.max(per_sample_overall_l1):.5f}")
    print(f"  Raw-space max |err|:  "
          f"mean={np.mean(per_sample_max_err):.5f}  "
          f"worst={np.max(per_sample_max_err):.5f}")
    print(f"  Normalized mean |err| (pure FAST codec): "
          f"mean={np.mean(per_sample_norm_l1):.2e}  "
          f"max={np.max(per_sample_norm_l1):.2e}")
    print(f"  Clipping gap max |err| across samples: "
          f"{np.max(per_sample_clip_gap):.2e}")

    # Per-dimension breakdown — label assumes LIBERO 6D EE + gripper (7d)
    if per_dim_arr.shape[1] == 7:
        dim_labels = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
    else:
        dim_labels = [f"d{i}" for i in range(per_dim_arr.shape[1])]

    print("  Per-dim L1 (raw scale):")
    for d, lbl in enumerate(dim_labels):
        col = per_dim_arr[:, d]
        print(f"    {lbl:>5}:  mean={col.mean():.5f}  "
              f"max={col.max():.5f}  std={col.std():.5f}")

    print("  Per-step L1 (raw scale, averaged over samples):")
    step_mean = per_step_arr.mean(axis=0)
    for h in range(per_step_arr.shape[1]):
        print(f"    step {h:>2d}: {step_mean[h]:.5f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

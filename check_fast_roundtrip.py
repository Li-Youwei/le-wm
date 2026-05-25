#!/usr/bin/env python3
"""check_fast_roundtrip.py — Sanity-check the FAST encode/decode round-trip.

Validates the full action codec pipeline on preprocessed LIBERO samples
under the **anchor-relative chunk format** (see preprocess_libero.py):

    obs/ee_pos, obs/ee_ori (base-frame anchor state + H+1 future frames)
      ── compute anchor-relative chunk (pos delta + rotvec delta + grip cmd) ──►
      ── normalize to [-1, 1] via 1/99 percentile ──► FAST encode ──► stored fast_tokens
      ── FAST decode ──► decoded_norm ── denormalize ──► decoded_phys

We compute the **ground-truth anchor-relative chunk** directly from the raw
LIBERO HDF5 (obs/ee_pos + obs/ee_ori, same formulas as preprocess_libero.py)
and compare it to the decoded chunks in three layers:

  1. Normalized-space round-trip (decoded_norm vs stored continuous_actions)
     — pure FAST codec error. Should be a few percent (DCT+BPE is lossy).
  2. Normalized-space GT comparison (decoded_norm vs renorm(gt_phys))
     — picks up any clipping that happened at normalization time.
  3. Physical-space (decoded_phys vs gt_phys) — what the robot actually
     sees after denormalization.

This validates both the FAST codec AND the new anchor-relative preprocessing
end-to-end.

Usage:
    python check_fast_roundtrip.py \\
        --processed-dir /data/lyw/libero_processed/libero_spatial/ \\
        --tokenizer /data/lyw/fast_tokenizer \\
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
from scipy.spatial.transform import Rotation as R

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


def load_raw_demo_obs(
    source_file: Path, demo_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the raw observations and gripper command stream for one demo.

    Returns (ee_pos, ee_ori, gripper_cmd) from the original LIBERO HDF5 —
    everything we need to reconstruct the anchor-relative chunk ground truth
    using the same formulas as preprocess_libero.py::extract_chunks.
    """
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
        ee_pos = np.asarray(f[f"data/{dk}/obs/ee_pos"][()], dtype=np.float64)  # (T, 3)
        ee_ori = np.asarray(f[f"data/{dk}/obs/ee_ori"][()], dtype=np.float64)  # (T, 3) axis-angle
        actions = np.asarray(f[f"data/{dk}/actions"][()], dtype=np.float64)   # (T, 7)
        gripper_cmd = actions[:, 6]  # (T,)
    return ee_pos, ee_ori, gripper_cmd


def compute_anchor_relative_chunk_gt(
    ee_pos: np.ndarray,
    ee_ori: np.ndarray,
    gripper_cmd: np.ndarray,
    anchor_step: int,
    chunk_size: int,
) -> np.ndarray:
    """Reconstruct one anchor-relative chunk from raw demo observations.

    Must use the SAME formulas as preprocess_libero.py::extract_chunks so that
    the round-trip comparison is apples-to-apples:

        pos_delta[k] = ee_pos[t+k+1] - ee_pos[t]
        rot_delta[k] = rotvec(R_{t+k+1} * R_t^{-1})  where R_x = from_rotvec(ee_ori[x])
        gripper[k]   = actions_raw[t+k, 6]

    Args:
        ee_pos, ee_ori, gripper_cmd: raw demo streams (see load_raw_demo_obs).
        anchor_step: t (raw step index; the preprocessed HDF5 stores this as chunk_idx).
        chunk_size: H (raw steps per chunk).

    Returns:
        (H, 7) float64 anchor-relative chunk in PHYSICAL units.
    """
    t = anchor_step
    H = chunk_size
    if t + H >= ee_pos.shape[0]:
        raise IndexError(
            f"anchor_step={t} + H={H} exceeds demo length {ee_pos.shape[0]}"
        )

    R_anchor = R.from_rotvec(ee_ori[t])
    R_target = R.from_rotvec(ee_ori[t + 1 : t + H + 1])  # (H,)
    pos_delta = ee_pos[t + 1 : t + H + 1] - ee_pos[t]     # (H, 3)
    rot_delta = (R_target * R_anchor.inv()).as_rotvec()   # (H, 3)
    grip = gripper_cmd[t : t + H, None]                   # (H, 1)

    return np.concatenate([pos_delta, rot_delta, grip], axis=-1).astype(np.float64)


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
        default=Path("/data/lyw/libero_processed/libero_spatial/"),
        help="Directory containing preprocessed HDF5 files (searched recursively).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("/data/lyw/fast_tokenizer"),
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
            chunk_stride = int(f.attrs.get("chunk_stride", 1))  # new format attr
            action_low = np.asarray(f.attrs["action_low"], dtype=np.float64)
            action_high = np.asarray(f.attrs["action_high"], dtype=np.float64)
            action_dim = int(f.attrs.get("action_dim", 7))
            source_file = str(f.attrs["source_file"])

            demo_idx = int(f["demo_idx"][local_idx])
            # chunk_idx is now the **raw step anchor** (not ci * H as in the old
            # non-overlapping format). preprocess_libero.py stores
            # samples["chunk_idx"].append(t) where t is the raw step.
            anchor_step = int(f["chunk_idx"][local_idx])

            tokens_np = np.asarray(f["fast_tokens"][local_idx], dtype=np.int64)
            token_len = int(f["fast_length"][local_idx])
            if token_len != len(tokens_np):
                # vlen stores the real length; `fast_length` is a redundant scalar.
                print(f"  [warn] fast_length={token_len} disagrees with "
                      f"len(fast_tokens)={len(tokens_np)}; using token array length.")
                token_len = len(tokens_np)

            stored_norm = np.asarray(
                f["continuous_actions"][local_idx], dtype=np.float64
            )  # (H, 7) — already anchor-relative and normalized to [-1, 1]

        # Ground-truth anchor-relative chunk reconstructed from raw LIBERO obs.
        src_path = resolve_source_file(source_file, args.libero_root)
        ee_pos, ee_ori, gripper_cmd = load_raw_demo_obs(src_path, demo_idx)
        gt_chunk_phys = compute_anchor_relative_chunk_gt(
            ee_pos, ee_ori, gripper_cmd, anchor_step, chunk_size,
        )  # (H, 7) physical units: meters / rad / [-1, 1] gripper cmd
        if gt_chunk_phys.shape != (chunk_size, action_dim):
            raise RuntimeError(
                f"GT chunk shape {gt_chunk_phys.shape}, expected "
                f"({chunk_size}, {action_dim}). demo={demo_idx} anchor={anchor_step} "
                f"src={src_path.name}"
            )

        # FAST decode: tokens → normalized → physical (via denormalize)
        decoded_norm, decoded_phys = decode_one_sample(
            tokens_np, token_len, processor,
            chunk_size, action_dim, action_low, action_high,
        )

        # ---- Error decomposition (three layers) ----
        # 1) Pure FAST codec round-trip (normalized space): decoded_norm vs
        #    stored_norm — no clipping / no GT involved. This is the intrinsic
        #    DCT+BPE quantization error.
        norm_err = np.abs(decoded_norm - stored_norm)

        # 2) Clipping gap: stored_norm vs re-normalize(gt_phys). If ~0, no
        #    clipping happened; if non-zero, the 1/99 percentile bounds cut off
        #    some of the GT distribution.
        renorm_gt = normalize_actions(gt_chunk_phys, action_low, action_high)
        clip_gap = np.abs(renorm_gt - stored_norm)

        # 3) Physical-space total error: decoded_phys vs gt_chunk_phys. This is
        #    what the robot ultimately consumes after denormalization (in meters
        #    for pos, rad for rot, gripper command unchanged).
        raw_err = np.abs(decoded_phys - gt_chunk_phys)

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
              f"demo={demo_idx}  anchor_step={anchor_step}  token_len={token_len}")
        print(f"  chunk_size H={chunk_size}  stride={chunk_stride}  action_dim={action_dim}")
        print(f"  source_file={src_path.name}")
        print(f"  physical      : mean |err|={overall:.5f}  max |err|={max_err:.5f}")
        print(f"  normalized    : mean |err|={norm_err.mean():.2e}  "
              f"max |err|={norm_err.max():.2e}   (pure FAST codec)")
        print(f"  clipping gap  : max |err|={clip_gap.max():.2e}   "
              f"(stored_norm vs renorm(gt); should be ~0)")
        print(f"  per-dim L1  (phys): {format_row(per_dim, width=7, prec=4)}")
        print(f"  per-step L1 (phys): {format_row(per_step, width=7, prec=4)}")

        # ---- Detailed per-step per-dim table for the first N samples ----
        if rank < args.print_samples:
            print()
            print("  Per-step per-dim detail (physical, anchor-relative):")
            header_dims = "  ".join(f"{'d' + str(d):>8}" for d in range(action_dim))
            print(f"    step      {header_dims}")
            print(f"    " + "-" * (11 + len(header_dims)))
            for h in range(chunk_size):
                print(f"    {h:>3d}  P | {format_row(decoded_phys[h])}")
                print(f"        G | {format_row(gt_chunk_phys[h])}")
                print(f"        E | "
                      + " ".join(f"{v:>+8.3f}" for v in raw_err[h]))

    # ---- Step 4: Aggregate summary ----
    per_dim_arr = np.stack(per_dim_l1_raw, axis=0)     # (N, action_dim)
    per_step_arr = np.stack(per_step_l1_raw, axis=0)   # (N, chunk_size)

    print()
    print("=" * 80)
    print(f"[Aggregate over {args.num_samples} samples]")
    print(f"  Physical mean |err|: "
          f"mean={np.mean(per_sample_overall_l1):.5f}  "
          f"std={np.std(per_sample_overall_l1):.5f}  "
          f"min={np.min(per_sample_overall_l1):.5f}  "
          f"max={np.max(per_sample_overall_l1):.5f}")
    print(f"  Physical max |err|:  "
          f"mean={np.mean(per_sample_max_err):.5f}  "
          f"worst={np.max(per_sample_max_err):.5f}")
    print(f"  Normalized mean |err| (pure FAST codec): "
          f"mean={np.mean(per_sample_norm_l1):.2e}  "
          f"max={np.max(per_sample_norm_l1):.2e}")
    print(f"  Clipping gap max |err| across samples: "
          f"{np.max(per_sample_clip_gap):.2e}")

    # Per-dimension breakdown — labels reflect anchor-relative chunk format:
    # [cumulative pos delta (m) × 3, cumulative rot delta (rad, axis-angle) × 3, gripper cmd]
    if per_dim_arr.shape[1] == 7:
        dim_labels = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
    else:
        dim_labels = [f"d{i}" for i in range(per_dim_arr.shape[1])]

    print("  Per-dim L1 (physical, anchor-relative):")
    for d, lbl in enumerate(dim_labels):
        col = per_dim_arr[:, d]
        print(f"    {lbl:>5}:  mean={col.mean():.5f}  "
              f"max={col.max():.5f}  std={col.std():.5f}")

    print("  Per-step L1 (physical, averaged over samples):")
    step_mean = per_step_arr.mean(axis=0)
    for h in range(per_step_arr.shape[1]):
        print(f"    step {h:>2d}: {step_mean[h]:.5f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
preprocess_libero.py — Convert LIBERO HDF5 demos into chunk-level samples with
anchor-relative action chunks and FAST tokenization.

LIBERO stores demonstration data in robomimic-style HDF5:
    data/demo_0/actions            (T, 7)         — raw OSC_POSE controller inputs
    data/demo_0/obs/ee_pos         (T, 3)         — base-frame EE position
    data/demo_0/obs/ee_ori         (T, 3)         — axis-angle rotation vector
    data/demo_0/obs/<image_key>    (T, H, W, 3)   — RGB uint8
    data/demo_0/robot_states       (T, 9)         — includes xyzw quat at [:, 5:9]
    data/demo_0/obs/gripper_states (T, 2)         — two finger joint positions

This script:
  1. Loads a single-task LIBERO HDF5 file
  2. **Sliding window** cuts trajectories into overlapping chunks at stride=1
     (was: non-overlapping stride=H, which threw away ~95% of the data)
  3. Builds **anchor-relative** 7D action chunks: each element is the cumulative
     displacement from the chunk-start observation state (anchor), computed
     from obs/ee_pos and obs/ee_ori via matrix composition. NOT raw step-deltas.
  4. Builds **9D proprio**: ee_pos(3) + xyzw quat(4) + gripper_states raw(2).
     NEVER apply mean/abs to the gripper fingers — it destroys half the info.
  5. Normalizes chunk dims to [-1, 1] using 1st/99th percentile per dim, then
     runs the FAST tokenizer (DCT + BPE).

Output HDF5 layout:
    image_agent        (N, H_img, W_img, 3) uint8  — agentview at chunk start (raw step t)
    image_hand         (N, H_img, W_img, 3) uint8  — eye-in-hand at chunk start
    proprio            (N, 9) float64               — ee_pos(3)+ee_quat(4)+gripper(2)
    continuous_actions (N, chunk_size, 7) float32   — anchor-relative, normalized to [-1,1]
    fast_tokens        (N,) vlen(int32)             — FAST token IDs per chunk
    fast_length        (N,) int32                   — token count per chunk
    demo_idx           (N,) int32                   — source demo index
    chunk_idx          (N,) int32                   — sliding-window offset within demo (raw step)
    attrs:
        language_instruction  str                   — task language description
        chunk_stride          int                   — sliding window stride (default 1)
        action_low/high       (7,) float64          — normalization bounds

Usage:
    python preprocess_libero.py \
        --input /path/to/LIBERO_task_demo.hdf5 \
        --output /path/to/output.h5 \
        --chunk-size 20 \
        --stride 1 \
        --image-key agentview_rgb \
        --hand-image-key eye_in_hand_rgb \
        --fit-tokenizer \
        --save-tokenizer /path/to/save/dir
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Step 1: Load and inspect LIBERO HDF5
# ---------------------------------------------------------------------------

def load_demo_keys(f: h5py.File) -> list[str]:
    """Enumerate demo_* groups under data/, sorted by numeric index."""
    data_group = f["data"]
    keys = sorted(
        [k for k in data_group.keys() if k.startswith("demo_")],
        key=lambda k: int(k.split("_")[1]),
    )
    return keys


def inspect_hdf5(f: h5py.File, image_key: str, hand_image_key: str, demo_keys: list[str]) -> None:
    """Print summary of the input HDF5 file and validate required keys."""
    print(f"[Step 1] Inspecting input HDF5 — {len(demo_keys)} demos found")

    # Show available keys in first demo for reference
    first_demo = f[f"data/{demo_keys[0]}"]
    print(f"  Keys in {demo_keys[0]}: {list(first_demo.keys())}")
    if "obs" in first_demo:
        print(f"  Obs keys: {list(first_demo['obs'].keys())}")

    # Validate required keys exist
    required_obs_keys = [image_key, hand_image_key, "ee_pos", "ee_ori", "gripper_states"]
    for dk in demo_keys:
        demo = f[f"data/{dk}"]
        if "actions" not in demo:
            raise ValueError(f"'actions' not found in data/{dk}")
        if "obs" not in demo:
            raise ValueError(f"'obs' not found in data/{dk}")
        for key in required_obs_keys:
            if key not in demo["obs"]:
                raise ValueError(f"'obs/{key}' not found in data/{dk}")
        # Validate robot_states for proprio extraction (contains quaternion at [5:9])
        if "robot_states" not in demo:
            raise ValueError(f"'robot_states' not found in data/{dk}")

    # Print trajectory lengths
    lengths = [f[f"data/{dk}/actions"].shape[0] for dk in demo_keys]
    print(f"  Trajectory lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, total_steps={sum(lengths)}")
    print(f"  Action dim: {f[f'data/{demo_keys[0]}/actions'].shape[1]}")
    img_shape = f[f"data/{demo_keys[0]}/obs/{image_key}"].shape[1:]
    print(f"  Agentview shape: {img_shape} (key='{image_key}')")
    hand_shape = f[f"data/{demo_keys[0]}/obs/{hand_image_key}"].shape[1:]
    print(f"  Hand image shape: {hand_shape} (key='{hand_image_key}')")
    rs_dim = f[f"data/{demo_keys[0]}/robot_states"].shape[1]
    grip_dim = f[f"data/{demo_keys[0]}/obs/gripper_states"].shape[1]
    print(f"  robot_states: {rs_dim}d (using [5:9] as xyzw quaternion)")
    print(f"  gripper_states: {grip_dim}d (both raw finger positions, no averaging)")
    print(f"  proprio layout: ee_pos(3) + xyzw_quat(4) + gripper_raw(2) = 9d")


# ---------------------------------------------------------------------------
# Step 2: Proprio + chunk normalization helpers
# ---------------------------------------------------------------------------

def normalize_proprio(raw: np.ndarray) -> np.ndarray:
    """Normalize 9d proprio vector: ee_pos(3) + ee_quat(4) + gripper(2).

    - Position (dims 0:3): passed through unchanged.
    - Quaternion (dims 3:7): re-normalize to unit length as a defensive safeguard
      against numerical drift.
    - Gripper (dims 7:9): two raw finger joint positions, passed through unchanged.
      Do NOT take mean/abs — the two fingers are symmetric around 0, so averaging
      either destroys half the information or zeros out entirely if `abs` is omitted.

    This function is the single source of truth for proprio normalization. Both
    preprocess_libero.py and eval_libero.py must call it on the same 9d layout.

    Args:
        raw: (..., 9) array — [ee_pos(3), ee_quat(4), gripper(2)]

    Returns:
        (..., 9) array with quaternion re-normalized.
    """
    out = raw.copy()
    quat = out[..., 3:7]
    quat_norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    out[..., 3:7] = quat / (quat_norm + 1e-8)
    return out


def compute_action_stats(
    action_chunks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1st and 99th percentile per dim from anchor-relative chunks.

    Args:
        action_chunks: list of (H, 7) float arrays in physical units
            (unnormalized anchor-relative displacements).

    Returns:
        action_low:  (7,) — 1st percentile per dimension
        action_high: (7,) — 99th percentile per dimension
    """
    print("[Step 2] Computing action normalization statistics ...")
    if not action_chunks:
        raise ValueError("No action chunks provided to compute_action_stats")

    # Stack and flatten time dim: (N, H, D) -> (N*H, D)
    stacked = np.stack(action_chunks, axis=0).astype(np.float64)  # (N, H, D)
    flat = stacked.reshape(-1, stacked.shape[-1])  # (N*H, D)
    print(f"  Total chunk elements: {flat.shape[0]}, dim: {flat.shape[1]}")

    action_low = np.percentile(flat, 1, axis=0)
    action_high = np.percentile(flat, 99, axis=0)

    zero_range_dims = np.where((action_high - action_low) < 1e-8)[0]
    if len(zero_range_dims) > 0:
        print(f"  WARNING: Dimensions {zero_range_dims.tolist()} have near-zero range "
              "and will be normalized to constant 0.0")

    dim_labels = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
    for d in range(flat.shape[1]):
        lbl = dim_labels[d] if d < len(dim_labels) else f"d{d}"
        print(f"  {lbl:>4}: raw range [{flat[:, d].min():.4f}, {flat[:, d].max():.4f}], "
              f"p1={action_low[d]:.4f}, p99={action_high[d]:.4f}")

    return action_low, action_high


def normalize_actions(
    actions: np.ndarray, action_low: np.ndarray, action_high: np.ndarray
) -> np.ndarray:
    """Normalize actions to [-1, 1] using percentile bounds, then clip.

    Formula: a_norm = 2 * (a - low) / (high - low) - 1
    Dimensions with zero range (high == low) are set to 0.
    """
    denom = action_high - action_low
    # For constant dimensions (near-zero range), output 0.0 instead of a misleading value
    zero_range = denom < 1e-8
    safe_denom = np.where(zero_range, 1.0, denom)
    normed = np.where(zero_range, 0.0, 2.0 * (actions - action_low) / safe_denom - 1.0)
    return np.clip(normed, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Step 3: Extract anchor-relative sliding-window samples
# ---------------------------------------------------------------------------

def extract_chunks(
    f: h5py.File,
    demo_keys: list[str],
    image_key: str,
    hand_image_key: str,
    chunk_size: int,
    stride: int,
) -> dict[str, list]:
    """Sliding-window extraction of anchor-relative action chunks.

    For each valid chunk start ``t`` (0, stride, 2*stride, ..., last-valid-start):

      - ``image_agent[t]``, ``image_hand[t]``  — raw-step-``t`` observations
      - ``proprio[t]`` — 9D: ``ee_pos(3) + robot_states[5:9] xyzw-quat(4) + gripper_states[0:2](2)``
      - ``action_chunk[t]`` — (H, 7) float32 in PHYSICAL units (meters + radians + command):
            for k in 0..H-1:
              pos_delta[k]  = obs/ee_pos[t+k+1] - obs/ee_pos[t]
              rot_delta[k]  = rotvec(R_{t+k+1} · R_t^{-1})  with R_x = from_rotvec(obs/ee_ori[x])
              gripper[k]    = actions_raw[t+k, 6]
        The chunks are NOT normalized here — they're returned in physical units so
        the caller can compute the 1/99 percentile from the real distribution and
        then normalize.

    Valid chunk start range: ``0 <= t <= T - H - 1`` (we need ``obs[t+H]`` to compute
    the last anchor-relative delta, so the last valid anchor is ``T - H - 1``).

    Args:
        f: open h5py.File
        demo_keys: list of "demo_N" strings under data/
        image_key: agentview image key under obs/
        hand_image_key: eye-in-hand image key under obs/
        chunk_size: H (raw steps per chunk)
        stride: sliding window stride in raw steps; ``stride=1`` is the standard
            VLA practice (maximum data augmentation via temporal shift).

    Returns:
        dict of lists (one entry per sample). Key ``continuous_actions`` holds
        the PHYSICAL-unit chunks; the caller must normalize them.
    """
    H = chunk_size
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    print(f"[Step 3] Extracting anchor-relative sliding-window chunks "
          f"(H={H}, stride={stride}) ...")

    samples: dict[str, list] = {
        "image_agent": [],
        "image_hand": [],
        "proprio": [],
        # State-prediction targets (frame at t+H — one step after the chunk
        # completes). These are required by libero_dataset.py when
        # ``use_state_prediction=True`` is set in the training config. They
        # always populated here so downstream code can pick them up without
        # needing to re-preprocess for every ablation toggle.
        "image_agent_future": [],
        "image_hand_future": [],
        "proprio_future": [],
        "continuous_actions": [],  # anchor-relative, physical units (meters/rad/command)
        "demo_idx": [],
        "chunk_idx": [],
    }

    total_chunks = 0
    skipped_demos = 0

    for demo_i, dk in enumerate(demo_keys):
        actions_raw = f[f"data/{dk}/actions"][()]                  # (T, 7)
        T = actions_raw.shape[0]

        # We need obs at step t+H to compute the last anchor-relative delta.
        # Last valid anchor: t <= T - H - 1. Minimum T for any chunk: T >= H + 1.
        if T < H + 1:
            print(f"  Skipping {dk}: T={T} < H+1={H + 1}")
            skipped_demos += 1
            continue

        # n_chunks with stride s: starts at 0, s, 2s, ..., last_start where
        # last_start <= T - H - 1. Count = (T - H - 1) // s + 1.
        n_chunks = (T - H - 1) // stride + 1

        # Highest raw-step index we access in this demo:
        #   obs / ee_pos / ee_ori: (n_chunks-1)*stride + H
        #   actions (gripper only): (n_chunks-1)*stride + H - 1
        last_obs_idx = (n_chunks - 1) * stride + H
        last_action_idx = last_obs_idx - 1  # actions[t+k] for k up to H-1

        agent_imgs = f[f"data/{dk}/obs/{image_key}"][: last_obs_idx + 1]
        hand_imgs = f[f"data/{dk}/obs/{hand_image_key}"][: last_obs_idx + 1]
        ee_pos = f[f"data/{dk}/obs/ee_pos"][: last_obs_idx + 1]              # (<=T, 3)
        ee_ori = f[f"data/{dk}/obs/ee_ori"][: last_obs_idx + 1]              # (<=T, 3) axis-angle
        robot_states = f[f"data/{dk}/robot_states"][: last_obs_idx + 1]      # (<=T, 9)
        grip_states = f[f"data/{dk}/obs/gripper_states"][: last_obs_idx + 1]  # (<=T, 2)
        gripper_cmd = actions_raw[: last_action_idx + 1, 6]                  # (<=T,)

        # Pre-compute all rotations for this demo once (scipy batches).
        Rs = R.from_rotvec(ee_ori)  # "shape" (last_obs_idx+1,)

        # Vectorized index tables for fast anchor-relative deltas.
        starts_arr = np.arange(n_chunks) * stride               # (n_chunks,)
        k_p1 = np.arange(1, H + 1)                              # (H,) = [1..H]
        # target_idx[ci, k] = starts_arr[ci] + k + 1 (raw step index to read obs)
        target_idx = starts_arr[:, None] + k_p1[None, :]        # (n_chunks, H)
        target_flat = target_idx.reshape(-1)                    # (n_chunks*H,)
        anchor_flat = np.repeat(starts_arr, H)                  # (n_chunks*H,)

        # Position delta: pos[t+k+1] - pos[t]
        pos_deltas = ee_pos[target_flat] - ee_pos[anchor_flat]  # (n_chunks*H, 3)
        pos_deltas = pos_deltas.reshape(n_chunks, H, 3)

        # Rotation delta: rotvec(R[t+k+1] * R[t]^-1), via scipy Rotation indexing
        R_target = Rs[target_flat]
        R_anchor = Rs[anchor_flat]
        R_delta = R_target * R_anchor.inv()
        rot_deltas = R_delta.as_rotvec().reshape(n_chunks, H, 3)  # (n_chunks, H, 3)

        # Gripper command at raw step t+k for k in 0..H-1  (= target_idx - 1)
        gripper_idx = (starts_arr[:, None] + np.arange(H)[None, :])  # (n_chunks, H)
        gripper_values = gripper_cmd[gripper_idx.reshape(-1)].reshape(n_chunks, H, 1)

        # Assemble (n_chunks, H, 7) anchor-relative chunks
        chunks = np.concatenate([pos_deltas, rot_deltas, gripper_values],
                                axis=-1).astype(np.float32)

        # Anchor state per chunk
        for ci in range(n_chunks):
            t = int(starts_arr[ci])
            t_future = t + H  # frame at t+H — same index already loaded above

            samples["image_agent"].append(agent_imgs[t])  # (H_img, W_img, 3) uint8
            samples["image_hand"].append(hand_imgs[t])

            # State-prediction targets at t+H (LeWM-style next-embedding loss).
            # The chunk-validity range t <= T-H-1 already guarantees t+H <= T-1,
            # and we read up to last_obs_idx = (n_chunks-1)*stride + H above,
            # so agent_imgs[t+H] / hand_imgs[t+H] are always in-bounds here.
            samples["image_agent_future"].append(agent_imgs[t_future])
            samples["image_hand_future"].append(hand_imgs[t_future])

            # 9D proprio: ee_pos(3) + xyzw-quat(4) + gripper_states raw(2)
            # NEVER mean/abs the gripper — fingers are symmetric, averaging destroys info.
            proprio_raw = np.concatenate([
                ee_pos[t],                # (3,)
                robot_states[t, 5:9],     # (4,) xyzw quaternion (verified)
                grip_states[t],           # (2,) raw two finger joint positions
            ])
            samples["proprio"].append(normalize_proprio(proprio_raw))  # (9,) float64

            # Same 9D layout at t+H — single source of truth for proprio
            # normalization is `normalize_proprio` (re-normalizes the quat).
            proprio_future_raw = np.concatenate([
                ee_pos[t_future],
                robot_states[t_future, 5:9],
                grip_states[t_future],
            ])
            samples["proprio_future"].append(normalize_proprio(proprio_future_raw))

            samples["continuous_actions"].append(chunks[ci])  # (H, 7) physical units
            samples["demo_idx"].append(demo_i)
            samples["chunk_idx"].append(t)  # record raw-step anchor, not an arbitrary counter

        total_chunks += n_chunks

    if skipped_demos > 0:
        print(f"  Skipped {skipped_demos} demos (too short for H+1={H + 1} frames)")
    print(f"  Extracted {total_chunks} chunks from {len(demo_keys) - skipped_demos} demos")

    return samples


# ---------------------------------------------------------------------------
# Step 4: FAST tokenization
# ---------------------------------------------------------------------------

def _patch_saved_tokenizer(save_dir: Path) -> None:
    """Ensure a saved FAST tokenizer can be reloaded via AutoProcessor.

    save_pretrained() writes the BPE vocab and config, but omits:
    1. The custom processor Python file (processing_action_tokenizer.py)
    2. The auto_map entry in processor_config.json

    Without these, AutoProcessor.from_pretrained() loads a plain HF tokenizer
    that can't handle numpy action chunks.

    Note on filename: ``ProcessorMixin.save_pretrained`` writes
    ``processor_config.json`` (NOT ``preprocessor_config.json``, which is
    the ``ImageProcessingMixin``/``FeatureExtractorMixin`` filename). An
    earlier version of this function wrote to ``preprocessor_config.json``,
    which silently created an unreferenced file and left ``--load-tokenizer``
    falling back to the plain HF tokenizer. Verified against the live
    layout in ``data/fast_tokenizer/`` (contains ``processor_config.json``).
    """
    # 1. Copy processing_action_tokenizer.py into the saved directory
    processor_src = Path(__file__).parent / "data" / "fast_tokenizer" / "processing_action_tokenizer.py"
    if processor_src.exists():
        shutil.copy2(processor_src, save_dir / "processing_action_tokenizer.py")
    else:
        print(f"  WARNING: {processor_src} not found, skipping processor file copy")

    # 2. Inject auto_map into processor_config.json (NOT preprocessor_config.json)
    config_path = save_dir / "processor_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = {}
    config["auto_map"] = {
        "AutoProcessor": "processing_action_tokenizer.UniversalActionProcessor"
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def tokenize_actions(
    action_chunks: np.ndarray,
    fit: bool = False,
    save_tokenizer_path: str | None = None,
    load_tokenizer_path: str | None = None,
) -> tuple[list[np.ndarray], Any]:
    """Tokenize action chunks using the FAST tokenizer.

    Args:
        action_chunks: (N, H, action_dim) float32, normalized to [-1, 1]
        fit: if True, train a domain-specific BPE on this data
        save_tokenizer_path: directory to save the fitted tokenizer
        load_tokenizer_path: directory to load a previously fitted tokenizer

    Returns:
        tokens_list: list of N numpy int32 arrays (variable length)
        tokenizer: the tokenizer instance (for potential later use)
    """
    from transformers import AutoProcessor

    print("[Step 4] FAST tokenization ...")
    print(f"  Action chunks shape: {action_chunks.shape}")

    # Load tokenizer
    if load_tokenizer_path is not None:
        print(f"  Loading fitted tokenizer from: {load_tokenizer_path}")
        tokenizer = AutoProcessor.from_pretrained(
            load_tokenizer_path, trust_remote_code=True
        )
    else:
        print("  Loading universal FAST tokenizer from HuggingFace ...")
        tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )

    # Optionally fit a domain-specific BPE vocabulary
    if fit:
        if load_tokenizer_path is not None:
            print("  WARNING: --fit-tokenizer ignored because --load-tokenizer was provided.")
        else:
            print(f"  Fitting tokenizer on {action_chunks.shape[0]} chunks ...")
            fitted = tokenizer.fit(action_chunks)
            if fitted is None:
                raise RuntimeError(
                    "tokenizer.fit() returned None — check FAST API version. "
                    "Expected it to return the fitted tokenizer instance."
                )
            tokenizer = fitted
            print("  Fitting complete.")

            if save_tokenizer_path is not None:
                save_dir = Path(save_tokenizer_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                tokenizer.save_pretrained(save_tokenizer_path)

                # save_pretrained() doesn't copy the custom processor code or
                # the auto_map needed for AutoProcessor.from_pretrained().
                # Fix both so --load-tokenizer works out of the box.
                _patch_saved_tokenizer(save_dir)
                print(f"  Saved fitted tokenizer to: {save_tokenizer_path}")

    # Encode all chunks in one call (batched)
    # FAST's AutoProcessor (trust_remote_code=True) is a custom processor.
    # Its __call__ may return either:
    #   - a list of token lists directly, or
    #   - a BatchEncoding dict with "input_ids" key (standard HF convention)
    print("  Encoding action chunks to FAST tokens ...")
    tokens_raw = tokenizer(action_chunks)

    # Handle both possible return types from the FAST processor
    if isinstance(tokens_raw, dict) or hasattr(tokens_raw, "input_ids"):
        # Standard HF BatchEncoding — extract input_ids
        tokens_raw = tokens_raw["input_ids"]

    # Convert to list of numpy int32 arrays
    tokens_list = [np.array(t, dtype=np.int32) for t in tokens_raw]

    # Print token length statistics
    lengths = [len(t) for t in tokens_list]
    print(f"  Token lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.0f}, "
          f"std={np.std(lengths):.1f}")

    return tokens_list, tokenizer


# ---------------------------------------------------------------------------
# Step 5: Save to output HDF5
# ---------------------------------------------------------------------------

def extract_language_instruction(f: h5py.File) -> str:
    """Extract language instruction from LIBERO HDF5 data attributes.

    Looks for the instruction in data attrs 'problem_info' (JSON with
    'language_instruction' key), falling back to empty string.
    """
    data_attrs = f["data"].attrs
    if "problem_info" in data_attrs:
        try:
            info = json.loads(data_attrs["problem_info"])
            return info.get("language_instruction", "")
        except (json.JSONDecodeError, TypeError):
            pass
    return ""


def save_hdf5(
    output_path: str,
    samples: dict[str, list],
    tokens_list: list[np.ndarray],
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    chunk_size: int,
    chunk_stride: int,
    image_key: str,
    source_file: str,
    num_demos: int,
    language_instruction: str = "",
    save_tokenizer_path: str | None = None,
    load_tokenizer_path: str | None = None,
) -> None:
    """Write chunk-level samples to output HDF5.

    fast_tokens uses h5py variable-length dataset for ragged token arrays.
    Images are stored with chunked layout + gzip compression.
    """
    N = len(tokens_list)
    print(f"[Step 5] Saving {N} samples to {output_path} ...")

    with h5py.File(output_path, "w") as out:

        # --- Agentview images (chunked + compressed) ---
        img_shape = samples["image_agent"][0].shape  # (H_img, W_img, 3)
        ds_agent = out.create_dataset(
            "image_agent",
            shape=(N, *img_shape),
            dtype=np.uint8,
            chunks=(1, *img_shape),
            compression="gzip",
            compression_opts=4,
        )
        # --- Hand images (chunked + compressed) ---
        ds_hand = out.create_dataset(
            "image_hand",
            shape=(N, *img_shape),
            dtype=np.uint8,
            chunks=(1, *img_shape),
            compression="gzip",
            compression_opts=4,
        )
        # --- Future-frame images at raw step t+H (state-prediction target) ---
        ds_agent_future = out.create_dataset(
            "image_agent_future",
            shape=(N, *img_shape),
            dtype=np.uint8,
            chunks=(1, *img_shape),
            compression="gzip",
            compression_opts=4,
        )
        ds_hand_future = out.create_dataset(
            "image_hand_future",
            shape=(N, *img_shape),
            dtype=np.uint8,
            chunks=(1, *img_shape),
            compression="gzip",
            compression_opts=4,
        )
        for i in range(N):
            ds_agent[i] = samples["image_agent"][i]
            ds_hand[i] = samples["image_hand"][i]
            ds_agent_future[i] = samples["image_agent_future"][i]
            ds_hand_future[i] = samples["image_hand_future"][i]

        # --- Proprioception (current and future) ---
        out.create_dataset(
            "proprio",
            data=np.stack(samples["proprio"], axis=0),  # (N, 9)
            dtype=np.float64,
        )
        out.create_dataset(
            "proprio_future",
            data=np.stack(samples["proprio_future"], axis=0),  # (N, 9) at t+H
            dtype=np.float64,
        )

        # --- Continuous actions ---
        action_dim = samples["continuous_actions"][0].shape[1]
        out.create_dataset(
            "continuous_actions",
            data=np.stack(samples["continuous_actions"], axis=0),  # (N, H, 7)
            dtype=np.float32,
        )

        # --- FAST tokens (variable-length) ---
        vlen_dt = h5py.vlen_dtype(np.int32)
        ds_tokens = out.create_dataset("fast_tokens", shape=(N,), dtype=vlen_dt)
        for i, tok in enumerate(tokens_list):
            ds_tokens[i] = tok

        # --- Token lengths ---
        fast_lengths = np.array([len(t) for t in tokens_list], dtype=np.int32)
        out.create_dataset("fast_length", data=fast_lengths)

        # --- Demo / chunk indices ---
        out.create_dataset("demo_idx", data=np.array(samples["demo_idx"], dtype=np.int32))
        out.create_dataset("chunk_idx", data=np.array(samples["chunk_idx"], dtype=np.int32))

        # --- Attributes (metadata for downstream use) ---
        out.attrs["action_low"] = action_low.astype(np.float64)
        out.attrs["action_high"] = action_high.astype(np.float64)
        out.attrs["chunk_size"] = chunk_size
        out.attrs["chunk_stride"] = chunk_stride
        out.attrs["image_key"] = image_key
        out.attrs["source_file"] = str(Path(source_file).resolve())
        out.attrs["num_demos"] = num_demos
        out.attrs["num_samples"] = N
        out.attrs["action_dim"] = action_dim
        out.attrs["language_instruction"] = language_instruction
        # Store tokenizer paths separately to avoid overwrite
        if save_tokenizer_path is not None:
            out.attrs["tokenizer_save_path"] = str(Path(save_tokenizer_path).resolve())
        if load_tokenizer_path is not None:
            out.attrs["tokenizer_load_path"] = str(Path(load_tokenizer_path).resolve())

    print(f"  Saved successfully. File size: {Path(output_path).stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Step 6: Print statistics
# ---------------------------------------------------------------------------

def print_statistics(
    samples: dict[str, list],
    tokens_list: list[np.ndarray],
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> None:
    """Print detailed statistics about the processed dataset."""
    N = len(tokens_list)
    print(f"\n{'=' * 60}")
    print(f"[Step 6] Dataset Statistics")
    print(f"{'=' * 60}")

    # --- Sample counts ---
    demo_indices = np.array(samples["demo_idx"])
    unique_demos = np.unique(demo_indices)
    print(f"  Total samples: {N}")
    print(f"  Demos used: {len(unique_demos)}")
    chunks_per_demo = [np.sum(demo_indices == d) for d in unique_demos]
    print(f"  Chunks per demo: min={min(chunks_per_demo)}, max={max(chunks_per_demo)}, "
          f"mean={np.mean(chunks_per_demo):.1f}")

    # --- Proprio stats ---
    if samples["proprio"]:
        proprio_all = np.stack(samples["proprio"], axis=0)  # (N, 9)
        print(f"\n  Proprioception stats ({proprio_all.shape[1]}d):")
        # xyzw quaternion order (verified — robosuite / scipy convention)
        labels = ["ee_pos_x", "ee_pos_y", "ee_pos_z",
                  "ee_quat_x", "ee_quat_y", "ee_quat_z", "ee_quat_w",
                  "grip_left", "grip_right"]
        for d in range(proprio_all.shape[1]):
            col = proprio_all[:, d]
            lbl = labels[d] if d < len(labels) else f"dim{d}"
            print(f"    {lbl}: min={col.min():.4f}, max={col.max():.4f}, "
                  f"mean={col.mean():.4f}, std={col.std():.4f}")

    # --- Action stats (after normalization) ---
    all_actions = np.stack(samples["continuous_actions"], axis=0)  # (N, H, 7)
    flat_actions = all_actions.reshape(-1, all_actions.shape[-1])  # (N*H, 7)
    print(f"\n  Normalized action stats (should be in [-1, 1]):")
    for d in range(flat_actions.shape[1]):
        col = flat_actions[:, d]
        print(f"    Dim {d}: min={col.min():.4f}, max={col.max():.4f}, "
              f"mean={col.mean():.4f}, std={col.std():.4f}")

    # --- Token statistics ---
    lengths = np.array([len(t) for t in tokens_list])
    print(f"\n  FAST token lengths:")
    print(f"    min={lengths.min()}, max={lengths.max()}, "
          f"mean={lengths.mean():.1f}, median={np.median(lengths):.0f}, "
          f"std={lengths.std():.1f}")

    # --- Vocab usage frequency (top-20) ---
    token_counter: Counter[int] = Counter()
    for tok_arr in tokens_list:
        token_counter.update(tok_arr.tolist())

    total_tokens = sum(token_counter.values())
    unique_tokens = len(token_counter)
    print(f"\n  Vocab usage: {unique_tokens} unique token IDs out of "
          f"{total_tokens} total tokens")
    print(f"  Top-20 most frequent token IDs:")
    for token_id, count in token_counter.most_common(20):
        pct = 100.0 * count / total_tokens
        print(f"    ID {token_id:>5d}: {count:>8d} ({pct:5.2f}%)")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess LIBERO HDF5 demos into chunk-level FAST-tokenized samples."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input LIBERO HDF5 file (one task).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the output HDF5 file.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=20,
        help="Action chunk length H in raw env steps (default: 20 = 1s at 20Hz).",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Sliding window stride in raw steps (default: 1). stride=1 is the "
             "standard VLA practice — consecutive chunks share H-1 actions but "
             "each gets a fresh observation anchor, maximizing data coverage. "
             "Using stride=H would reduce per-demo sample count by ~H and is "
             "strongly discouraged.",
    )
    parser.add_argument(
        "--image-key", default="agentview_rgb",
        help="Agentview image key in the HDF5 (default: agentview_rgb).",
    )
    parser.add_argument(
        "--hand-image-key", default="eye_in_hand_rgb",
        help="Eye-in-hand image key in the HDF5 (default: eye_in_hand_rgb).",
    )
    parser.add_argument(
        "--max-action-tokens", type=int, default=None,
        help="Assert that no token sequence exceeds this length. If None, skip assertion.",
    )
    parser.add_argument(
        "--fit-tokenizer", action="store_true",
        help="Train a LIBERO-specific FAST BPE tokenizer instead of using the universal one.",
    )
    parser.add_argument(
        "--save-tokenizer", default=None,
        help="Directory to save the fitted FAST tokenizer.",
    )
    parser.add_argument(
        "--load-tokenizer", default=None,
        help="Directory to load a previously fitted FAST tokenizer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Step 1: Load and inspect
    print(f"Opening {args.input} ...")
    with h5py.File(args.input, "r") as f:
        demo_keys = load_demo_keys(f)
        if not demo_keys:
            sys.exit("ERROR: No demo_* groups found under data/")
        inspect_hdf5(f, args.image_key, args.hand_image_key, demo_keys)

        # Extract language instruction
        language_instruction = extract_language_instruction(f)
        print(f"  Language instruction: '{language_instruction}'")

        # Step 2+3: Extract anchor-relative sliding-window chunks in PHYSICAL units.
        # Stats are computed from the chunks themselves (not raw HDF5 actions) so
        # that the 1/99 percentile reflects the actual distribution the model sees.
        samples = extract_chunks(
            f, demo_keys, args.image_key, args.hand_image_key,
            args.chunk_size, stride=args.stride,
        )

    if not samples["continuous_actions"]:
        sys.exit("ERROR: No chunks extracted. Check trajectory lengths vs chunk size.")

    # Compute per-dim percentile stats from the anchor-relative chunks.
    action_low, action_high = compute_action_stats(samples["continuous_actions"])

    # Normalize chunks in place to [-1, 1] using the fitted bounds.
    samples["continuous_actions"] = [
        normalize_actions(chunk, action_low, action_high)
        for chunk in samples["continuous_actions"]
    ]

    # Step 4: FAST tokenization (on normalized anchor-relative chunks).
    all_action_chunks = np.stack(samples["continuous_actions"], axis=0)  # (N, H, 7)
    tokens_list, _tokenizer = tokenize_actions(
        all_action_chunks,
        fit=args.fit_tokenizer,
        save_tokenizer_path=args.save_tokenizer,
        load_tokenizer_path=args.load_tokenizer,
    )

    # Token length assertion (CRITICAL — see CLAUDE.md)
    # Do NOT silently truncate; raise an error so max_action_tokens can be increased.
    max_observed = max(len(t) for t in tokens_list)
    print(f"  Max observed token length: {max_observed}")
    if args.max_action_tokens is not None and max_observed > args.max_action_tokens:
        sys.exit(
            f"ERROR: max observed FAST token length ({max_observed}) exceeds "
            f"max_action_tokens ({args.max_action_tokens}). Increase max_action_tokens "
            f"in config and rerun."
        )

    # Step 5: Save output HDF5
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_hdf5(
        args.output, samples, tokens_list, action_low, action_high,
        chunk_size=args.chunk_size,
        chunk_stride=args.stride,
        image_key=args.image_key,
        source_file=args.input,
        num_demos=len(demo_keys),
        language_instruction=language_instruction,
        save_tokenizer_path=args.save_tokenizer,
        load_tokenizer_path=args.load_tokenizer,
    )

    # Step 6: Print statistics
    print_statistics(samples, tokens_list, action_low, action_high)

    print("Done!")


if __name__ == "__main__":
    main()

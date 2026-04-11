"""
preprocess_libero.py — Convert LIBERO HDF5 demos into chunk-level samples with FAST-tokenized actions.

LIBERO stores demonstration data in robomimic-style HDF5:
    data/demo_0/actions          (T, 7)         — 6D EE pose + 1D gripper
    data/demo_0/obs/<image_key>  (T, H, W, 3)   — RGB uint8

This script:
  1. Loads a single-task LIBERO HDF5 file
  2. Normalizes actions to [-1, 1] using 1st/99th percentile per dimension
  3. Cuts trajectories into non-overlapping chunks of H steps (chunk-level indexing)
  4. Tokenizes each action chunk with the FAST tokenizer (DCT + BPE)
  5. Saves chunk-level samples to a new HDF5 with variable-length token arrays

Output HDF5 layout:
    image_agent        (N, H_img, W_img, 3) uint8   — agentview at chunk start
    image_hand         (N, H_img, W_img, 3) uint8   — eye-in-hand at chunk start
    proprio            (N, 8) float64                 — ee_pos(3)+ee_quat(4)+gripper(1)
    continuous_actions  (N, chunk_size, 7)  float32   — normalized actions
    fast_tokens        (N,) vlen(int32)               — FAST token IDs per chunk
    fast_length        (N,) int32                     — token count per chunk
    demo_idx           (N,) int32                     — source demo index
    chunk_idx          (N,) int32                     — chunk index within demo
    attrs:
        language_instruction  str                     — task language description

Usage:
    python preprocess_libero.py \
        --input /path/to/LIBERO_task_demo.hdf5 \
        --output /path/to/output.h5 \
        --chunk-size 20 \
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
    for dk in demo_keys:
        demo = f[f"data/{dk}"]
        if "actions" not in demo:
            raise ValueError(f"'actions' not found in data/{dk}")
        if "obs" not in demo or image_key not in demo["obs"]:
            raise ValueError(f"'obs/{image_key}' not found in data/{dk}")
        if "obs" not in demo or hand_image_key not in demo["obs"]:
            raise ValueError(f"'obs/{hand_image_key}' not found in data/{dk}")
        # Validate robot_states for proprio extraction (contains ee_pos, ee_quat, gripper)
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
    print(f"  robot_states: {rs_dim}d → proprio: ee_pos(3) + ee_quat(4) + gripper(1) = 8d")


# ---------------------------------------------------------------------------
# Step 2: Collect actions and compute normalization statistics
# ---------------------------------------------------------------------------

def normalize_proprio(raw_8d: np.ndarray) -> np.ndarray:
    """Normalize 8d proprio vector: ee_pos(3) + ee_quat(4) + gripper(1).

    - Quaternion (dims 3:7): re-normalize to unit length as a defensive safeguard.
    - Position and gripper: passed through unchanged.

    This function is the single source of truth for proprio normalization.
    Dataset loading and eval must both call this same function.

    Args:
        raw_8d: (..., 8) array — [ee_pos(3), ee_quat(4), gripper(1)]

    Returns:
        (..., 8) array with quaternion re-normalized.
    """
    out = raw_8d.copy()
    quat = out[..., 3:7]
    quat_norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    out[..., 3:7] = quat / (quat_norm + 1e-8)
    return out


def compute_action_stats(
    f: h5py.File, demo_keys: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1st and 99th percentile per action dimension across all demos.

    Returns:
        action_low:  (action_dim,) — 1st percentile per dimension
        action_high: (action_dim,) — 99th percentile per dimension
    """
    print("[Step 2] Computing action normalization statistics ...")
    all_actions = []
    for dk in demo_keys:
        actions = f[f"data/{dk}/actions"][()]  # (T, 7)
        all_actions.append(actions)
    all_actions = np.concatenate(all_actions, axis=0)  # (total_steps, 7)
    print(f"  Total action steps: {all_actions.shape[0]}, dim: {all_actions.shape[1]}")

    action_low = np.percentile(all_actions, 1, axis=0)   # (7,)
    action_high = np.percentile(all_actions, 99, axis=0)  # (7,)

    # Warn about constant dimensions
    zero_range_dims = np.where((action_high - action_low) < 1e-8)[0]
    if len(zero_range_dims) > 0:
        print(f"  WARNING: Dimensions {zero_range_dims.tolist()} have near-zero range "
              "and will be normalized to constant 0.0")

    # Print per-dimension stats
    for d in range(all_actions.shape[1]):
        print(f"  Dim {d}: raw range [{all_actions[:, d].min():.4f}, "
              f"{all_actions[:, d].max():.4f}], "
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
# Step 3: Extract chunk-aligned samples
# ---------------------------------------------------------------------------

def extract_chunks(
    f: h5py.File,
    demo_keys: list[str],
    image_key: str,
    hand_image_key: str,
    chunk_size: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> dict[str, list]:
    """Cut each demo into non-overlapping chunks of `chunk_size` raw steps.

    Chunk-level indexing (see CLAUDE.md "Temporal Indexing Convention"):
      - Chunk i uses raw steps [i*H, (i+1)*H - 1] for actions
      - image_agent = frame at raw step i*H
      - image_hand  = frame at raw step i*H
      - proprio     = state at raw step i*H
      - n_chunks = T // H (no future frame needed)

    Returns dict of lists (one entry per chunk sample).
    """
    H = chunk_size
    print(f"[Step 3] Extracting chunk-aligned samples (H={H}) ...")

    samples: dict[str, list] = {
        "image_agent": [],
        "image_hand": [],
        "proprio": [],
        "continuous_actions": [],
        "demo_idx": [],
        "chunk_idx": [],
    }

    total_chunks = 0
    skipped_demos = 0

    for demo_i, dk in enumerate(demo_keys):
        actions_raw = f[f"data/{dk}/actions"][()]    # (T, 7)
        T = actions_raw.shape[0]

        # Need at least H steps for one complete action chunk
        n_chunks = T // H
        if n_chunks == 0:
            print(f"  Skipping {dk}: T={T}, yields 0 chunks with H={H}")
            skipped_demos += 1
            continue

        # Load images for this demo (only frames we need)
        last_needed_frame = (n_chunks - 1) * H  # highest frame index we'll access
        agent_imgs = f[f"data/{dk}/obs/{image_key}"][:last_needed_frame + 1]
        hand_imgs = f[f"data/{dk}/obs/{hand_image_key}"][:last_needed_frame + 1]

        # Load proprio sources:
        #   ee_pos from obs/ee_pos (3d)
        #   ee_quat from robot_states[5:9] (4d) — NOT obs/ee_ori which is euler
        #   gripper from mean(obs/gripper_states) (1d) — mean of 2 symmetric fingers
        ee_pos = f[f"data/{dk}/obs/ee_pos"][:last_needed_frame + 1]          # (<=T, 3)
        robot_states = f[f"data/{dk}/robot_states"][:last_needed_frame + 1]  # (<=T, 9)
        grip_states = f[f"data/{dk}/obs/gripper_states"][:last_needed_frame + 1]  # (<=T, 2)

        # Normalize this demo's actions
        actions_norm = normalize_actions(actions_raw, action_low, action_high)

        for ci in range(n_chunks):
            start = ci * H
            end = start + H           # exclusive for action slice

            samples["image_agent"].append(agent_imgs[start])   # (H_img, W_img, 3) uint8
            samples["image_hand"].append(hand_imgs[start])     # (H_img, W_img, 3) uint8

            # Proprio: obs/ee_pos(3) + robot_states[5:9] quat(4) + mean(gripper_states)(1) = 8d
            proprio_raw = np.concatenate([
                ee_pos[start],                                    # (3,) from obs/ee_pos
                robot_states[start, 5:9],                         # (4,) quaternion from robot_states
                [np.mean(np.abs(grip_states[start]))],            # (1,) mean of 2 finger widths
            ])
            samples["proprio"].append(normalize_proprio(proprio_raw))  # (8,) float64

            samples["continuous_actions"].append(actions_norm[start:end])  # (H, 7)
            samples["demo_idx"].append(demo_i)
            samples["chunk_idx"].append(ci)

        total_chunks += n_chunks

    if skipped_demos > 0:
        print(f"  Skipped {skipped_demos} demos (too short or zero chunks)")
    print(f"  Extracted {total_chunks} chunks from {len(demo_keys) - skipped_demos} demos")

    return samples


# ---------------------------------------------------------------------------
# Step 4: FAST tokenization
# ---------------------------------------------------------------------------

def _patch_saved_tokenizer(save_dir: Path) -> None:
    """Ensure a saved FAST tokenizer can be reloaded via AutoProcessor.

    save_pretrained() writes the BPE vocab and config, but omits:
    1. The custom processor Python file (processing_action_tokenizer.py)
    2. The auto_map entry in preprocessor_config.json

    Without these, AutoProcessor.from_pretrained() loads a plain HF tokenizer
    that can't handle numpy action chunks.
    """
    # 1. Copy processing_action_tokenizer.py into the saved directory
    processor_src = Path(__file__).parent / "data" / "fast_tokenizer" / "processing_action_tokenizer.py"
    if processor_src.exists():
        shutil.copy2(processor_src, save_dir / "processing_action_tokenizer.py")
    else:
        print(f"  WARNING: {processor_src} not found, skipping processor file copy")

    # 2. Inject auto_map into preprocessor_config.json
    config_path = save_dir / "preprocessor_config.json"
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
        for i in range(N):
            ds_agent[i] = samples["image_agent"][i]
            ds_hand[i] = samples["image_hand"][i]

        # --- Proprioception ---
        out.create_dataset(
            "proprio",
            data=np.stack(samples["proprio"], axis=0),  # (N, 8)
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
        proprio_all = np.stack(samples["proprio"], axis=0)  # (N, 8)
        print(f"\n  Proprioception stats (8d):")
        labels = ["ee_pos_x", "ee_pos_y", "ee_pos_z",
                  "ee_quat_w", "ee_quat_x", "ee_quat_y", "ee_quat_z",
                  "gripper"]
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

        # Step 2: Action normalization stats
        action_low, action_high = compute_action_stats(f, demo_keys)

        # Step 3: Extract chunk-aligned samples (dual-view + proprio)
        samples = extract_chunks(
            f, demo_keys, args.image_key, args.hand_image_key,
            args.chunk_size, action_low, action_high,
        )

    if not samples["continuous_actions"]:
        sys.exit("ERROR: No chunks extracted. Check trajectory lengths vs chunk size.")

    # Step 4: FAST tokenization
    # Stack all action chunks into a single array for batch tokenization
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

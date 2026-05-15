"""diag_object_actions.py — Dump predicted vs GT action chunks for ONE libero_object task.

Diagnoses why libero_object rollouts return 0/200 success despite val CE ~3.2 (on par
with libero_spatial which gets 83% success). Captures:
  * Predicted FAST tokens, decoded normalized chunk, denormalized physical chunk
    at every chunk step of a single rollout.
  * Env state (EE position, gripper joints, reward) right before each chunk.
  * GT chunks from demo 0 of the same task (already physical units, anchor-relative,
    stored in the preprocessed HDF5).

Saves to a single .npz, plus prints a side-by-side comparison summary.

Usage (on the GPU server):
    python diag_object_actions.py \\
        --ckpt /Data/lyw/stable-wm/all4_sp_sigreg_seed3072/lewm_step_64000_object.ckpt \\
        --tokenizer /Data/lyw/fast_tokenizer_all4 \\
        --task-name pick_up_the_alphabet_soup_and_place_it_in_the_basket \\
        --suite libero_object --num-chunks 15 --out /Data/lyw/diag_object_actions.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from transformers import T5Tokenizer

# Reuse helpers from eval_libero / fast_utils.
from eval_libero import (
    _denormalize_gripper_aux,
    _read_osc_scales,
    build_model,
    find_processed_h5,
    load_action_stats,
    load_checkpoint,
    load_language_instruction,
    preprocess_obs,
    tokenize_language,
)
from fast_utils import denormalize_actions, fast_decode, load_fast_processor

POS_SCALE_FALLBACK = 0.05
ROT_SCALE_FALLBACK = 0.5


def gather_gt_chunks(
    h5_path: Path, demo_idx: int, n_chunks: int
) -> dict[str, np.ndarray]:
    """Pull rollout-aligned chunks from `demo_idx` in the preprocessed HDF5.

    `continuous_actions` is stored normalized to [-1, 1] (see preprocess_libero.py).
    Denormalize back to physical units using the per-task action_low/high attrs.

    Rollout diagnostics execute one H-step action chunk, then query the policy
    again. Compare against GT anchors at raw steps 0, H, 2H, ... rather than
    consecutive sliding-window anchors 0, 1, 2, ... .
    """
    with h5py.File(h5_path, "r") as f:
        demo_arr = f["demo_idx"][()]
        chunk_idx_arr = f["chunk_idx"][()]
        chunk_size = int(f.attrs["chunk_size"])
        selected: list[int] = []
        for anchor in range(0, n_chunks * chunk_size, chunk_size):
            matches = np.where((demo_arr == demo_idx) & (chunk_idx_arr == anchor))[0]
            if matches.size:
                selected.append(int(matches[0]))
        idx = np.array(selected, dtype=np.int64)
        if idx.size == 0:
            raise ValueError(f"No samples in {h5_path} with demo_idx=={demo_idx}")
        chunks_norm = f["continuous_actions"][idx]  # (n, H, 7) in [-1, 1]
        fast_tokens_list = [f["fast_tokens"][int(i)] for i in idx]
        fast_lengths = f["fast_length"][idx]
        chunk_idx = f["chunk_idx"][idx]
        action_low = np.array(f.attrs["action_low"])
        action_high = np.array(f.attrs["action_high"])
        proprio = f["proprio"][idx]  # (n, 9)

    # Denormalize to physical units (same formula eval_libero uses).
    chunks_phys = denormalize_actions(chunks_norm, action_low, action_high)
    # Pad fast_tokens to a 2D array for easier saving.
    max_len = max(len(t) for t in fast_tokens_list)
    fast_tokens_arr = np.full((len(fast_tokens_list), max_len), -1, dtype=np.int64)
    for i, toks in enumerate(fast_tokens_list):
        fast_tokens_arr[i, : len(toks)] = np.asarray(toks, dtype=np.int64)
    return {
        "gt_chunks_norm": chunks_norm.astype(np.float32),
        "gt_chunks_phys": chunks_phys.astype(np.float32),
        "gt_fast_tokens": fast_tokens_arr,
        "gt_fast_lengths": fast_lengths.astype(np.int32),
        "gt_chunk_idx": chunk_idx.astype(np.int32),
        "gt_proprio": proprio.astype(np.float32),
        "action_low": action_low.astype(np.float32),
        "action_high": action_high.astype(np.float32),
        "chunk_size": chunk_size,
    }


def run_diag_rollout(
    model,
    processor,
    t5_tokenizer,
    env,
    init_state,
    action_low: np.ndarray,
    action_high: np.ndarray,
    chunk_size: int,
    action_dim: int,
    language_instruction: str,
    *,
    num_chunks: int,
    max_steps_per_chunk: int,
    img_size: int = 224,
    max_lang_tokens: int = 25,
    device: torch.device = torch.device("cuda"),
    temperature: float = 0.0,
) -> dict[str, np.ndarray]:
    """Run a single rollout, capturing per-chunk model output + env state.

    Differs from eval_libero.evaluate_task by:
      * Single episode, deterministic (temperature=0).
      * Captures every intermediate tensor for inspection.
      * Caps at `num_chunks` chunks regardless of success.
    """
    env.reset()
    obs = env.set_init_state(init_state)
    pos_scale, rot_scale = _read_osc_scales(env)

    lang_ids, lang_mask = tokenize_language(
        language_instruction,
        t5_tokenizer,
        max_lang_tokens,
        device,
    )

    pred_tokens_list: list[np.ndarray] = []
    pred_lengths_list: list[int] = []
    pred_norm_chunks: list[np.ndarray] = []
    pred_phys_chunks: list[np.ndarray] = []
    pred_fast_phys_chunks: list[np.ndarray] = []
    pred_aux_grip_chunks: list[np.ndarray] = []
    anchor_pos_list: list[np.ndarray] = []
    anchor_quat_list: list[np.ndarray] = []
    rewards_at_chunk: list[float] = []
    gripper_joints_at_chunk: list[np.ndarray] = []
    inner_step_rewards: list[list[float]] = []
    inner_step_gripper: list[list[np.ndarray]] = []

    for chunk_i in range(num_chunks):
        # Snapshot anchor pose for this chunk (matches eval_libero closed-loop).
        anchor_pos = obs["robot0_eef_pos"].copy()
        anchor_quat = obs["robot0_eef_quat"].copy()
        anchor_pos_list.append(anchor_pos.astype(np.float32))
        anchor_quat_list.append(anchor_quat.astype(np.float32))
        gripper_joints_at_chunk.append(
            obs["robot0_gripper_qpos"].astype(np.float32).copy()
        )

        # Encode + generate.
        pixels_agent, pixels_hand, proprio_t = preprocess_obs(obs, img_size, device)
        z_agent, z_hand, lang_embeds, lang_lengths = model.encode(
            pixels_agent,
            pixels_hand,
            lang_ids,
            lang_mask,
        )
        tokens, lengths = model.predict_actions(
            z_agent,
            z_hand,
            proprio_t,
            lang_embeds,
            lang_lengths,
            temperature=temperature,
        )
        tokens_np = tokens.cpu().numpy()
        lengths_np = lengths.cpu().numpy()
        pred_tokens_list.append(tokens_np[0].astype(np.int64))
        pred_lengths_list.append(int(lengths_np[0]))

        actions_norm = fast_decode(
            tokens,
            lengths,
            processor,
            time_horizon=chunk_size,
            action_dim=action_dim,
        )  # (B, H, D) in [-1, 1]
        actions_phys = denormalize_actions(actions_norm, action_low, action_high)
        pred_fast_phys_chunks.append(actions_phys[0].astype(np.float32).copy())

        if getattr(model.predictor, "use_gripper_aux", False):
            pred_grip = model.predict_gripper_aux(
                z_agent,
                z_hand,
                proprio_t,
                lang_embeds,
                lang_lengths,
            )
            aux_grip = _denormalize_gripper_aux(
                pred_grip[0].detach().cpu().numpy(),
                action_low,
                action_high,
            )
            actions_phys[0, :, 6] = aux_grip
            pred_aux_grip_chunks.append(aux_grip.astype(np.float32))

        pred_norm_chunks.append(actions_norm[0].astype(np.float32))
        pred_phys_chunks.append(actions_phys[0].astype(np.float32))

        # Execute the chunk closed-loop, mirroring eval_libero._execute_chunk_closed_loop
        # but capture per-step state.
        from scipy.spatial.transform import Rotation as R

        R_anchor = R.from_quat(anchor_quat)
        chunk_phys = actions_phys[0]
        inner_rewards: list[float] = []
        inner_grip: list[np.ndarray] = []
        reward = 0.0
        done = False
        for k in range(min(chunk_size, max_steps_per_chunk)):
            target_pos = anchor_pos + chunk_phys[k, 0:3]
            target_R = R.from_rotvec(chunk_phys[k, 3:6]) * R_anchor
            cur_pos = obs["robot0_eef_pos"]
            cur_R = R.from_quat(obs["robot0_eef_quat"])
            step_dpos = target_pos - cur_pos
            step_drot = (target_R * cur_R.inv()).as_rotvec()
            gripper_cmd = chunk_phys[k, 6]
            action_input = np.concatenate(
                [
                    np.clip(step_dpos / pos_scale, -1.0, 1.0),
                    np.clip(step_drot / rot_scale, -1.0, 1.0),
                    [gripper_cmd],
                ]
            )
            obs, reward, done, _ = env.step(action_input)
            inner_rewards.append(float(reward))
            inner_grip.append(obs["robot0_gripper_qpos"].astype(np.float32).copy())
            if done:
                break
        rewards_at_chunk.append(float(reward))
        inner_step_rewards.append(inner_rewards)
        inner_step_gripper.append(inner_grip)
        if done:
            print(
                f"  Chunk {chunk_i}: env done (reward={reward:.3f}, success={reward > 0})"
            )
            break

    # Pad token list for npz storage.
    max_t = max(len(t) for t in pred_tokens_list)
    tok_arr = np.full((len(pred_tokens_list), max_t), -1, dtype=np.int64)
    for i, t in enumerate(pred_tokens_list):
        tok_arr[i, : len(t)] = t

    out = {
        "pred_tokens": tok_arr,
        "pred_token_lengths": np.array(pred_lengths_list, dtype=np.int32),
        "pred_norm_chunks": np.stack(pred_norm_chunks),  # (n, H, 7)
        "pred_phys_chunks": np.stack(pred_phys_chunks),  # (n, H, 7)
        "pred_fast_phys_chunks": np.stack(pred_fast_phys_chunks),  # before aux override
        "anchor_pos": np.stack(anchor_pos_list),  # (n, 3)
        "anchor_quat": np.stack(anchor_quat_list),  # (n, 4)
        "chunk_start_gripper": np.stack(gripper_joints_at_chunk),  # (n, 2)
        "chunk_end_reward": np.array(rewards_at_chunk, dtype=np.float32),
        "inner_step_rewards": np.array(
            [
                np.pad(r, (0, chunk_size - len(r)), constant_values=np.nan)
                for r in inner_step_rewards
            ],
            dtype=np.float32,
        ),
    }
    if pred_aux_grip_chunks:
        out["pred_aux_grip_chunks"] = np.stack(pred_aux_grip_chunks)
    return out


def summarize_comparison(pred: dict, gt: dict, n_show: int = 5) -> None:
    """Print a side-by-side summary of predicted vs GT trajectories."""
    print()
    print("=" * 70)
    print("COMPARISON: predicted (rollout) vs GT (demo 0)")
    print("=" * 70)

    n_pred = pred["pred_phys_chunks"].shape[0]
    n_gt = gt["gt_chunks_phys"].shape[0]
    n_cmp = min(n_pred, n_gt, n_show)
    print(f"Comparing first {n_cmp} chunks (pred has {n_pred}, gt has {n_gt})")

    print()
    print(
        "--- Per-chunk physical-unit MEAN ABS magnitudes (dx, dy, dz, drx, dry, drz, grip) ---"
    )
    print(f"{'idx':>3} | {'PRED':<55} | {'GT':<55}")
    for i in range(n_cmp):
        p = np.abs(pred["pred_phys_chunks"][i]).mean(axis=0)  # over H steps
        g = np.abs(gt["gt_chunks_phys"][i]).mean(axis=0)
        p_str = "[" + ", ".join(f"{x:5.3f}" for x in p) + "]"
        g_str = "[" + ", ".join(f"{x:5.3f}" for x in g) + "]"
        print(f"{i:>3} | {p_str} | {g_str}")

    print()
    print("--- GRIPPER COMMAND trace (chunk[*, 6]) — first 3 chunks, all H steps ---")
    for i in range(min(3, n_cmp)):
        p = pred["pred_phys_chunks"][i, :, 6]
        g = gt["gt_chunks_phys"][i, :, 6]
        print(f"  chunk {i} PRED grip:  [{' '.join(f'{x:+5.2f}' for x in p)}]")
        if "pred_aux_grip_chunks" in pred:
            f = pred["pred_fast_phys_chunks"][i, :, 6]
            print(f"  chunk {i} FAST grip:  [{' '.join(f'{x:+5.2f}' for x in f)}]")
        print(f"  chunk {i}  GT  grip:  [{' '.join(f'{x:+5.2f}' for x in g)}]")

    print()
    print("--- Token length per chunk ---")
    for i in range(n_cmp):
        pl = int(pred["pred_token_lengths"][i])
        gl = int(gt["gt_fast_lengths"][i])
        print(f"  chunk {i}: pred_tok_len={pl}, gt_tok_len={gl}")

    print()
    print("--- Reward at end of each chunk (predicted rollout) ---")
    print(f"  {pred['chunk_end_reward'].round(3).tolist()}")

    print()
    print("--- Gripper joint qpos at start of each chunk (predicted rollout) ---")
    print(f"  Episode start: {pred['chunk_start_gripper'][0].round(3).tolist()}")
    print(
        f"  Mid          : {pred['chunk_start_gripper'][n_pred // 2].round(3).tolist()}"
    )
    print(f"  Final chunk  : {pred['chunk_start_gripper'][-1].round(3).tolist()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--processed-dir", default="/Data/lyw/libero_processed_v5/libero_object"
    )
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument(
        "--task-name",
        required=True,
        help="Substring match against the LIBERO benchmark task names.",
    )
    parser.add_argument("--num-chunks", type=int, default=15)
    parser.add_argument("--max-steps-per-chunk", type=int, default=20)
    parser.add_argument(
        "--demo-idx", type=int, default=0, help="GT demo to compare against"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="/Data/lyw/diag_object_actions.npz")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading ckpt: {args.ckpt}")
    if args.ckpt.endswith("_object.ckpt"):
        model = torch.load(args.ckpt, map_location=device, weights_only=False)
    else:
        model = build_model(device, use_language=True)
        load_checkpoint(model, args.ckpt, device)
    model.eval()

    print(f"Loading FAST tokenizer: {args.tokenizer}")
    processor = load_fast_processor(args.tokenizer)
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Look up env + h5 by task name.
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    target_tid = None
    for tid in range(task_suite.n_tasks):
        if args.task_name in task_suite.get_task(tid).name:
            target_tid = tid
            break
    if target_tid is None:
        print(f"ERROR: task '{args.task_name}' not in {args.suite}", file=sys.stderr)
        return 1
    task = task_suite.get_task(target_tid)
    print(f"Task {target_tid}: {task.name}")

    h5_path = find_processed_h5(args.processed_dir, task.name)
    if h5_path is None:
        print(
            f"ERROR: no h5 for task '{task.name}' under {args.processed_dir}",
            file=sys.stderr,
        )
        return 1
    print(f"  h5: {h5_path}")
    action_low, action_high, chunk_size, action_dim = load_action_stats(h5_path)
    language_instruction = load_language_instruction(h5_path)
    print(f"  language: {language_instruction!r}")
    print(f"  action_low : {action_low.round(3).tolist()}")
    print(f"  action_high: {action_high.round(3).tolist()}")

    # Build env.
    bddl_path = os.path.join(get_libero_path("bddl_files"), args.suite, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path, camera_heights=128, camera_widths=128
    )
    init_states = task_suite.get_task_init_states(target_tid)

    try:
        pred = run_diag_rollout(
            model,
            processor,
            t5_tokenizer,
            env,
            init_states[0],
            action_low,
            action_high,
            chunk_size,
            action_dim,
            language_instruction,
            num_chunks=args.num_chunks,
            max_steps_per_chunk=args.max_steps_per_chunk,
            device=device,
        )
    finally:
        env.close()

    print()
    print(f"Loading GT chunks from {h5_path} (demo {args.demo_idx}) ...")
    gt = gather_gt_chunks(h5_path, args.demo_idx, args.num_chunks)
    print(
        f"  GT has {gt['gt_chunks_phys'].shape[0]} chunks at chunk_idx="
        f"{gt['gt_chunk_idx'].tolist()[:6]}..."
    )

    summarize_comparison(pred, gt, n_show=min(args.num_chunks, 8))

    # Save .npz for offline plotting / scp'ing back.
    np.savez(
        args.out,
        **pred,
        **gt,
        task_name=task.name,
        language_instruction=language_instruction,
        ckpt=args.ckpt,
    )
    print()
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

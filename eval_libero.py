"""LIBERO evaluation: VLA baseline — autoregressive action generation → environment execution.

Loads a trained checkpoint, runs the model in LIBERO environments, and
measures task success rate. The model takes dual-view images, proprioception,
and language instruction as input.

Usage:
    # Single task
    python eval_libero.py \
        --checkpoint /Data/lyw/stable-wm/lewm_weights.ckpt \
        --tokenizer /Data/lyw/fast_tokenizer \
        --processed-dir /Data/lyw/libero_processed/libero_90 \
        --suite libero_spatial --task-id 0 \
        --num-episodes 20

    # All tasks in a suite
    python eval_libero.py \
        --checkpoint /Data/lyw/stable-wm/lewm_weights.ckpt \
        --tokenizer /Data/lyw/fast_tokenizer \
        --processed-dir /Data/lyw/libero_processed/libero_90 \
        --suite libero_spatial \
        --num-episodes 20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

try:
    import imageio.v3 as iio
except ImportError:
    iio = None

os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from fast_utils import denormalize_actions, fast_decode, load_fast_processor
from libero_dataset import _preprocess_image
from preprocess_libero import normalize_proprio


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(device: torch.device, use_language: bool = True) -> torch.nn.Module:
    """Build the JEPA model with the same architecture as train.py.

    Args:
        device: target device.
        use_language: must match the training-time setting. When False, the
            T5 encoder and language projection are omitted, and the predictor
            runs on a vision+proprio-only prefix.
    """
    import stable_pretraining as spt

    from jepa import JEPA
    from module import ARPredictor, MLP

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = 192

    predictor = ARPredictor(
        embed_dim=embed_dim, max_action_tokens=80, max_lang_tokens=25,
        proprio_dim=8,
        depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1, emb_dropout=0.0,
    )
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)

    if use_language:
        # T5-small (frozen)
        lang_encoder = T5EncoderModel.from_pretrained("t5-small")
        lang_encoder.eval()
        for p in lang_encoder.parameters():
            p.requires_grad_(False)
        lang_proj = torch.nn.Linear(lang_encoder.config.d_model, embed_dim)
    else:
        print("[Ablation] use_language=False — evaluating with vision+proprio only.")
        lang_encoder = None
        lang_proj = None

    model = JEPA(
        encoder=encoder, predictor=predictor, projector=projector,
        lang_encoder=lang_encoder, lang_proj=lang_proj,
    )
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """Load Lightning weights checkpoint into JEPA model.

    spt.Module stores the JEPA as self.model, so checkpoint keys are
    prefixed with "model." (e.g. "model.encoder.xxx"). We strip that prefix.
    T5 encoder is loaded separately (from pretrained), so we skip those keys.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt["state_dict"]

    has_lang_module = getattr(model, "lang_encoder", None) is not None

    # Strip "model." prefix from spt.Module wrapper
    model_sd = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k.removeprefix("model.")
            # Skip T5 encoder keys (loaded from pretrained)
            if new_key.startswith("lang_encoder."):
                continue
            # In no-language mode the model has no lang_proj — skip any residual keys
            if not has_lang_module and new_key.startswith("lang_proj."):
                continue
            model_sd[new_key] = v

    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    # T5 encoder keys are expected to be missing (loaded from pretrained)
    real_missing = [k for k in missing if not k.startswith("lang_encoder.")]
    if real_missing:
        raise RuntimeError(
            f"Checkpoint missing {len(real_missing)} non-T5 keys — model architecture "
            f"likely does not match checkpoint. Missing: {real_missing[:10]}"
        )
    if unexpected:
        raise RuntimeError(
            f"Checkpoint has {len(unexpected)} unexpected keys — model architecture "
            f"likely does not match checkpoint. Unexpected: {unexpected[:10]}"
        )
    print(f"Loaded checkpoint from {ckpt_path} ({len(model_sd)} keys)")


# ---------------------------------------------------------------------------
# Preprocessed data lookup
# ---------------------------------------------------------------------------

def find_processed_h5(processed_dir: str, task_name: str) -> Path | None:
    """Find the preprocessed H5 file matching a LIBERO task name."""
    processed_dir = Path(processed_dir)
    task_key = task_name.lower().replace(" ", "_")

    for ext in ("*.h5", "*.hdf5"):
        for h5_path in sorted(processed_dir.glob(ext)):
            if task_key in h5_path.stem.lower():
                return h5_path

    # Fallback: check source_file attr
    for ext in ("*.h5", "*.hdf5"):
        for h5_path in sorted(processed_dir.glob(ext)):
            with h5py.File(h5_path, "r") as f:
                source = f.attrs.get("source_file", "")
                if task_key in str(source).lower():
                    return h5_path

    return None


def load_action_stats(h5_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Read action normalization stats and chunk_size from preprocessed HDF5."""
    with h5py.File(h5_path, "r") as f:
        action_low = np.array(f.attrs["action_low"])
        action_high = np.array(f.attrs["action_high"])
        chunk_size = int(f.attrs["chunk_size"])
        action_dim = int(f.attrs["action_dim"])
    return action_low, action_high, chunk_size, action_dim


def load_language_instruction(h5_path: Path) -> str:
    """Read language instruction from preprocessed HDF5."""
    with h5py.File(h5_path, "r") as f:
        return f.attrs.get("language_instruction", "")


# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------

def preprocess_obs(
    obs: dict,
    img_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract dual-view images and proprio from LIBERO env observation.

    Args:
        obs: LIBERO observation dict.
        img_size: target image size (224).
        device: target device.

    Returns:
        pixels_agent: (1, 3, img_size, img_size) agentview tensor.
        pixels_hand: (1, 3, img_size, img_size) hand tensor.
        proprio: (1, 8) proprioceptive state tensor.
    """
    # Agentview image
    img_agent = _preprocess_image(obs["agentview_image"], img_size)
    pixels_agent = img_agent.unsqueeze(0).to(device)

    # Eye-in-hand image
    img_hand = _preprocess_image(obs["robot0_eye_in_hand_image"], img_size)
    pixels_hand = img_hand.unsqueeze(0).to(device)

    # Proprioception: ee_pos(3) + ee_quat(4) + gripper(1) = 8d
    ee_pos = obs["robot0_eef_pos"]          # (3,)
    ee_quat = obs["robot0_eef_quat"]        # (4,) quaternion
    grip_2d = obs["robot0_gripper_qpos"]    # (2,) two finger widths
    gripper = np.array([np.mean(np.abs(grip_2d))])  # (1,) mean of 2 fingers
    proprio_raw = np.concatenate([ee_pos, ee_quat, gripper])  # (8,)
    proprio_np = normalize_proprio(proprio_raw)  # re-normalize quaternion
    proprio = torch.from_numpy(proprio_np).float().unsqueeze(0).to(device)

    return pixels_agent, pixels_hand, proprio


def tokenize_language(
    instruction: str,
    t5_tokenizer: T5Tokenizer,
    max_lang_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize language instruction for model input.

    Returns:
        lang_ids: (1, max_lang_tokens) token IDs.
        lang_mask: (1, max_lang_tokens) attention mask.
    """
    tok_out = t5_tokenizer(
        instruction,
        max_length=max_lang_tokens,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tok_out["input_ids"].to(device), tok_out["attention_mask"].to(device)


# ---------------------------------------------------------------------------
# Single-task evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_task(
    model: torch.nn.Module,
    processor,
    t5_tokenizer: T5Tokenizer | None,
    env: OffScreenRenderEnv,
    init_states: list,
    action_low: np.ndarray,
    action_high: np.ndarray,
    chunk_size: int,
    action_dim: int,
    language_instruction: str,
    *,
    num_episodes: int = 20,
    max_steps: int = 300,
    img_size: int = 224,
    max_lang_tokens: int = 25,
    device: torch.device = torch.device("cuda"),
    temperature: float = 0.0,
    save_videos: bool = False,
    video_dir: str | None = None,
    task_name: str = "",
    max_video_episodes: int = 3,
    use_language: bool = True,
) -> tuple[int, int]:
    """Run episodes and count successes."""
    successes = 0
    max_chunks = max_steps // chunk_size

    # Pre-tokenize language instruction (same for all episodes); skipped in no-language mode.
    if use_language:
        assert t5_tokenizer is not None, "t5_tokenizer required when use_language=True"
        lang_ids, lang_mask = tokenize_language(
            language_instruction, t5_tokenizer, max_lang_tokens, device,
        )
    else:
        lang_ids, lang_mask = None, None

    for ep in range(num_episodes):
        # Reset with deterministic initial state
        env.reset()
        init_idx = ep % len(init_states)
        obs = env.set_init_state(init_states[init_idx])

        # Collect frames for video (first N episodes only)
        recording = save_videos and ep < max_video_episodes
        frames: list[np.ndarray] = []
        if recording:
            frames.append(obs["agentview_image"])

        reward = 0
        done = False
        for _ in range(max_chunks):
            # Preprocess observation (dual-view + proprio)
            pixels_agent, pixels_hand, proprio = preprocess_obs(obs, img_size, device)

            # Encode visual + language
            z_agent, z_hand, lang_embeds, lang_lengths = model.encode(
                pixels_agent, pixels_hand, lang_ids, lang_mask,
            )

            # Generate FAST action tokens
            tokens, lengths = model.predict_actions(
                z_agent, z_hand, proprio, lang_embeds, lang_lengths,
                temperature=temperature,
            )

            # Decode to continuous actions
            actions_norm = fast_decode(tokens, lengths, processor,
                                       time_horizon=chunk_size, action_dim=action_dim)
            actions_raw = denormalize_actions(actions_norm, action_low, action_high)

            # Execute chunk (H steps)
            for h in range(chunk_size):
                obs, reward, done, info_env = env.step(actions_raw[0, h])
                if recording:
                    frames.append(obs["agentview_image"])
                if done:
                    break
            if done:
                break

        success = reward > 0
        if success:
            successes += 1
        print(f"  Episode {ep}: {'SUCCESS' if success else 'FAIL'}")

        # Save video
        if recording and frames and video_dir is not None:
            _save_video(frames, video_dir, task_name, ep, success)

    return successes, num_episodes


def _save_video(
    frames: list[np.ndarray],
    video_dir: str,
    task_name: str,
    episode_id: int,
    success: bool,
) -> None:
    """Save collected frames as an mp4 video."""
    if iio is None:
        print("    [WARN] imageio not installed, skipping video save")
        return

    os.makedirs(video_dir, exist_ok=True)
    # Sanitize task name for filename
    safe_name = task_name.replace(" ", "_").replace("/", "_")
    tag = "success" if success else "fail"
    filename = f"{safe_name}_ep{episode_id}_{tag}.mp4"
    filepath = os.path.join(video_dir, filename)

    # Stack frames: LIBERO obs images are (H, W, 3) uint8
    video = np.stack(frames, axis=0)
    iio.imwrite(filepath, video, fps=20, codec="h264")
    print(f"    Saved video: {filepath} ({len(frames)} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LIBERO VLA baseline evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to _weights.ckpt")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to saved FAST tokenizer directory")
    parser.add_argument("--processed-dir", type=str, required=True,
                        help="Directory with preprocessed H5 files (one per task)")
    parser.add_argument("--suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Single task index (0-9). Omit to run all tasks in suite")
    parser.add_argument("--num-episodes", type=int, default=20, help="Episodes per task")
    parser.add_argument("--max-steps", type=int, default=300, help="Max raw steps per episode")
    parser.add_argument("--camera-size", type=int, default=128,
                        help="LIBERO camera resolution (env render size). "
                             "Must match the resolution of images stored in preprocessed "
                             "training HDF5 (LIBERO default: 128).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.0, help="0=greedy, >0=sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-videos", action="store_true", help="Save rollout videos for first 3 episodes per task")
    parser.add_argument("--video-dir", type=str, default="/Data/lyw/eval_videos", help="Directory to save videos")
    parser.add_argument("--max-video-episodes", type=int, default=3, help="Max episodes per task to record")
    parser.add_argument("--no-language", action="store_true",
                        help="Ablation: evaluate a model trained without the language instruction. "
                             "Must match the checkpoint's training-time use_language setting.")
    args = parser.parse_args()

    if args.save_videos and iio is None:
        raise ImportError("imageio required for --save-videos: pip install imageio imageio-ffmpeg")

    use_language = not args.no_language

    device = torch.device(args.device)
    np.random.seed(args.seed)

    # Load model
    print("Loading model...")
    model = build_model(device, use_language=use_language)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # Load FAST tokenizer
    print(f"Loading FAST tokenizer from {args.tokenizer}")
    processor = load_fast_processor(args.tokenizer)

    # Load T5 tokenizer for language (skipped in no-language mode)
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small") if use_language else None

    # Get task suite
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    num_tasks = task_suite.n_tasks

    if args.task_id is not None:
        task_ids = [args.task_id]
    else:
        task_ids = list(range(num_tasks))

    # Run evaluation
    total_successes = 0
    total_episodes = 0
    results = {}

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        task_name = task.name
        print(f"\n{'='*60}")
        print(f"Task {task_id}: {task_name}")
        print(f"{'='*60}")

        # Find preprocessed H5 for this task
        h5_path = find_processed_h5(args.processed_dir, task_name)
        if h5_path is None:
            print(f"  WARNING: No preprocessed H5 found for '{task_name}', skipping.")
            continue
        print(f"  Preprocessed data: {h5_path}")

        action_low, action_high, chunk_size, action_dim = load_action_stats(h5_path)
        language_instruction = load_language_instruction(h5_path)
        print(f"  Language: '{language_instruction}'")

        # Create environment — task.bddl_file is just a filename, need full path
        bddl_path = os.path.join(
            get_libero_path("bddl_files"), args.suite, task.bddl_file,
        )
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=args.camera_size,
            camera_widths=args.camera_size,
        )
        init_states = task_suite.get_task_init_states(task_id)

        try:
            successes, n_eps = evaluate_task(
                model, processor, t5_tokenizer, env, init_states,
                action_low, action_high, chunk_size, action_dim,
                language_instruction,
                num_episodes=args.num_episodes,
                max_steps=args.max_steps,
                device=device,
                temperature=args.temperature,
                save_videos=args.save_videos,
                video_dir=args.video_dir,
                task_name=task_name,
                max_video_episodes=args.max_video_episodes,
                use_language=use_language,
            )
        finally:
            env.close()

        rate = successes / n_eps * 100
        results[task_id] = {"name": task_name, "success": successes, "total": n_eps, "rate": rate}
        total_successes += successes
        total_episodes += n_eps

        print(f"  Result: {successes}/{n_eps} ({rate:.1f}%)")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {args.suite}")
    print(f"{'='*60}")
    for tid, r in results.items():
        print(f"  Task {tid:2d}: {r['success']:2d}/{r['total']} ({r['rate']:5.1f}%) — {r['name']}")
    if total_episodes > 0:
        overall = total_successes / total_episodes * 100
        print(f"\n  Overall: {total_successes}/{total_episodes} ({overall:.1f}%)")


if __name__ == "__main__":
    main()

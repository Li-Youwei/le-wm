"""LIBERO evaluation: VLA baseline — autoregressive action generation → environment execution.

Loads a trained checkpoint, runs the model in LIBERO environments, and
measures task success rate. The model takes dual-view images, proprioception,
and language instruction as input.

Usage:
    # Single task
    python eval_libero.py \
        --checkpoint /data/lyw/stable-wm/lewm_weights.ckpt \
        --tokenizer /data/lyw/fast_tokenizer \
        --processed-dir /data/lyw/libero_processed/libero_90 \
        --suite libero_spatial --task-id 0 \
        --num-episodes 20

    # All tasks in a suite
    python eval_libero.py \
        --checkpoint /data/lyw/stable-wm/lewm_weights.ckpt \
        --tokenizer /data/lyw/fast_tokenizer \
        --processed-dir /data/lyw/libero_processed/libero_90 \
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
from scipy.spatial.transform import Rotation as R
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


# Per-suite policy horizon (max env steps per episode, EXCLUDING the no-op
# warmup). These are the OpenVLA / RynnVLA-002 LIBERO eval values, verbatim
# from RynnVLA-002's run_libero_eval.py (each comment = "longest training demo
# has N steps"). Used as the default when --max-steps is not given explicitly,
# so a standalone single-suite eval automatically uses the correct horizon.
LIBERO_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def build_model(device: torch.device) -> torch.nn.Module:
    """Build the JEPA model matching train.py's DEFAULT architecture.

    The default visual backbone is now a FROZEN DINOv2-base (see
    config/train/lewm.yaml). This ``_weights.ckpt`` path therefore reconstructs
    DINOv2-base via build_visual_encoder so the encoder.* keys line up. SP /
    SIGReg / MoT / V17 and the legacy random-init ViT-Tiny baseline are
    evaluated via the pickled ``_object.ckpt`` instead.

    Args:
        device: target device.
    """
    import stable_pretraining as spt
    from omegaconf import OmegaConf

    from jepa import JEPA
    from module import ARPredictor, MLP
    from vision_backbone import build_visual_encoder

    # Mirror config/train/lewm.yaml's default backbone: a FROZEN DINOv2-base
    # loaded via AutoModel. build_visual_encoder reproduces the exact training
    # construction, so the encoder.* keys in a _weights.ckpt line up.
    # `local_files_only` is intentionally omitted so build_visual_encoder falls
    # back to _offline_default() (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE): a
    # standalone eval on an un-cached machine can then still fetch DINOv2-base
    # (export HF_HUB_OFFLINE=1 to force offline once it is cached).
    enc_cfg = OmegaConf.create(
        {
            "encoder_scale": "tiny",
            "patch_size": 14,
            "img_size": 224,
            "vision_encoder": {
                "source": "hf",
                "model_name_or_path": "facebook/dinov2-base",
                "freeze": True,
                "trust_remote_code": False,
            },
        }
    )
    encoder, hidden_dim, freeze_encoder = build_visual_encoder(enc_cfg, spt)
    embed_dim = 384

    predictor = ARPredictor(
        embed_dim=embed_dim,
        max_action_tokens=80,
        max_lang_tokens=25,
        proprio_dim=9,
        # depth/dropout mirror config/train/lewm.yaml's current default (depth
        # 12, dropout 0.15). depth sets the block count, which MUST match the
        # checkpoint for the state_dict to load; dropout is a no-op at eval.
        # This build_model path reconstructs the simple CLS-only architecture
        # only — pool_grid (multi-token visual), MoT, and state-prediction are
        # NOT rebuilt here, so any pool_grid>0 / MoT / SP / non-DINOv2-base
        # checkpoint must be eval'd via its _object.ckpt instead.
        # residual_scale_init is omitted: it only affects init, which the
        # loaded weights overwrite, so it has no effect on eval.
        depth=12,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.15,
        emb_dropout=0.0,
    )
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.LayerNorm,
    )

    # T5-small (frozen)
    lang_encoder = T5EncoderModel.from_pretrained("t5-small")
    lang_encoder.eval()
    for p in lang_encoder.parameters():
        p.requires_grad_(False)
    lang_proj = torch.nn.Linear(lang_encoder.config.d_model, embed_dim)

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        projector=projector,
        lang_encoder=lang_encoder,
        lang_proj=lang_proj,
        freeze_encoder=freeze_encoder,
    )
    return model.to(device)


def load_checkpoint(
    model: torch.nn.Module, ckpt_path: str, device: torch.device
) -> None:
    """Load Lightning weights checkpoint into JEPA model.

    spt.Module stores the JEPA as self.model, so checkpoint keys are
    prefixed with "model." (e.g. "model.encoder.xxx"). We strip that prefix.
    T5 encoder is loaded separately (from pretrained), so we skip those keys.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt["state_dict"]

    # Reject any non-baseline-architecture checkpoint loaded via the
    # _weights.ckpt path — build_model() constructs the baseline architecture
    # (no state_pred_head_*, no state_query_embeddings, LayerNorm projector),
    # so loading a checkpoint with extra keys would either fail with a
    # confusing torch shape-mismatch error or (worse) silently pass with
    # strict=False and skip critical parameters.
    #
    # Two known deviations both require _object.ckpt:
    #   1. SP-trained: state_pred_head_* / state_query_embeddings keys.
    #   2. BatchNorm-projector: projector.net.1.running_mean buffer (BN1d
    #      tracks running stats; LayerNorm has no such buffer). This catches
    #      the legitimate "SIGReg-only, no SP" ablation as well, where
    #      projector.norm_type='batch' but no SP heads exist.
    sp_keys = [
        k for k in state_dict if "state_pred_head" in k or "state_query_embeddings" in k
    ]
    bn_keys = [k for k in state_dict if "projector.net.1.running_mean" in k]
    if (sp_keys or bn_keys) and not ckpt_path.endswith("_object.ckpt"):
        reasons = []
        if sp_keys:
            reasons.append(
                f"state-prediction keys ({sp_keys[:2]}{'...' if len(sp_keys) > 2 else ''})"
            )
        if bn_keys:
            reasons.append(f"BatchNorm projector running stats ({bn_keys[:1]})")
        raise ValueError(
            f"Checkpoint '{ckpt_path}' has {' and '.join(reasons)} but is not "
            "an _object.ckpt. Non-baseline architectures (SP-trained or "
            "BatchNorm-projector / SIGReg-trained) "
            "must be evaluated via the per-epoch object checkpoint produced "
            "by ModelObjectCallBack — pass "
            "--checkpoint .../lewm_*_object.ckpt instead."
        )

    # Strip "model." prefix from spt.Module wrapper
    model_sd = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k.removeprefix("model.")
            # Skip T5 encoder keys (loaded from pretrained)
            if new_key.startswith("lang_encoder."):
                continue
            model_sd[new_key] = v

    # Backbone-mismatch guard: build_model() constructs the default DINOv2-base
    # (projector input 768). A legacy ViT-Tiny _weights.ckpt has projector input
    # 192, which would otherwise raise a cryptic torch size-mismatch inside
    # load_state_dict. Surface an actionable message first.
    proj_key = "projector.net.0.weight"
    if proj_key in model_sd and hasattr(model.projector, "net"):
        ckpt_in = model_sd[proj_key].shape[1]
        model_in = model.projector.net[0].weight.shape[1]
        if ckpt_in != model_in:
            raise ValueError(
                f"Projector input dim mismatch: checkpoint={ckpt_in}, model={model_in}. "
                "This _weights.ckpt was trained with a different visual backbone "
                "(legacy ViT-Tiny=192 vs default DINOv2-base=768). Evaluate it via the "
                "matching _object.ckpt, or rebuild build_model() with that backbone."
            )

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

    Must match preprocess_libero.py's 9D proprio layout exactly:
        [ee_pos(3), xyzw_quat(4), gripper_raw(2)]

    The gripper dims are the two raw finger positions (obs["robot0_gripper_qpos"]),
    used directly without mean/abs — averaging collapses the signal because the
    two fingers are symmetric around 0.

    Args:
        obs: LIBERO observation dict.
        img_size: target image size (224).
        device: target device.

    Returns:
        pixels_agent: (1, 3, img_size, img_size) agentview tensor.
        pixels_hand: (1, 3, img_size, img_size) hand tensor.
        proprio: (1, 9) proprioceptive state tensor.
    """
    # Agentview image
    img_agent = _preprocess_image(obs["agentview_image"], img_size)
    pixels_agent = img_agent.unsqueeze(0).to(device)

    # Eye-in-hand image
    img_hand = _preprocess_image(obs["robot0_eye_in_hand_image"], img_size)
    pixels_hand = img_hand.unsqueeze(0).to(device)

    # 9D proprio — no averaging / no abs on the gripper.
    ee_pos = obs["robot0_eef_pos"]  # (3,)
    ee_quat = obs["robot0_eef_quat"]  # (4,) xyzw (robosuite default)
    grip_2d = obs["robot0_gripper_qpos"]  # (2,) raw two finger joint positions
    proprio_raw = np.concatenate([ee_pos, ee_quat, grip_2d])  # (9,)
    proprio_np = normalize_proprio(proprio_raw)  # re-normalize quaternion at [3:7]
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


def _read_osc_scales(env: OffScreenRenderEnv) -> tuple[float, float]:
    """Read robosuite OSC_POSE scales from the running controller.

    The anchor-relative action chunks are stored in PHYSICAL units (meters for
    position, radians for rotation). To convert a physical step-delta into the
    [-1, 1] input range that `env.step` expects, we divide by these scales:

        output_max[0:3] = pos_scale (typically 0.05 m)
        output_max[3:6] = rot_scale (typically 0.5 rad)

    We read them at runtime instead of hardcoding so the code automatically
    adapts if LIBERO / robosuite changes the defaults. Asserts uniform scales
    for pos/rot (the 3 translation axes and 3 rotation axes must all match) and
    input range [-1, 1].
    """
    ctrl = env.env.robots[0].controller
    output_max = np.asarray(ctrl.output_max, dtype=np.float64)
    assert output_max.shape == (6,), f"unexpected output_max shape: {output_max.shape}"
    pos_scale = float(output_max[0])
    rot_scale = float(output_max[3])
    assert np.allclose(output_max[:3], pos_scale), (
        f"OSC pos scales not uniform: {output_max[:3]}"
    )
    assert np.allclose(output_max[3:6], rot_scale), (
        f"OSC rot scales not uniform: {output_max[3:6]}"
    )
    assert np.all(np.asarray(ctrl.input_max) == 1.0) and np.all(
        np.asarray(ctrl.input_min) == -1.0
    ), f"OSC input range not [-1, 1]: [{ctrl.input_min}, {ctrl.input_max}]"
    print(f"  OSC scales read from controller: pos={pos_scale}, rot={rot_scale}")
    return pos_scale, rot_scale


def _execute_chunk_closed_loop(
    env: OffScreenRenderEnv,
    obs: dict,
    chunk: np.ndarray,
    pos_scale: float,
    rot_scale: float,
    frames: list[np.ndarray] | None,
    n_steps: int | None = None,
) -> tuple[dict, float, bool, dict]:
    """Execute an anchor-relative action chunk in closed-loop against the env.

    Args:
        env: LIBERO env.
        obs: latest observation dict — used to snapshot the anchor state
            (p_t, R_t) at chunk start.
        chunk: (H, 7) float32 in PHYSICAL units (anchor-relative):
            chunk[k, 0:3] = cumulative position delta from anchor
            chunk[k, 3:6] = cumulative rotation delta from anchor (axis-angle)
            chunk[k, 6]   = gripper command (unchanged, not a delta)
        pos_scale, rot_scale: robosuite OSC output_max[0] / [3], read once
            per task via `_read_osc_scales`.
        frames: optional list to append agentview images to for video capture.

    Returns:
        obs, reward (scalar), done (bool), info dict — the values returned by
        the last env.step inside the chunk (or the first one that reports done).

    Closed-loop logic: for each step k we compute the target pose relative to
    the snapshotted anchor, then read the robot's current pose from the env and
    send the residual (target - current) scaled into [-1, 1]. This lets the
    controller self-correct when the robot drifts away from the predicted
    trajectory, which is the main robustness gap of naive open-loop execution.
    """
    # Snapshot the anchor state once per chunk (p_t, R_t)
    anchor_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
    anchor_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
    R_anchor = R.from_quat(anchor_quat)  # scipy: xyzw, same as robosuite default

    reward = 0.0
    done = False
    info: dict = {}
    # Receding horizon: execute only the first n_steps of the H-step chunk, then
    # let the caller re-observe and re-predict. n_steps=None executes the whole
    # chunk (legacy execute-then-replan behavior).
    H = chunk.shape[0]
    n_exec = H if n_steps is None else max(1, min(int(n_steps), H))
    for h in range(n_exec):
        # Anchor-relative target derived from the predicted displacement
        target_pos = anchor_pos + chunk[h, 0:3]
        R_step_delta = R.from_rotvec(chunk[h, 3:6])
        target_R = R_step_delta * R_anchor
        gripper_cmd = float(chunk[h, 6])

        # Current pose (closed-loop — reads the env state each step)
        cur_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        cur_R = R.from_quat(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))

        # Residual delta that takes the robot from its current pose to the target
        step_delta_pos = target_pos - cur_pos
        step_delta_rotvec = (target_R * cur_R.inv()).as_rotvec()

        # Robosuite's Controller.scale_action also clips to input_max/min before
        # linear rescaling. We clip explicitly for readability — the clip is
        # bit-for-bit identical to robosuite's internal behavior.
        action_input = np.concatenate(
            [
                np.clip(step_delta_pos / pos_scale, -1.0, 1.0),
                np.clip(step_delta_rotvec / rot_scale, -1.0, 1.0),
                [np.clip(gripper_cmd, -1.0, 1.0)],
            ]
        ).astype(np.float32)

        obs, reward, done, info = env.step(action_input)
        if frames is not None:
            frames.append(obs["agentview_image"])
        if done:
            break

    return obs, reward, done, info


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
    max_video_episodes: int = 999,
    num_warmup_steps: int = 10,
    exec_steps: int | None = None,
) -> tuple[int, int]:
    """Run episodes and count successes.

    Each episode runs ``num_warmup_steps`` no-op env steps after reset to let
    the scene physically settle, then up to ``max_steps`` policy-driven steps.
    The warmup steps are NOT counted toward ``max_steps`` (the policy horizon).

    Receding horizon: when ``exec_steps`` is set (< chunk_size), each predicted
    H-step chunk is executed for only the first ``exec_steps`` steps, then the
    policy re-observes and re-predicts — tighter closed loop, less drift /
    chunk-boundary jitter. ``exec_steps=None`` (or >= chunk_size) keeps the
    legacy execute-the-whole-chunk-then-replan behavior.
    """
    successes = 0
    eff_exec = (
        chunk_size
        if (exec_steps is None or exec_steps <= 0)
        else min(int(exec_steps), chunk_size)
    )

    # Read OSC scales once per task from the running controller (not hardcoded).
    pos_scale, rot_scale = _read_osc_scales(env)

    # No-op OSC action to let the scene settle before the policy acts (LIBERO
    # convention): zero pose deltas, gripper held open.
    dummy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

    # Pre-tokenize language instruction (same for all episodes).
    assert t5_tokenizer is not None, "t5_tokenizer required"
    lang_ids, lang_mask = tokenize_language(
        language_instruction,
        t5_tokenizer,
        max_lang_tokens,
        device,
    )

    for ep in range(num_episodes):
        # Reset with deterministic initial state
        env.reset()
        init_idx = ep % len(init_states)
        obs = env.set_init_state(init_states[init_idx])

        # Collect frames for video (first N episodes only)
        recording = save_videos and ep < max_video_episodes
        frames: list[np.ndarray] | None = [] if recording else None
        if recording:
            frames.append(obs["agentview_image"])

        # Warmup: no-op steps so objects drop into place and the controller
        # stabilizes BEFORE the policy starts. Not counted toward max_steps.
        for _ in range(num_warmup_steps):
            obs, _, _, _ = env.step(dummy_action)
            if frames is not None:
                frames.append(obs["agentview_image"])

        reward = 0.0
        done = False
        steps_done = 0
        while steps_done < max_steps:
            # Preprocess observation (dual-view + proprio)
            pixels_agent, pixels_hand, proprio = preprocess_obs(obs, img_size, device)

            # Encode visual + language
            z_agent, z_hand, lang_embeds, lang_lengths = model.encode(
                pixels_agent,
                pixels_hand,
                lang_ids,
                lang_mask,
            )

            # Generate FAST action tokens
            tokens, lengths = model.predict_actions(
                z_agent,
                z_hand,
                proprio,
                lang_embeds,
                lang_lengths,
                temperature=temperature,
            )

            # Decode tokens → normalized → physical anchor-relative displacements
            actions_norm = fast_decode(
                tokens,
                lengths,
                processor,
                time_horizon=chunk_size,
                action_dim=action_dim,
            )
            actions_phys = denormalize_actions(actions_norm, action_low, action_high)
            # actions_phys[0] is (H, 7) in physical units:
            #   [0:3] = anchor-relative pos delta (m)
            #   [3:6] = anchor-relative rot delta (rad, axis-angle)
            #   [6]   = gripper cmd (unchanged)

            # Clamp the last window so total executed steps never exceed
            # max_steps (the episode horizon cap), even when eff_exec does not
            # divide max_steps under receding-horizon. Without this the policy
            # would get up to eff_exec-1 extra steps vs the protocol.
            this_exec = min(eff_exec, max_steps - steps_done)
            obs, reward, done, _ = _execute_chunk_closed_loop(
                env,
                obs,
                actions_phys[0],
                pos_scale,
                rot_scale,
                frames,
                n_steps=this_exec,
            )
            steps_done += this_exec
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
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to _weights.ckpt"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to saved FAST tokenizer directory",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        required=True,
        help="Directory with preprocessed H5 files (one per task)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ],
        help="LIBERO task suite",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Single task index (0-9). Omit to run all tasks in suite",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Rollouts per task (RynnVLA-002 protocol: num_trials_per_task=50).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max policy-driven env steps per episode (excludes warmup). "
        "When omitted, auto-selected per --suite from LIBERO_MAX_STEPS "
        "(spatial 220 / object 280 / goal 300 / 10 (long) 520 / 90 400) — the "
        "OpenVLA / RynnVLA-002 convention. Pass explicitly to override.",
    )
    parser.add_argument(
        "--num-warmup-steps",
        type=int,
        default=10,
        help="No-op env steps to settle the scene before the policy acts "
        "(not counted toward --max-steps).",
    )
    parser.add_argument(
        "--exec-steps",
        type=int,
        default=None,
        help="Receding-horizon: execute only the first N steps of each predicted "
        "H-step (=chunk_size) chunk, then re-observe and re-predict. None or "
        ">=chunk_size = legacy execute-whole-chunk-then-replan. Try 5-8 for a "
        "tighter loop / less drift. Inference-only; no retrain needed.",
    )
    parser.add_argument(
        "--camera-size",
        type=int,
        default=128,
        help="LIBERO camera resolution (env render size). "
        "Must match the resolution of images stored in preprocessed "
        "training HDF5 (LIBERO default: 128).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="0=greedy, >0=sampling"
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save rollout videos for first 3 episodes per task",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="/data/lyw/eval_videos",
        help="Directory to save videos",
    )
    parser.add_argument(
        "--max-video-episodes",
        type=int,
        default=999,
        help="Max episodes per task to record (default: all)",
    )
    args = parser.parse_args()

    if args.save_videos and iio is None:
        raise ImportError(
            "imageio required for --save-videos: pip install imageio imageio-ffmpeg"
        )

    device = torch.device(args.device)
    np.random.seed(args.seed)

    # Resolve per-suite policy horizon (OpenVLA / RynnVLA-002 convention) when
    # --max-steps is not given explicitly. Explicit --max-steps still wins.
    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else LIBERO_MAX_STEPS.get(args.suite, 300)
    )
    print(
        f"Eval horizon: max_steps={max_steps} (warmup={args.num_warmup_steps}, "
        f"episodes={args.num_episodes}) for suite={args.suite}"
    )

    # Load model — two formats supported:
    #   _weights.ckpt  = Lightning state_dict (from spt.Manager)
    #   _object.ckpt   = torch.save(model) pickle (from ModelObjectCallBack)
    print("Loading model...")
    if args.checkpoint.endswith("_object.ckpt"):
        # Object checkpoint: torch.save(model) — loads the full JEPA directly.
        print(f"  Loading object checkpoint: {args.checkpoint}")
        model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    else:
        # Lightning checkpoint: build architecture then load state_dict.
        model = build_model(device)
        load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # Load FAST tokenizer
    print(f"Loading FAST tokenizer from {args.tokenizer}")
    processor = load_fast_processor(args.tokenizer)

    # Load T5 tokenizer for language
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

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
        print(f"\n{'=' * 60}")
        print(f"Task {task_id}: {task_name}")
        print(f"{'=' * 60}")

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
            get_libero_path("bddl_files"),
            args.suite,
            task.bddl_file,
        )
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=args.camera_size,
            camera_widths=args.camera_size,
        )
        init_states = task_suite.get_task_init_states(task_id)

        try:
            successes, n_eps = evaluate_task(
                model,
                processor,
                t5_tokenizer,
                env,
                init_states,
                action_low,
                action_high,
                chunk_size,
                action_dim,
                language_instruction,
                num_episodes=args.num_episodes,
                max_steps=max_steps,
                num_warmup_steps=args.num_warmup_steps,
                exec_steps=args.exec_steps,
                device=device,
                temperature=args.temperature,
                save_videos=args.save_videos,
                video_dir=args.video_dir,
                task_name=task_name,
                max_video_episodes=args.max_video_episodes,
            )
        finally:
            env.close()

        rate = successes / n_eps * 100
        results[task_id] = {
            "name": task_name,
            "success": successes,
            "total": n_eps,
            "rate": rate,
        }
        total_successes += successes
        total_episodes += n_eps

        print(f"  Result: {successes}/{n_eps} ({rate:.1f}%)")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {args.suite}")
    print(f"{'=' * 60}")
    for tid, r in results.items():
        print(
            f"  Task {tid:2d}: {r['success']:2d}/{r['total']} ({r['rate']:5.1f}%) — {r['name']}"
        )
    if total_episodes > 0:
        overall = total_successes / total_episodes * 100
        print(f"\n  Overall: {total_successes}/{total_episodes} ({overall:.1f}%)")


if __name__ == "__main__":
    main()

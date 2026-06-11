"""LIBERO evaluation: VLA baseline — autoregressive action generation → environment execution.

Loads a trained checkpoint, runs the model in LIBERO environments, and
measures task success rate. The model takes dual-view images, proprioception,
and language instruction as input.

Usage:
    # Single task
    python eval_libero.py \
        --checkpoint /Data/lyw/stable-wm/lewm_weights.ckpt \
        --tokenizer /Data/lyw/fast_tokenizer \
        --processed-dir /Data/lyw/libero_processed_v5/libero_spatial \
        --suite libero_spatial --task-id 0 \
        --num-episodes 50

    # All tasks in a suite
    python eval_libero.py \
        --checkpoint /Data/lyw/stable-wm/lewm_weights.ckpt \
        --tokenizer /Data/lyw/fast_tokenizer \
        --processed-dir /Data/lyw/libero_processed_v5/libero_spatial \
        --suite libero_spatial \
        --num-episodes 50
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
from libero_baseline_protocol import (
    DEFAULT_CAMERA_SIZE,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_NUM_STEPS_WAIT,
    DUMMY_ACTION,
    SUITE_MAX_STEPS,
    get_suite_max_steps,
    rotate_image_180,
)
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
        "tiny",
        patch_size=14,
        image_size=224,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = 192

    predictor = ARPredictor(
        embed_dim=embed_dim,
        max_action_tokens=80,
        max_lang_tokens=25,
        proprio_dim=9,
        # dropout=0.2 mirrors config/train/lewm.yaml (the frozen baseline's
        # actual training value). Eval mode is no-op for nn.Dropout, so this
        # has no inference effect, but it keeps the constructor call honest
        # and stops future maintainers from chasing a phantom mismatch when
        # they cross-reference eval and training code.
        depth=6,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.2,
        emb_dropout=0.0,
    )
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.LayerNorm,
    )

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
        encoder=encoder,
        predictor=predictor,
        projector=projector,
        lang_encoder=lang_encoder,
        lang_proj=lang_proj,
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
    # (no state_pred_head_*, no state_query_embeddings, LayerNorm projector,
    # no gripper_aux_head), so loading a checkpoint with extra keys would
    # either fail with a confusing torch shape-mismatch error or (worse)
    # silently pass with strict=False and skip critical parameters.
    #
    # Four known deviations all require _object.ckpt:
    #   1. SP-trained: state_pred_head_* / state_query_embeddings keys.
    #   2. BatchNorm-projector: projector.net.1.running_mean buffer (BN1d
    #      tracks running stats; LayerNorm has no such buffer). This catches
    #      the legitimate "SIGReg-only, no SP" ablation as well, where
    #      projector.norm_type='batch' but no SP heads exist.
    #   3. Gripper-aux-trained: gripper_aux_head.* keys. Bypass-FAST head
    #      for the gripper dim; build_model() never constructs it.
    #   4. V17 / visual-prefix trained: view_embedding or per-view 2D patch
    #      position keys. build_model() constructs CLS-only visual inputs.
    sp_keys = [
        k for k in state_dict if "state_pred_head" in k or "state_query_embeddings" in k
    ]
    bn_keys = [k for k in state_dict if "projector.net.1.running_mean" in k]
    grip_keys = [k for k in state_dict if "gripper_aux_head" in k]
    visual_keys = [
        k
        for k in state_dict
        if (
            "view_embedding" in k
            or "agent_patch_2d_pos" in k
            or "hand_patch_2d_pos" in k
        )
    ]
    if (
        sp_keys or bn_keys or grip_keys or visual_keys
    ) and not ckpt_path.endswith("_object.ckpt"):
        reasons = []
        if sp_keys:
            reasons.append(
                f"state-prediction keys ({sp_keys[:2]}{'...' if len(sp_keys) > 2 else ''})"
            )
        if bn_keys:
            reasons.append(f"BatchNorm projector running stats ({bn_keys[:1]})")
        if grip_keys:
            reasons.append(
                f"gripper-aux head keys ({grip_keys[:2]}{'...' if len(grip_keys) > 2 else ''})"
            )
        if visual_keys:
            reasons.append(
                f"visual-prefix keys ({visual_keys[:2]}{'...' if len(visual_keys) > 2 else ''})"
            )
        raise ValueError(
            f"Checkpoint '{ckpt_path}' has {' and '.join(reasons)} but is not "
            "an _object.ckpt. Non-baseline architectures (SP-trained, "
            "BatchNorm-projector / SIGReg-trained, gripper-aux-trained, "
            "or visual-prefix-trained) "
            "must be evaluated via the per-epoch object checkpoint produced "
            "by ModelObjectCallBack — pass "
            "--checkpoint .../lewm_*_object.ckpt instead."
        )

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


def _denormalize_gripper_aux(
    grip_norm: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """Map gripper aux output from normalized space to env command space."""
    grip_norm = np.clip(np.asarray(grip_norm, dtype=np.float64), -1.0, 1.0)
    grip_low = float(action_low[6])
    grip_high = float(action_high[6])
    grip_mid = (grip_high + grip_low) / 2.0
    grip_half_range = (grip_high - grip_low) / 2.0
    if grip_half_range < 1e-8:
        return np.full_like(grip_norm, grip_mid, dtype=np.float32)
    return (grip_norm * grip_half_range + grip_mid).astype(np.float32)


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
    # Agentview image — OpenVLA / WorldVLA rotate LIBERO camera images by 180 deg.
    img_agent = _preprocess_image(rotate_image_180(obs["agentview_image"]), img_size)
    pixels_agent = img_agent.unsqueeze(0).to(device)

    # Eye-in-hand image — same rotation as training preprocessing.
    img_hand = _preprocess_image(
        rotate_image_180(obs["robot0_eye_in_hand_image"]),
        img_size,
    )
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


def _seed_env(env: OffScreenRenderEnv, seed: int) -> None:
    """Best-effort seed for LIBERO / robosuite env variants."""
    for candidate in (env, getattr(env, "env", None)):
        if candidate is None or not hasattr(candidate, "seed"):
            continue
        try:
            candidate.seed(seed)
        except TypeError:
            try:
                candidate.seed(seed=seed)
            except Exception:
                continue
        except Exception:
            continue


def _execute_chunk_closed_loop(
    env: OffScreenRenderEnv,
    obs: dict,
    chunk: np.ndarray,
    pos_scale: float,
    rot_scale: float,
    frames: list[np.ndarray] | None,
    *,
    max_chunk_steps: int | None = None,
) -> tuple[dict, float, bool, dict, int]:
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
        max_chunk_steps: optional cap on raw env steps for this chunk. Used at
            the end of an episode so evaluation never steps past
            max_steps + num_steps_wait.

    Returns:
        obs, reward (scalar), done (bool), info dict, steps_executed — the
        values returned by the last env.step inside the chunk (or the first one
        that reports done).

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
    H = chunk.shape[0]
    if max_chunk_steps is not None:
        H = min(H, max(0, int(max_chunk_steps)))
    steps_executed = 0
    for h in range(H):
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
        steps_executed += 1
        if frames is not None:
            frames.append(obs["agentview_image"])
        if done:
            break

    return obs, reward, done, info, steps_executed


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
    num_episodes: int = DEFAULT_EVAL_EPISODES,
    max_steps: int = 300,
    num_steps_wait: int = DEFAULT_NUM_STEPS_WAIT,
    img_size: int = DEFAULT_CAMERA_SIZE,
    max_lang_tokens: int = 25,
    device: torch.device = torch.device("cuda"),
    temperature: float = 0.0,
    save_videos: bool = False,
    video_dir: str | None = None,
    task_name: str = "",
    max_video_episodes: int = 999,
    use_language: bool = True,
) -> tuple[int, int]:
    """Run episodes and count successes."""
    successes = 0
    total_step_budget = int(max_steps) + int(num_steps_wait)

    # Read OSC scales once per task from the running controller (not hardcoded).
    pos_scale, rot_scale = _read_osc_scales(env)

    # Pre-tokenize language instruction (same for all episodes); skipped in no-language mode.
    if use_language:
        assert t5_tokenizer is not None, "t5_tokenizer required when use_language=True"
        lang_ids, lang_mask = tokenize_language(
            language_instruction,
            t5_tokenizer,
            max_lang_tokens,
            device,
        )
    else:
        lang_ids, lang_mask = None, None

    if temperature > 0:
        print(
            f"  Sampling decode enabled: temperature={temperature}, "
            "seed controlled by --seed."
        )

    for ep in range(num_episodes):
        done = False
        frames: list[np.ndarray] | None = None
        recording = save_videos and ep < max_video_episodes
        exception_msg: str | None = None
        try:
            if ep >= len(init_states):
                raise IndexError(
                    f"LIBERO init_states has {len(init_states)} entries, "
                    f"but episode {ep} requires init_states[{ep}]"
                )

            # Reset with deterministic benchmark initial state: episode i uses
            # initial_states[i], with no modulo or additional randomization.
            env.reset()
            obs = env.set_init_state(init_states[ep])

            # Collect frames for video (first N episodes only)
            frames = [] if recording else None
            if recording:
                frames.append(obs["agentview_image"])

            t = 0
            for _ in range(num_steps_wait):
                obs, _reward, done, _info = env.step(
                    np.asarray(DUMMY_ACTION, dtype=np.float32)
                )
                t += 1
                if frames is not None:
                    frames.append(obs["agentview_image"])
                if done:
                    break

            while not done and t < total_step_budget:
                remaining = total_step_budget - t
                # Preprocess observation (dual-view + proprio)
                pixels_agent, pixels_hand, proprio = preprocess_obs(
                    obs,
                    img_size,
                    device,
                )

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

                # Gripper aux override: when the model was trained with the
                # auxiliary gripper-command head, bypass FAST for dim 6 and
                # take the direct regression head's output instead. This
                # addresses the libero_object 0% failure where FAST joint BPE
                # diluted the gripper signal — see CLAUDE.md diagnostic notes.
                if getattr(model.predictor, "use_gripper_aux", False):
                    pred_grip = model.predict_gripper_aux(
                        z_agent,
                        z_hand,
                        proprio,
                        lang_embeds,
                        lang_lengths,
                    )  # (B=1, H)
                    actions_phys[0, :, 6] = _denormalize_gripper_aux(
                        pred_grip[0].detach().cpu().numpy(),
                        action_low,
                        action_high,
                    )

                obs, _reward, done, _info, steps_executed = _execute_chunk_closed_loop(
                    env,
                    obs,
                    actions_phys[0],
                    pos_scale,
                    rot_scale,
                    frames,
                    max_chunk_steps=remaining,
                )
                if steps_executed <= 0:
                    break
                t += steps_executed
        except Exception as exc:  # noqa: BLE001
            done = False
            exception_msg = str(exc)

        success = bool(done)
        if success:
            successes += 1
        status = "SUCCESS" if success else "FAIL"
        if exception_msg is not None:
            status = f"{status} (exception: {exception_msg})"
        print(f"  Episode {ep}: {status}")

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
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a Lightning _weights.ckpt or full-model _object.ckpt",
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
        default=DEFAULT_EVAL_EPISODES,
        help="Episodes per task (50 gives 500 trials per suite).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Max raw policy steps per episode. Defaults by suite: "
            f"{SUITE_MAX_STEPS}."
        ),
    )
    parser.add_argument(
        "--num-steps-wait",
        type=int,
        default=DEFAULT_NUM_STEPS_WAIT,
        help="Initial dummy-action settling steps before policy rollout.",
    )
    parser.add_argument(
        "--camera-size",
        type=int,
        default=DEFAULT_CAMERA_SIZE,
        help="LIBERO camera resolution (env render size). "
        "Must match this model's input resolution (default: 224).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="0=greedy, >0=sampling"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save rollout videos for first 3 episodes per task",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="/Data/lyw/eval_videos",
        help="Directory to save videos",
    )
    parser.add_argument(
        "--max-video-episodes",
        type=int,
        default=999,
        help="Max episodes per task to record (default: all)",
    )
    parser.add_argument(
        "--no-language",
        action="store_true",
        help="Ablation: evaluate a model trained without the language instruction. "
        "Must match the checkpoint's training-time use_language setting.",
    )
    args = parser.parse_args()

    if args.save_videos and iio is None:
        raise ImportError(
            "imageio required for --save-videos: pip install imageio imageio-ffmpeg"
        )

    use_language = not args.no_language

    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    effective_max_steps = (
        get_suite_max_steps(args.suite) if args.max_steps is None else int(args.max_steps)
    )
    if args.temperature <= 0:
        print("Decoding: greedy (temperature=0.0), deterministic for fixed init states.")
    else:
        print(f"Decoding: sampling (temperature={args.temperature}, seed={args.seed}).")
    print(
        f"Eval protocol: suite={args.suite}, episodes/task={args.num_episodes}, "
        f"max_steps={effective_max_steps}, wait={args.num_steps_wait}, "
        f"camera={args.camera_size}"
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
        if use_language and not str(language_instruction).strip():
            raise ValueError(
                f"Empty language_instruction in {h5_path}. This makes LIBERO "
                "multi-task evaluation ambiguous; re-run preprocess_libero.py "
                "with a version that records task language, or pass "
                "--no-language only for a checkpoint trained with "
                "data.dataset.use_language=false."
            )
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
        _seed_env(env, args.seed)
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
                max_steps=effective_max_steps,
                num_steps_wait=args.num_steps_wait,
                img_size=args.camera_size,
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
        sr = total_successes / total_episodes
        print(f"\n  SR: {total_successes}/{total_episodes} = {sr:.4f}")
        print(f"\n  Overall: {total_successes}/{total_episodes} ({overall:.1f}%)")


if __name__ == "__main__":
    main()

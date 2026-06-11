"""Regenerate LIBERO demos with OpenVLA-style success/no-op filtering.

The output keeps the robomimic-style HDF5 layout consumed by
``preprocess_libero.py`` while replacing stored observations with simulator
renders at the target model resolution. Actions are replayed in the simulator
for fidelity; only successful trajectories are saved, and no-op actions are
filtered from the saved sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from libero_baseline_protocol import (
    DEFAULT_CAMERA_SIZE,
    DEFAULT_NUM_STEPS_WAIT,
    DUMMY_ACTION,
    is_noop_action,
)
from preprocess_libero import load_demo_keys


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def _normalize_task_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\.(hdf5|h5)$", "", value)
    value = re.sub(r"_demo$", "", value)
    value = value.replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _resolve_task_id(suite: str, input_path: Path, task_id: int | None) -> int:
    task_suite = benchmark.get_benchmark_dict()[suite]()
    if task_id is not None:
        if task_id < 0 or task_id >= task_suite.n_tasks:
            raise ValueError(f"task_id={task_id} outside 0..{task_suite.n_tasks - 1}")
        return task_id

    stem = _normalize_task_name(input_path.stem)
    for tid in range(task_suite.n_tasks):
        task_name = _normalize_task_name(task_suite.get_task(tid).name)
        if task_name in stem or stem in task_name:
            return tid
    raise ValueError(
        f"Could not infer task_id for {input_path}. Pass --task-id explicitly."
    )


def _state_vector(env: OffScreenRenderEnv) -> np.ndarray:
    sim = getattr(getattr(env, "env", env), "sim", None)
    if sim is None:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(sim.get_state().flatten(), dtype=np.float32)


def _robot_state_from_obs(obs: dict[str, Any]) -> np.ndarray:
    robot_state = np.zeros((9,), dtype=np.float32)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    robot_state[5:9] = quat
    return robot_state


def _obs_record(obs: dict[str, Any], env: OffScreenRenderEnv) -> dict[str, np.ndarray]:
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    return {
        "agentview_rgb": np.asarray(obs["agentview_image"], dtype=np.uint8),
        "eye_in_hand_rgb": np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8),
        "ee_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        "ee_ori": R.from_quat(quat).as_rotvec().astype(np.float32),
        "gripper_states": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        "robot_states": _robot_state_from_obs(obs),
        "states": _state_vector(env),
    }


def _stack_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = records[0].keys()
    return {key: np.stack([rec[key] for rec in records], axis=0) for key in keys}


def _initial_state(demo: h5py.Group) -> np.ndarray | None:
    if "states" in demo and demo["states"].shape[0] > 0:
        return demo["states"][0]
    return None


def _replay_demo(
    env: OffScreenRenderEnv,
    demo: h5py.Group,
    *,
    wait_steps: int,
    noop_threshold: float,
) -> tuple[bool, np.ndarray, dict[str, np.ndarray]]:
    raw_actions = np.asarray(demo["actions"][()], dtype=np.float32)
    env.reset()
    init_state = _initial_state(demo)
    if init_state is not None:
        obs = env.set_init_state(init_state)
    else:
        obs = env.reset()

    done = False
    for _ in range(wait_steps):
        obs, _reward, done, _info = env.step(np.asarray(DUMMY_ACTION, dtype=np.float32))
        if done:
            break

    kept_actions: list[np.ndarray] = []
    kept_obs: list[dict[str, np.ndarray]] = []
    previous_kept_action: np.ndarray | None = None
    for action in raw_actions:
        keep_action = not is_noop_action(
            action,
            previous_kept_action,
            threshold=noop_threshold,
        )
        pre_step_record = _obs_record(obs, env) if keep_action else None
        obs, _reward, done, _info = env.step(action)
        if keep_action:
            kept_action = action.astype(np.float32)
            kept_actions.append(kept_action)
            assert pre_step_record is not None
            kept_obs.append(pre_step_record)
            previous_kept_action = kept_action
        if done:
            break

    if not done or not kept_actions:
        return False, np.zeros((0, 7), dtype=np.float32), {}
    return True, np.stack(kept_actions, axis=0), _stack_records(kept_obs)


def _write_demo(
    out_data: h5py.Group,
    demo_name: str,
    source_demo: h5py.Group,
    actions: np.ndarray,
    obs_data: dict[str, np.ndarray],
) -> None:
    demo_out = out_data.create_group(demo_name)
    _copy_attrs(source_demo.attrs, demo_out.attrs)
    demo_out.create_dataset("actions", data=actions, dtype=np.float32)
    if obs_data["states"].size > 0:
        demo_out.create_dataset("states", data=obs_data["states"], dtype=np.float32)
    demo_out.create_dataset("robot_states", data=obs_data["robot_states"], dtype=np.float32)
    obs_out = demo_out.create_group("obs")
    for key in (
        "agentview_rgb",
        "eye_in_hand_rgb",
        "ee_pos",
        "ee_ori",
        "gripper_states",
    ):
        obs_out.create_dataset(key, data=obs_data[key])


def regenerate_file(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    task_id = _resolve_task_id(args.suite, input_path, args.task_id)
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(task_id)
    bddl_path = os.path.join(get_libero_path("bddl_files"), args.suite, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    failed = 0
    noop_removed = 0
    total_actions = 0
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as out:
        _copy_attrs(src.attrs, out.attrs)
        out_data = out.create_group("data")
        _copy_attrs(src["data"].attrs, out_data.attrs)
        out_data.attrs["regenerated_filter"] = "openvla_no_noops"
        out_data.attrs["camera_size"] = int(args.camera_size)
        out_data.attrs["num_steps_wait"] = int(args.wait_steps)
        out_data.attrs["noop_threshold"] = float(args.noop_threshold)
        out_data.attrs["source_file"] = str(input_path.resolve())
        out_data.attrs["task_id"] = int(task_id)
        out_data.attrs["task_name"] = task.name

        for demo_name in load_demo_keys(src):
            source_demo = src[f"data/{demo_name}"]
            raw_len = int(source_demo["actions"].shape[0])
            total_actions += raw_len
            ok, actions, obs_data = _replay_demo(
                env,
                source_demo,
                wait_steps=args.wait_steps,
                noop_threshold=args.noop_threshold,
            )
            if not ok:
                failed += 1
                print(f"[drop] {demo_name}: replay did not reach success")
                continue
            new_name = f"demo_{kept}"
            noop_removed += raw_len - int(actions.shape[0])
            _write_demo(out_data, new_name, source_demo, actions, obs_data)
            print(f"[keep] {demo_name} -> {new_name}: {raw_len} -> {actions.shape[0]}")
            kept += 1

        summary = {
            "input": str(input_path),
            "output": str(output_path),
            "suite": args.suite,
            "task_id": task_id,
            "task_name": task.name,
            "kept_demos": kept,
            "failed_demos": failed,
            "total_actions": total_actions,
            "noop_actions_removed": noop_removed,
        }
        out_data.attrs["filter_summary_json"] = json.dumps(summary, sort_keys=True)
    env.close()
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Official LIBERO task HDF5")
    parser.add_argument("--output", required=True, help="Filtered output HDF5")
    parser.add_argument(
        "--suite",
        required=True,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Task id in the LIBERO suite. If omitted, infer from file name.",
    )
    parser.add_argument("--camera-size", type=int, default=DEFAULT_CAMERA_SIZE)
    parser.add_argument("--wait-steps", type=int, default=DEFAULT_NUM_STEPS_WAIT)
    parser.add_argument("--noop-threshold", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    regenerate_file(parse_args())


if __name__ == "__main__":
    main()

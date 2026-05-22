"""quick_probe_eval.py — In-training health probe.

Stage A: 10 task per suite × num_episodes rollouts (default 1) = 40 rollouts total.
Used by EarlyProbeCallback to surface obvious failure modes (0% success, BN
collapse) early in a long training run.

Stage B: 4 fixed tasks × num_episodes (default 2) = 8 rollouts. Disabled by
default in the 4-suite single-arm run; scaffolded here for future revival.

Reuses eval_libero.py's model loader + rollout helpers verbatim. Outputs a
single JSON line on the LAST line of stdout for the parent EarlyProbeCallback
to parse:

    {"overall_success_rate": 0.05, "per_suite": {"libero_spatial": 0.10, ...},
     "n_total": 40, "n_per_suite": 10}
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from transformers import T5Tokenizer

from eval_libero import (
    build_model,
    evaluate_task,
    find_processed_h5,
    load_action_stats,
    load_checkpoint,
    load_language_instruction,
)
from fast_utils import load_fast_processor


SUITES_A: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

# Stage B fixed task list: (suite, task_name). Populated after first Stage A
# run on baseline arm finalizes the choices (medium-difficulty per suite).
STAGE_B_TASKS: tuple[tuple[str, str], ...] = (
    (
        "libero_spatial",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    ),
    # TBD for object/goal/long — placeholders, replace after baseline run.
    ("libero_object", ""),
    ("libero_goal", ""),
    ("libero_10", ""),
)


def _resolve_processed_dir(processed_root: Path, suite: str) -> Path:
    """Resolve <root>/<suite>/ or <root>/<suite-stripped>/ for the suite-prefixed flat dir."""
    direct = processed_root / suite
    if direct.is_dir():
        return direct
    raise FileNotFoundError(f"Processed dir for {suite} not under {processed_root}")


def probe_one_task(
    suite: str,
    task_id: int,
    task,
    model,
    processor,
    t5_tokenizer,
    processed_dir: Path,
    num_episodes: int,
    max_steps: int,
    device,
) -> tuple[int, int]:
    """Run a single task. Returns (successes, total_episodes)."""
    task_name = task.name
    h5_path = find_processed_h5(str(processed_dir), task_name)
    if h5_path is None:
        return 0, 0
    action_low, action_high, chunk_size, action_dim = load_action_stats(h5_path)
    language_instruction = load_language_instruction(h5_path)
    bddl_path = os.path.join(get_libero_path("bddl_files"), suite, task.bddl_file)

    task_suite = benchmark.get_benchmark_dict()[suite]()
    init_states = task_suite.get_task_init_states(task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path, camera_heights=128, camera_widths=128
    )
    try:
        # Suppress per-episode "SUCCESS/FAIL" stdout chatter — the parent
        # EarlyProbeCallback parses our JSON, not the verbose log.
        with contextlib.redirect_stdout(io.StringIO()):
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
                num_episodes=num_episodes,
                max_steps=max_steps,
                device=device,
                temperature=0.0,
                save_videos=False,
                video_dir=None,
                task_name=task_name,
                max_video_episodes=0,
            )
    finally:
        env.close()
    return successes, n_eps


def run_stage_a(
    model,
    processor,
    t5_tokenizer,
    processed_root: Path,
    num_episodes: int,
    max_steps: int,
    device,
) -> dict:
    per_suite: dict[str, dict[str, int]] = {}
    total_succ = 0
    total_eps = 0
    for suite in SUITES_A:
        suite_succ = 0
        suite_eps = 0
        task_suite = benchmark.get_benchmark_dict()[suite]()
        processed_dir = _resolve_processed_dir(processed_root, suite)
        for tid in range(task_suite.n_tasks):
            task = task_suite.get_task(tid)
            succ, n = probe_one_task(
                suite,
                tid,
                task,
                model,
                processor,
                t5_tokenizer,
                processed_dir,
                num_episodes,
                max_steps,
                device,
            )
            suite_succ += succ
            suite_eps += n
            print(
                f"  [{suite} t{tid}] {succ}/{n} — {task.name[:60]}",
                flush=True,
                file=sys.stderr,
            )
        per_suite[suite] = {"succ": suite_succ, "eps": suite_eps}
        total_succ += suite_succ
        total_eps += suite_eps
    overall = total_succ / total_eps if total_eps > 0 else 0.0
    return {
        "stage": "A",
        "overall_success_rate": overall,
        "per_suite": {
            k: (v["succ"] / v["eps"] if v["eps"] > 0 else 0.0)
            for k, v in per_suite.items()
        },
        "per_suite_counts": per_suite,
        "n_total": total_eps,
    }


def run_stage_b(
    model,
    processor,
    t5_tokenizer,
    processed_root: Path,
    num_episodes: int,
    max_steps: int,
    device,
) -> dict:
    per_suite: dict[str, dict[str, int]] = {}
    total_succ = 0
    total_eps = 0
    for suite, task_slug in STAGE_B_TASKS:
        if not task_slug:
            continue
        # Find task_id by matching task.name
        task_suite = benchmark.get_benchmark_dict()[suite]()
        match_tid = None
        for tid in range(task_suite.n_tasks):
            if task_suite.get_task(tid).name == task_slug:
                match_tid = tid
                break
        if match_tid is None:
            print(f"  [WARN] task '{task_slug}' not in {suite}", file=sys.stderr)
            continue
        processed_dir = _resolve_processed_dir(processed_root, suite)
        task = task_suite.get_task(match_tid)
        succ, n = probe_one_task(
            suite,
            match_tid,
            task,
            model,
            processor,
            t5_tokenizer,
            processed_dir,
            num_episodes,
            max_steps,
            device,
        )
        per_suite[suite] = {"succ": succ, "eps": n}
        total_succ += succ
        total_eps += n
        print(f"  [{suite} {task_slug[:40]}] {succ}/{n}", file=sys.stderr)
    overall = total_succ / total_eps if total_eps > 0 else 0.0
    return {
        "stage": "B",
        "overall_success_rate": overall,
        "per_suite": {
            k: (v["succ"] / v["eps"] if v["eps"] > 0 else 0.0)
            for k, v in per_suite.items()
        },
        "per_suite_counts": per_suite,
        "n_total": total_eps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/Data/lyw/libero_processed_v5"),
        help="Parent dir containing libero_{spatial,object,goal,10}/ subdirs",
    )
    parser.add_argument(
        "--stage",
        choices=("A", "B"),
        default="A",
        help="A: 10 task per suite × num-episodes (40 rollouts at default). "
        "B: 4 fixed tasks × num-episodes (8 rollouts at default 2 episodes).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Stage A default: 1. Stage B default: 2 (caller sets per stage).",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model — object ckpt path (required for BN / SP architectures).
    print(f"Loading model from {args.checkpoint}", file=sys.stderr)
    if args.checkpoint.endswith("_object.ckpt"):
        model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    else:
        model = build_model(device)
        load_checkpoint(model, args.checkpoint, device)
    model.eval()

    processor = load_fast_processor(args.tokenizer)
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    if args.stage == "A":
        result = run_stage_a(
            model,
            processor,
            t5_tokenizer,
            args.processed_root,
            args.num_episodes,
            args.max_steps,
            device,
        )
    else:
        result = run_stage_b(
            model,
            processor,
            t5_tokenizer,
            args.processed_root,
            args.num_episodes,
            args.max_steps,
            device,
        )

    # Single-line JSON on stdout — required for EarlyProbeCallback parser.
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

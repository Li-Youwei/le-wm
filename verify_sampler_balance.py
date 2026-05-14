"""verify_sampler_balance.py — Sanity-check the 3-level WeightedRandomSampler.

Builds a LiberoDataset on the given flat hdf5 directory, draws a full epoch's
worth of samples via WeightedRandomSampler, then reports:
  - Per-task sampling frequency + coefficient of variation (CV).
  - Per-suite aggregate (spatial / object / goal / 10).
  - Pass/fail: CV < 5% per-task AND per-suite within ±1pp of 25%.

The intent is to catch sampler weight bugs (e.g., wrong dict keys, missing
demos) BEFORE kicking off a 17h training run.

Usage:
    python verify_sampler_balance.py \\
        --hdf5-dir /Data/lyw/libero_processed_v5/all4_flat \\
        --num-epochs 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from libero_dataset import LiberoDataset

logger = logging.getLogger("verify_sampler_balance")

SUITE_PREFIXES = ("spatial", "object", "goal", "10")


def classify_suite(stem: str) -> str:
    """Map a flat-dir filename stem to one of the four suite prefixes.

    Files are named like ``spatial_pick_up_the_black_bowl_xxx`` (after Phase 3
    symlinking). Empty string if no prefix matches.
    """
    stem_l = stem.lower()
    for prefix in SUITE_PREFIXES:
        if stem_l.startswith(prefix + "_"):
            return prefix
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-dir", type=Path, required=True)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument(
        "--max-action-tokens",
        type=int,
        default=80,
        help="Passed to LiberoDataset; doesn't matter for sampler verification "
        "but must match how training will construct it (for the n_demos count).",
    )
    parser.add_argument(
        "--use-language",
        action="store_true",
        help="Load T5 tokenizer too. Default off — faster init; sampler weights "
        "don't depend on language.",
    )
    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=0.05,
        help="Pass criterion: per-task sample-count CV must be < this fraction.",
    )
    parser.add_argument(
        "--suite-tol",
        type=float,
        default=0.01,
        help="Pass criterion: each suite's share must be within ±suite-tol of 0.25.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.hdf5_dir.is_dir():
        raise FileNotFoundError(args.hdf5_dir)

    logger.info("Building LiberoDataset from %s ...", args.hdf5_dir)
    dataset = LiberoDataset(
        hdf5_dir=str(args.hdf5_dir),
        max_action_tokens=args.max_action_tokens,
        max_lang_tokens=25,
        img_size=224,
        use_language=args.use_language,
        use_state_prediction=False,
    )
    n_tasks = len(dataset.files)
    n_samples = len(dataset)
    logger.info("Dataset: %d files (tasks), %d total chunks", n_tasks, n_samples)

    # Build a suite classification per task_id (matching dataset's sorted file order).
    suite_of_task: list[str] = []
    for fpath in dataset.files:
        suite = classify_suite(fpath.stem)
        suite_of_task.append(suite)
    n_per_suite_tasks = Counter(suite_of_task)
    logger.info("Suite task counts: %s", dict(n_per_suite_tasks))

    weights = dataset.get_sampler_weights()
    logger.info(
        "Sampler weights: min=%.6e median=%.6e max=%.6e (sum=%.4f)",
        float(weights.min()),
        float(weights.median()),
        float(weights.max()),
        float(weights.sum()),
    )

    rnd = torch.Generator().manual_seed(args.seed)
    total_draws = args.num_epochs * n_samples
    logger.info("Drawing %d samples via WeightedRandomSampler ...", total_draws)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=total_draws,
        replacement=True,
        generator=rnd,
    )
    drawn = list(sampler)

    # Per-task counts.
    task_ids = np.array(
        [dataset._file_to_task_id[dataset._index[i][0]] for i in drawn],
        dtype=np.int64,
    )
    per_task_counts = np.bincount(task_ids, minlength=n_tasks)

    # Per-suite counts.
    per_suite_counts: Counter[str] = Counter()
    for tid, count in enumerate(per_task_counts.tolist()):
        per_suite_counts[suite_of_task[tid]] += count

    # Reports.
    per_task_mean = per_task_counts.mean()
    per_task_std = per_task_counts.std()
    per_task_cv = per_task_std / per_task_mean if per_task_mean > 0 else float("inf")

    print()
    print("=== Per-task draw counts (40 tasks expected) ===")
    for tid in range(n_tasks):
        suite = suite_of_task[tid]
        name = dataset.files[tid].stem
        share = per_task_counts[tid] / total_draws
        print(
            f"  task_{tid:02d} [{suite:7s}] n={per_task_counts[tid]:6d} "
            f"share={share * 100:5.2f}%  {name[:80]}"
        )
    print(
        f"  >> mean={per_task_mean:.0f}  std={per_task_std:.0f}  "
        f"CV={per_task_cv * 100:.2f}%  threshold<{args.cv_threshold * 100:.1f}%"
    )

    print()
    print("=== Per-suite shares (expect ~25% each) ===")
    suite_shares: dict[str, float] = {}
    for suite in SUITE_PREFIXES:
        cnt = per_suite_counts.get(suite, 0)
        share = cnt / total_draws if total_draws > 0 else 0.0
        suite_shares[suite] = share
        print(
            f"  {suite:7s}: n={cnt:7d}  share={share * 100:5.2f}%  "
            f"(target=25.00% ±{args.suite_tol * 100:.2f}pp)"
        )

    # Pass / fail.
    task_ok = per_task_cv < args.cv_threshold
    suite_ok = all(abs(s - 0.25) <= args.suite_tol for s in suite_shares.values())
    overall_ok = task_ok and suite_ok

    print()
    print("=== Sanity gate ===")
    print(
        f"  per-task CV ok ({per_task_cv * 100:.2f}% < {args.cv_threshold * 100:.1f}%): {task_ok}"
    )
    print(
        f"  per-suite distribution ok (each within ±{args.suite_tol * 100:.2f}pp of 25%): {suite_ok}"
    )
    print(f"  OVERALL: {'PASS' if overall_ok else 'FAIL'}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())

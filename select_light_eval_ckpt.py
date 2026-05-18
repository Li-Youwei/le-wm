"""Select a checkpoint using light rollout-eval logs.

This is the second stage after ``pick_best_ckpt.py``: the CE-based top-k
checkpoints are each evaluated with a small number of LIBERO episodes, then the
checkpoint with the highest 4-suite rollout success rate is selected for the
full evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

TASK_LINE_RE = re.compile(r"^\s*Task\s+(\d+):\s+(\d+)/(\d+)\s*\(\s*([\d.]+)%\)")


def parse_suite_log(log_path: Path) -> tuple[int, int] | None:
    """Return (successes, episodes) from one eval_libero.py log."""
    if not log_path.is_file():
        return None
    by_task: dict[int, tuple[int, int]] = {}
    with log_path.open() as f:
        for line in f:
            m = TASK_LINE_RE.match(line)
            if not m:
                continue
            task_id = int(m.group(1))
            by_task[task_id] = (int(m.group(2)), int(m.group(3)))
    if not by_task:
        return None
    succ = sum(row[0] for row in by_task.values())
    total = sum(row[1] for row in by_task.values())
    return succ, total


def load_candidates(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    seen: set[str] = set()
    candidates: list[dict] = []
    for row in data.get("all", []):
        ckpt = row.get("ckpt")
        if not ckpt or ckpt in seen:
            continue
        seen.add(ckpt)
        candidates.append(row)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--candidates-json", type=Path, required=True)
    parser.add_argument("--log-prefix", default="light_eval")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    candidates = load_candidates(args.candidates_json)
    if not candidates:
        print("ERROR: no candidate checkpoints in candidates JSON", file=sys.stderr)
        return 2

    results: list[dict] = []
    for rank, row in enumerate(candidates, start=1):
        ckpt = Path(row["ckpt"])
        stem = ckpt.stem
        suites: dict[str, dict] = {}
        succ_total = 0
        episode_total = 0
        missing: list[str] = []
        for suite in SUITES:
            parsed = parse_suite_log(args.ckpt_dir / f"{args.log_prefix}_{stem}_{suite}.log")
            if parsed is None:
                missing.append(suite)
                continue
            succ, total = parsed
            suites[suite] = {
                "successes": succ,
                "episodes": total,
                "rate": (succ / total * 100.0) if total else 0.0,
            }
            succ_total += succ
            episode_total += total
        complete = not missing and episode_total > 0
        results.append(
            {
                "rank": rank,
                "ckpt": str(ckpt),
                "ce_step": row.get("step"),
                "ce_value": row.get("value"),
                "complete": complete,
                "missing": missing,
                "successes": succ_total,
                "episodes": episode_total,
                "rate": (succ_total / episode_total * 100.0) if episode_total else 0.0,
                "suites": suites,
            }
        )

    complete = [row for row in results if row["complete"]]
    if not complete:
        print(
            json.dumps(
                {
                    "error": "no complete light-eval result found",
                    "candidates": results,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 3

    best = sorted(complete, key=lambda r: (-r["rate"], r["ce_value"], r["rank"]))[0]
    output = {
        "selection_metric": "light_eval_4suite_success_rate",
        "top_1": best["ckpt"],
        "top_1_rate": best["rate"],
        "top_1_successes": best["successes"],
        "top_1_episodes": best["episodes"],
        "all": results,
    }
    text = json.dumps(output, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

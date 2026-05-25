"""pick_best_ckpt.py — Select best checkpoint(s) from a training run.

Reads TB event files under ``<ckpt-dir>/tb_logs/vla_baseline/version_*/`` and
returns the steps where ``validate/ce_loss_taskbal`` (or fallback metric) is
minimized. Maps each step to the matching ``lewm_step_<N>_object.ckpt`` on
disk.

Usage:
    python pick_best_ckpt.py --ckpt-dir /data/lyw/stable-wm/all4_sp_sigreg_seed3072 --top-k 1

Output (JSON to stdout):
    {
      "metric": "validate/ce_loss_taskbal",
      "top_1": "/data/lyw/stable-wm/.../lewm_step_56000_object.ckpt",
      "top_1_value": 0.4123,
      "all": [
        {"rank": 1, "step": 56000, "value": 0.4123, "ckpt": "/data/lyw/.../lewm_step_56000_object.ckpt"},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CKPT_RE = re.compile(r"lewm_step_(\d+)_object\.ckpt$")


def find_event_files(ckpt_dir: Path) -> list[Path]:
    """Return all TB event files under <ckpt_dir>/tb_logs/vla_baseline/version_*/."""
    tb_root = ckpt_dir / "tb_logs" / "vla_baseline"
    if not tb_root.is_dir():
        return []
    events: list[Path] = []
    for version_dir in sorted(tb_root.glob("version_*")):
        events.extend(version_dir.glob("events.out.tfevents.*"))
    return sorted(events)


def load_scalar_series(event_files: list[Path], metric: str) -> list[tuple[int, float]]:
    """Return [(step, value), ...] for the metric across all event files (concat)."""
    out: list[tuple[int, float]] = []
    for ev in event_files:
        acc = EventAccumulator(str(ev.parent), size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        if metric not in tags:
            continue
        for evt in acc.Scalars(metric):
            out.append((int(evt.step), float(evt.value)))
    # Deduplicate by step (last wins).
    out_by_step: dict[int, float] = {}
    for step, val in out:
        out_by_step[step] = val
    return sorted(out_by_step.items(), key=lambda x: x[0])


def find_ckpt_for_step(ckpt_dir: Path, target_step: int) -> Path | None:
    """Find lewm_step_<N>_object.ckpt with N nearest to target_step."""
    available: list[tuple[int, Path]] = []
    for path in ckpt_dir.glob("lewm_step_*_object.ckpt"):
        m = CKPT_RE.search(path.name)
        if m:
            available.append((int(m.group(1)), path))
    if not available:
        return None
    available.sort(key=lambda x: abs(x[0] - target_step))
    return available[0][1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--metric", default="validate/ce_loss_taskbal")
    parser.add_argument(
        "--fallback-metric",
        default="validate/ce_loss_epoch",
        help="Used if --metric is absent from TB logs (e.g., legacy runs).",
    )
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=("min", "max"),
        default="min",
        help="min for losses, max for accuracy",
    )
    args = parser.parse_args()

    event_files = find_event_files(args.ckpt_dir)
    if not event_files:
        print(
            json.dumps({"error": f"No TB event files under {args.ckpt_dir}"}, indent=2),
            file=sys.stderr,
        )
        return 2

    metric = args.metric
    series = load_scalar_series(event_files, metric)
    if not series:
        metric = args.fallback_metric
        series = load_scalar_series(event_files, metric)
    if not series:
        print(
            json.dumps(
                {
                    "error": (
                        f"Metric '{args.metric}' (and fallback '{args.fallback_metric}') "
                        "not found in TB logs"
                    )
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 3

    # Skip step 0 (Lightning sanity-check val pass) and any non-positive values.
    series = [(s, v) for s, v in series if s > 0 and v == v]  # NaN-skip via v==v
    if not series:
        print(
            json.dumps({"error": "Series empty after filtering step>0"}, indent=2),
            file=sys.stderr,
        )
        return 4

    reverse = args.mode == "max"
    sorted_pts = sorted(series, key=lambda x: x[1], reverse=reverse)
    top = sorted_pts[: max(1, args.top_k)]

    entries = []
    for rank, (step, val) in enumerate(top, start=1):
        ckpt = find_ckpt_for_step(args.ckpt_dir, step)
        entries.append(
            {
                "rank": rank,
                "step": step,
                "value": val,
                "ckpt": str(ckpt) if ckpt else None,
            }
        )

    result = {
        "metric": metric,
        "mode": args.mode,
        "top_1": entries[0]["ckpt"] if entries else None,
        "top_1_step": entries[0]["step"] if entries else None,
        "top_1_value": entries[0]["value"] if entries else None,
        "all": entries,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

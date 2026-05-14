"""aggregate_all4_results.py — Parse 4-suite eval logs into a single summary.

Walks ``<ckpt-dir>/eval_libero_<suite>.log`` for each of the 4 suites, regex-
extracts the per-task success lines (e.g. ``Task  0:  18/20 ( 90.0%)``), sums
successes per suite, and writes a markdown table to stdout (and optionally a
file). Mirrors the regex used in ``run_sp_libero_spatial.sh:193``.

Usage:
    python aggregate_all4_results.py \\
        --ckpt-dir /Data/lyw/stable-wm/all4_sp_sigreg_seed3072 \\
        --out /Data/lyw/all4_sp_sigreg_seed3072.md
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

# Matches lines like: "  Task  0:  18/20 ( 90.0%) — task_name"
TASK_LINE_RE = re.compile(r"^\s*Task\s+(\d+):\s+(\d+)/(\d+)\s*\(\s*([\d.]+)%\)")


@dataclass
class SuiteResult:
    suite: str
    per_task: list[tuple[int, int, int, float, str]]  # (id, succ, total, rate%, name)
    succ: int
    total: int

    @property
    def rate(self) -> float:
        return (self.succ / self.total * 100.0) if self.total > 0 else 0.0


def parse_suite_log(log_path: Path, suite: str) -> SuiteResult | None:
    if not log_path.is_file():
        return None
    per_task: list[tuple[int, int, int, float, str]] = []
    succ = 0
    total = 0
    with log_path.open() as f:
        for line in f:
            m = TASK_LINE_RE.match(line)
            if not m:
                continue
            tid = int(m.group(1))
            s = int(m.group(2))
            t = int(m.group(3))
            r = float(m.group(4))
            # Best-effort: extract task name after the percentage, if present.
            tail = line[m.end() :].strip(" ——-:\n")
            per_task.append((tid, s, t, r, tail))
    # The per-task lines appear twice in eval_libero.py output: once mid-eval
    # and once in the SUMMARY block. Dedup by keeping the last occurrence per
    # tid (SUMMARY entries are more complete).
    by_tid: dict[int, tuple[int, int, int, float, str]] = {}
    for row in per_task:
        by_tid[row[0]] = row
    final = sorted(by_tid.values(), key=lambda r: r[0])
    succ = sum(r[1] for r in final)
    total = sum(r[2] for r in final)
    return SuiteResult(suite=suite, per_task=final, succ=succ, total=total)


def render_markdown(arm: str, seed: int, results: list[SuiteResult]) -> str:
    out: list[str] = []
    out.append(f"# 4-suite LIBERO results — arm={arm}, seed={seed}\n")
    out.append("## Summary\n")
    out.append("| suite          | succ / total | rate    |")
    out.append("|----------------|--------------|---------|")
    grand_s, grand_t = 0, 0
    for r in results:
        if r is None:
            continue
        out.append(
            f"| {r.suite:14s} | {r.succ:>3d} / {r.total:>3d}      | {r.rate:5.1f}% |"
        )
        grand_s += r.succ
        grand_t += r.total
    overall = (grand_s / grand_t * 100.0) if grand_t > 0 else 0.0
    out.append(f"| **overall**    | **{grand_s} / {grand_t}**  | **{overall:5.1f}%** |")
    out.append("")
    out.append("## Per-task breakdown\n")
    for r in results:
        if r is None:
            continue
        out.append(f"### {r.suite}\n")
        out.append("| task | succ / total | rate    | name |")
        out.append("|-----:|--------------|---------|------|")
        for tid, s, t, rate, name in r.per_task:
            out.append(
                f"| {tid:>3d} | {s:>2d} / {t:>2d}        | {rate:5.1f}% | {name} |"
            )
        out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--arm", default="sp_sigreg")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results: list[SuiteResult] = []
    missing: list[str] = []
    for suite in SUITES:
        log_path = args.ckpt_dir / f"eval_{suite}.log"
        r = parse_suite_log(log_path, suite)
        if r is None or r.total == 0:
            missing.append(suite)
            continue
        results.append(r)

    if missing:
        print(f"[WARN] missing or empty: {missing}", file=sys.stderr)
    if not results:
        print("ERROR: no eval logs parsed", file=sys.stderr)
        return 1

    md = render_markdown(args.arm, args.seed, results)
    print(md)
    if args.out is not None:
        args.out.write_text(md)
        print(f"[wrote] {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

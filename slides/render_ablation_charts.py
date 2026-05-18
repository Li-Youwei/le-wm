"""Render TensorBoard scalar charts for the 3-arm ablation comparison.

Reads TB event files from:
  - baseline (LN, no SP, no SIGReg)               1 run
  - SP+SIGReg                                     2 seeds
  - bn_only      (LN→BN switch alone)             3 seeds
  - sp_only_ln   (state pred, LN proj, no SIGReg) 3 seeds
  - sigreg_only  (SIGReg + BN, no SP)             3 seeds

Produces a fixed set of PNGs under slides/abl_charts/. Two chart families:

  1. Cross-arm panels — 1 panel per metric, 1 line per arm (mean of seeds
     where multiple seeds exist). Used for cross-arm comparison slides.
     Files: ce_loss.png, token_acc.png, total_loss.png, lr.png

  2. Within-arm panels — 1 panel per arm × metric, 1 line per seed.
     Used to read off seed-variance per arm.
     File: per_arm_grid.png  (4×3 subplots: 4 arms × 3 metrics)

  3. SP-specific — pred_loss components for arms that have it
     (sp_only_ln + sp+sigreg).
     File: pred_loss.png

  4. SIGReg-specific — sigreg_loss for arms that have it
     (sigreg_only + sp+sigreg).
     File: sigreg_loss.png

Run locally; reads from slides/tb_events/<run_name>/version_*.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---------------------------------------------------------------------------
# Run inventory — local TB-events tree under slides/tb_events/<run>/version_*
# ---------------------------------------------------------------------------
TB_ROOT = Path(__file__).parent / "tb_events"

# (display_name, run_subdir, arm_key, seed_label)
RUNS: list[tuple[str, str, str, str]] = [
    ("baseline",            "baseline",                          "baseline",     "—"),
    ("SP+SIGReg s=1234",    "sp_libero_spatial_joint_seed1234",  "sp_sigreg",    "1234"),
    ("SP+SIGReg s=3072",    "sp_libero_spatial_joint_seed3072",  "sp_sigreg",    "3072"),
    ("bn_only s=2024",      "abl_bn_only_seed2024",              "bn_only",      "2024"),
    ("bn_only s=1234",      "abl_bn_only_seed1234",              "bn_only",      "1234"),
    ("bn_only s=3072",      "abl_bn_only_seed3072",              "bn_only",      "3072"),
    ("sp_only_ln s=2024",   "abl_sp_only_ln_seed2024",           "sp_only_ln",   "2024"),
    ("sp_only_ln s=1234",   "abl_sp_only_ln_seed1234",           "sp_only_ln",   "1234"),
    ("sp_only_ln s=3072",   "abl_sp_only_ln_seed3072",           "sp_only_ln",   "3072"),
    ("sigreg_only s=2024",  "abl_sigreg_only_seed2024",          "sigreg_only",  "2024"),
    ("sigreg_only s=1234",  "abl_sigreg_only_seed1234",          "sigreg_only",  "1234"),
    ("sigreg_only s=3072",  "abl_sigreg_only_seed3072",          "sigreg_only",  "3072"),
]

# Arm color scheme — light theme matching slides/generate_*_slides.js.
ARM_COLORS: dict[str, str] = {
    "baseline":    "#64748B",   # slate
    "bn_only":     "#DC2626",   # red
    "sp_only_ln":  "#059669",   # green
    "sigreg_only": "#2563EB",   # blue
    "sp_sigreg":   "#7C3AED",   # violet
}
ARM_LABEL: dict[str, str] = {
    "baseline":    "baseline (LN)",
    "bn_only":     "bn_only",
    "sp_only_ln":  "sp_only_ln",
    "sigreg_only": "sigreg_only",
    "sp_sigreg":   "sp+sigreg",
}
ARMS_ORDER = ["baseline", "bn_only", "sp_only_ln", "sigreg_only", "sp_sigreg"]

# Style: solid for train, dashed for val, identical color across train/val.
LINESTYLE = {"train": "-", "val": "--"}

OUT_DIR = Path(__file__).parent / "abl_charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

Series = list[tuple[int, float]]


def load_scalars(run_subdir: str) -> dict[str, Series]:
    """Find the latest version_N under slides/tb_events/<subdir> and load
    every scalar tag present. Returns {tag: [(step, value), ...]}."""
    base = TB_ROOT / run_subdir
    if not base.exists():
        return {}
    versions = sorted(p for p in base.glob("vla_baseline/version_*") if p.is_dir())
    if not versions:
        # Some local rsync setups skip the vla_baseline/ middle dir.
        versions = sorted(p for p in base.glob("version_*") if p.is_dir())
    if not versions:
        return {}
    acc = EventAccumulator(str(versions[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    out: dict[str, Series] = {}
    for tag in acc.Tags()["scalars"]:
        evs = acc.Scalars(tag)
        out[tag] = [(e.step, e.value) for e in evs]
    return out


def setup_axes(ax, title: str, ylabel: str, xlabel: str = "global step") -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=9, color="#64748B")
    ax.set_ylabel(ylabel, fontsize=9, color="#64748B")
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.tick_params(labelsize=8, color="#94A3B8")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#CBD5E1")


def mean_curve(curves: list[Series]) -> Series:
    """Average several (step, value) curves at their common steps. Each
    seed has the same step grid (same epoch count), so we can stack."""
    if not curves:
        return []
    # Trim to shortest curve (defensive against partial runs).
    min_len = min(len(c) for c in curves)
    if min_len == 0:
        return []
    steps = [c[0] for c in curves[0][:min_len]]
    arr = np.array([[c[i][1] for i in range(min_len)] for c in curves])
    means = arr.mean(axis=0).tolist()
    return list(zip(steps, means))


# ---------------------------------------------------------------------------
# Chart 1: cross-arm comparison — one line per arm, average across seeds
# ---------------------------------------------------------------------------
def plot_cross_arm(
    runs_data: dict[str, tuple[str, dict[str, Series]]],
    train_tag: str,
    val_tag: str | None,
    title: str,
    ylabel: str,
    out_name: str,
    yscale: str = "linear",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=160)

    # Group runs by arm. RUNS is the canonical (display, sub, arm, seed)
    # mapping; runs_data is keyed by display.
    by_arm: dict[str, list[dict[str, Series]]] = defaultdict(list)
    for display, _sub, arm, _seed in RUNS:
        entry = runs_data.get(display)
        if entry is None:
            continue
        _stored_sub, scalars = entry
        by_arm[arm].append(scalars)

    for arm in ARMS_ORDER:
        if arm not in by_arm:
            continue
        color = ARM_COLORS[arm]
        train_curves = [s[train_tag] for s in by_arm[arm] if train_tag in s]
        if train_curves:
            xs, ys = zip(*mean_curve(train_curves))
            ax.plot(xs, ys, color=color, linestyle=LINESTYLE["train"], linewidth=1.7,
                    label=f"{ARM_LABEL[arm]} (train)")
        if val_tag:
            val_curves = [s[val_tag] for s in by_arm[arm] if val_tag in s]
            if val_curves:
                xs, ys = zip(*mean_curve(val_curves))
                ax.plot(xs, ys, color=color, linestyle=LINESTYLE["val"], linewidth=1.5,
                        alpha=0.85, label=f"{ARM_LABEL[arm]} (val)")

    setup_axes(ax, title, ylabel)
    if yscale != "linear":
        ax.set_yscale(yscale)
    ax.legend(loc="best", fontsize=7.5, frameon=True, framealpha=0.9, ncol=2)
    fig.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote: {out_path}")


# ---------------------------------------------------------------------------
# Chart 2: within-arm grid — one panel per (arm × metric), seed-level
# ---------------------------------------------------------------------------
def plot_per_arm_grid(runs_data: dict[str, tuple[str, dict[str, Series]]]) -> None:
    """4 rows (one per ablation arm + baseline-fold) × 3 cols (CE / acc / total)."""
    metrics = [
        ("fit/ce_loss_epoch",       "validate/ce_loss_epoch",       "L_CE",     "linear"),
        ("fit/token_accuracy_epoch","validate/token_accuracy_epoch","token acc","linear"),
        ("fit/total_loss_epoch",    "validate/total_loss_epoch",    "L_total",  "linear"),
    ]
    arms = ["bn_only", "sp_only_ln", "sigreg_only", "sp_sigreg"]
    fig, axes = plt.subplots(len(arms), len(metrics),
                             figsize=(11.5, 2.6 * len(arms)), dpi=160)

    # Build per-arm group with seed labels.
    by_arm: dict[str, list[tuple[str, dict[str, Series]]]] = defaultdict(list)
    base_scalars: dict[str, Series] | None = None
    for display, sub, arm, seed in RUNS:
        scalars = runs_data.get(display, (None, None))[1]
        if scalars is None:
            continue
        if arm == "baseline":
            base_scalars = scalars
        else:
            by_arm[arm].append((seed, scalars))

    seed_styles = {"2024": "-", "1234": "--", "3072": ":"}

    for r, arm in enumerate(arms):
        color = ARM_COLORS[arm]
        for c, (train_tag, val_tag, ylabel, yscale) in enumerate(metrics):
            ax = axes[r, c]
            # Baseline overlay (gray) for reference.
            if base_scalars is not None and train_tag in base_scalars:
                xs, ys = zip(*base_scalars[train_tag])
                ax.plot(xs, ys, color=ARM_COLORS["baseline"], linestyle="-",
                        linewidth=1.3, alpha=0.55, label="baseline (train)")
                if val_tag in base_scalars:
                    xs, ys = zip(*base_scalars[val_tag])
                    ax.plot(xs, ys, color=ARM_COLORS["baseline"], linestyle="--",
                            linewidth=1.1, alpha=0.55, label="baseline (val)")
            # This arm's seeds.
            for seed, scalars in by_arm.get(arm, []):
                ls = seed_styles.get(seed, "-")
                if train_tag in scalars:
                    xs, ys = zip(*scalars[train_tag])
                    ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.3,
                            label=f"s={seed} (train)")
                if val_tag and val_tag in scalars:
                    xs, ys = zip(*scalars[val_tag])
                    ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.1,
                            alpha=0.7)
            title = f"{ARM_LABEL[arm]} · {ylabel}" if c == 0 else ylabel
            setup_axes(ax, title, ylabel)
            if yscale != "linear":
                ax.set_yscale(yscale)
            if r == 0 and c == len(metrics) - 1:
                ax.legend(loc="best", fontsize=6.5, frameon=True, framealpha=0.9)

    fig.tight_layout()
    out_path = OUT_DIR / "per_arm_grid.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote: {out_path}")


# ---------------------------------------------------------------------------
# Chart 3: pred_loss components for arms that have SP (sp_only_ln, sp_sigreg)
# ---------------------------------------------------------------------------
def plot_pred_loss(runs_data: dict[str, tuple[str, dict[str, Series]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.5), dpi=160, sharey=True)
    components = [
        ("fit/pred_loss_ag_epoch", "agent latent",  "-"),
        ("fit/pred_loss_hd_epoch", "hand latent",   "--"),
        ("fit/pred_loss_pr_epoch", "proprio (raw)", ":"),
    ]
    panels: list[tuple[str, list[tuple[str, dict[str, Series]]]]] = [
        ("sp_only_ln (no SIGReg)", []),
        ("sp+sigreg",              []),
    ]
    for display, sub, arm, seed in RUNS:
        scalars = runs_data.get(display, (None, None))[1]
        if scalars is None:
            continue
        if arm == "sp_only_ln":
            panels[0][1].append((seed, scalars))
        elif arm == "sp_sigreg":
            panels[1][1].append((seed, scalars))

    for ax, (title, group) in zip(axes, panels):
        arm_color = ARM_COLORS["sp_only_ln" if "sp_only_ln" in title else "sp_sigreg"]
        for seed, scalars in group:
            seed_alpha = {"2024": 1.0, "1234": 0.75, "3072": 0.50}.get(seed, 0.7)
            for tag, label, ls in components:
                if tag not in scalars:
                    continue
                xs, ys = zip(*scalars[tag])
                ax.plot(xs, ys, color=arm_color, linestyle=ls, linewidth=1.3,
                        alpha=seed_alpha, label=f"s={seed} · {label}")
        setup_axes(ax, title, "MSE")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=6.5, frameon=True, framealpha=0.9, ncol=2)

    fig.tight_layout()
    out_path = OUT_DIR / "pred_loss.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote: {out_path}")


# ---------------------------------------------------------------------------
# Chart 4: sigreg_loss for arms that have it (sigreg_only, sp_sigreg)
# ---------------------------------------------------------------------------
def plot_sigreg_loss(runs_data: dict[str, tuple[str, dict[str, Series]]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=160)
    seed_alpha = {"2024": 1.0, "1234": 0.75, "3072": 0.50}
    for display, sub, arm, seed in RUNS:
        scalars = runs_data.get(display, (None, None))[1]
        if scalars is None or arm not in ("sigreg_only", "sp_sigreg"):
            continue
        if "fit/sigreg_loss_epoch" not in scalars:
            continue
        xs, ys = zip(*scalars["fit/sigreg_loss_epoch"])
        color = ARM_COLORS[arm]
        alpha = seed_alpha.get(seed, 0.7)
        ax.plot(xs, ys, color=color, linewidth=1.4, alpha=alpha,
                label=f"{ARM_LABEL[arm]} s={seed}")

    setup_axes(ax, "L_sigreg  (anti-collapse on encoder outputs)", "sigreg loss")
    ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.9)
    fig.tight_layout()
    out_path = OUT_DIR / "sigreg_loss.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading TB event files from {TB_ROOT} ...")
    runs_data: dict[str, tuple[str, dict[str, Series]]] = {}
    for display, sub, arm, seed in RUNS:
        scalars = load_scalars(sub)
        if not scalars:
            print(f"  skip {display}: no events under {TB_ROOT / sub}")
            continue
        runs_data[display] = (sub, scalars)
        print(f"  {display:24s}  {len(scalars)} tags")

    if not runs_data:
        print("ERROR: no TB events loaded")
        return

    print("\nRendering cross-arm charts...")
    plot_cross_arm(runs_data,
                   train_tag="fit/ce_loss_epoch", val_tag="validate/ce_loss_epoch",
                   title="L_CE  (action token cross-entropy) — arm mean across seeds",
                   ylabel="CE loss",
                   out_name="ce_loss.png")
    plot_cross_arm(runs_data,
                   train_tag="fit/token_accuracy_epoch",
                   val_tag="validate/token_accuracy_epoch",
                   title="Token accuracy — arm mean across seeds",
                   ylabel="accuracy",
                   out_name="token_acc.png")
    plot_cross_arm(runs_data,
                   train_tag="fit/total_loss_epoch", val_tag="validate/total_loss_epoch",
                   title="Total loss = L_CE + pred_w·L_pred + sigreg_w·L_sigreg",
                   ylabel="total loss",
                   out_name="total_loss.png")
    plot_cross_arm(runs_data,
                   train_tag="fit/lr_epoch", val_tag=None,
                   title="Learning rate schedule (all arms share)",
                   ylabel="lr",
                   out_name="lr.png",
                   yscale="log")

    print("\nRendering per-arm seed-variance grid...")
    plot_per_arm_grid(runs_data)

    print("\nRendering SP-specific charts...")
    plot_pred_loss(runs_data)

    print("\nRendering SIGReg-specific charts...")
    plot_sigreg_loss(runs_data)

    print(f"\nAll charts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

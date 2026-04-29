"""Render TensorBoard scalar charts for SP comparison PPT.

Reads TB event files from baseline + 2 SP seeds, overlays curves with
TensorBoard-like styling, saves PNGs to slides/sp_charts/.
Run on the GPU server (where TB logs live), then SCP PNGs to local.

Charts produced:
  ce_loss.png       — train/val CE for baseline + SP-3072 + SP-1234
  token_acc.png     — train/val token accuracy for all 3
  pred_loss.png     — SP-only L_pred breakdown (ag/hd/pr) for both seeds
  sigreg_loss.png   — SP-only L_sigreg for both seeds
  lr.png            — learning rate schedule (all 3 share schedule)
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Hard-coded paths on the GPU server (server-51:/Data/lyw/...)
RUNS = {
    "baseline":  "/Data/lyw/checkpoints/multitask_ln_100ep/tb_logs/vla_baseline/version_0",
    "SP s=3072": "/Data/lyw/stable-wm/sp_libero_spatial_joint_seed3072/tb_logs/vla_baseline/version_0",
    "SP s=1234": "/Data/lyw/stable-wm/sp_libero_spatial_joint_seed1234/tb_logs/vla_baseline/version_0",
}

# Color scheme matching slides/generate_results_slides.js (light theme)
COLORS = {
    "baseline":  "#64748B",   # slate (textDim)
    "SP s=3072": "#2563EB",   # blue (accent)
    "SP s=1234": "#EA580C",   # orange
}
LINESTYLE = {
    "train": "-",
    "val":   "--",
}

OUT_DIR = Path("/tmp/sp_charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_scalars(run_path: str) -> dict[str, list[tuple[int, float]]]:
    """Load all scalar tags from a TB events directory."""
    acc = EventAccumulator(run_path, size_guidance={"scalars": 0})
    acc.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in acc.Tags()["scalars"]:
        evs = acc.Scalars(tag)
        out[tag] = [(e.step, e.value) for e in evs]
    return out


def setup_axes(ax, title: str, ylabel: str, xlabel: str = "global step") -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10, color="#64748B")
    ax.set_ylabel(ylabel, fontsize=10, color="#64748B")
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.tick_params(labelsize=9, color="#94A3B8")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#CBD5E1")


def plot_metric(
    runs_data: dict[str, dict],
    train_tag: str,
    val_tag: str | None,
    title: str,
    ylabel: str,
    out_name: str,
    runs_to_plot: list[str] | None = None,
    yscale: str = "linear",
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=160)
    runs_to_plot = runs_to_plot or list(runs_data.keys())
    for run_name in runs_to_plot:
        scalars = runs_data[run_name]
        color = COLORS[run_name]
        if train_tag in scalars:
            xs, ys = zip(*scalars[train_tag])
            ax.plot(xs, ys, color=color, linestyle=LINESTYLE["train"], linewidth=1.6,
                    label=f"{run_name} (train)")
        if val_tag and val_tag in scalars:
            xs, ys = zip(*scalars[val_tag])
            ax.plot(xs, ys, color=color, linestyle=LINESTYLE["val"], linewidth=1.6,
                    alpha=0.8, label=f"{run_name} (val)")
    setup_axes(ax, title, ylabel)
    if yscale != "linear":
        ax.set_yscale(yscale)
    ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.9)
    fig.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote: {out_path}")


def plot_pred_loss_breakdown(runs_data: dict[str, dict], out_name: str) -> None:
    """Per-component pred_loss for SP runs (3 components × 2 seeds = 6 lines)."""
    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=160)
    components = [
        ("fit/pred_loss_ag_epoch", "agent latent", "-"),
        ("fit/pred_loss_hd_epoch", "hand latent",  "--"),
        ("fit/pred_loss_pr_epoch", "proprio (raw)", ":"),
    ]
    for run_name in ("SP s=3072", "SP s=1234"):
        color = COLORS[run_name]
        for tag, label, ls in components:
            if tag in runs_data[run_name]:
                xs, ys = zip(*runs_data[run_name][tag])
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.4,
                        label=f"{run_name} · {label}")
    setup_axes(ax, "L_pred breakdown (3 components × 2 seeds)", "MSE")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=7.5, frameon=True, framealpha=0.9, ncol=2)
    fig.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote: {out_path}")


def main() -> None:
    print("Loading TB event files...")
    runs_data: dict[str, dict] = {}
    for name, path in RUNS.items():
        if not os.path.isdir(path):
            print(f"  skip {name}: not found at {path}")
            continue
        runs_data[name] = load_scalars(path)
        n_tags = len(runs_data[name])
        print(f"  {name}: {n_tags} scalar tags")

    print("\nRendering charts...")

    # 1. CE loss (all 3 runs, train + val)
    plot_metric(
        runs_data,
        train_tag="fit/ce_loss_epoch", val_tag="validate/ce_loss_epoch",
        title="L_CE  (action token cross-entropy)",
        ylabel="CE loss",
        out_name="ce_loss.png",
    )

    # 2. Token accuracy (all 3 runs)
    plot_metric(
        runs_data,
        train_tag="fit/token_accuracy_epoch", val_tag="validate/token_accuracy_epoch",
        title="Token accuracy  (top-1 match, PAD/prefix excluded)",
        ylabel="accuracy",
        out_name="token_acc.png",
    )

    # 3. Total loss (all 3, train+val)
    plot_metric(
        runs_data,
        train_tag="fit/total_loss_epoch", val_tag="validate/total_loss_epoch",
        title="Total loss  (L_CE + pred_w · L_pred + sigreg_w · L_sigreg)",
        ylabel="total loss",
        out_name="total_loss.png",
    )

    # 4. Pred loss breakdown (SP only)
    plot_pred_loss_breakdown(runs_data, "pred_loss.png")

    # 5. SIGReg loss (SP only)
    plot_metric(
        runs_data,
        train_tag="fit/sigreg_loss_epoch", val_tag=None,
        title="L_sigreg  (anti-collapse on encoder outputs)",
        ylabel="sigreg loss",
        out_name="sigreg_loss.png",
        runs_to_plot=["SP s=3072", "SP s=1234"],
    )

    # 6. LR schedule (all 3 — should overlap perfectly since same schedule)
    plot_metric(
        runs_data,
        train_tag="fit/lr_epoch", val_tag=None,
        title="Learning rate schedule",
        ylabel="lr",
        out_name="lr.png",
        yscale="log",
    )

    print(f"\nAll charts in: {OUT_DIR}")


if __name__ == "__main__":
    main()

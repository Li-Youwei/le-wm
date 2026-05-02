#!/usr/bin/env bash
# =============================================================================
# eval_videos.sh — re-run eval ONLY (no training) on an existing ablation
# checkpoint with --save-videos to produce mp4 files for the PPT.
#
# The first ablation pass (run_ablation.sh) ran eval without --save-videos so
# only per-task pass/fail counts were captured. Training is fully done and
# best-val ckpts are kept on disk; we re-eval from the same best-val ckpt
# with the --save-videos flag.
#
# Usage:
#   ARM=bn_only      SEED=2024 CUDA_VISIBLE_DEVICES=1 bash eval_videos.sh
#   ARM=sp_only_ln   SEED=1234 CUDA_VISIBLE_DEVICES=3 bash eval_videos.sh
#   ARM=sigreg_only  SEED=3072 CUDA_VISIBLE_DEVICES=1 bash eval_videos.sh
# =============================================================================
set -uo pipefail

ARM="${ARM:?must set ARM=bn_only|sp_only_ln|sigreg_only}"
SEED="${SEED:?must set SEED=<int>}"

case "$ARM" in
    bn_only|sp_only_ln|sigreg_only) ;;
    *) echo "ARM must be one of: bn_only | sp_only_ln | sigreg_only (got '$ARM')" >&2; exit 2 ;;
esac

DATA_ROOT="/Data/lyw"
TOKENIZER="${DATA_ROOT}/fast_tokenizer"
PROCESSED_DIR="${DATA_ROOT}/libero_processed_v4/libero_spatial"

ARM_TAG="abl_${ARM}_seed${SEED}"
CKPT_DIR="${DATA_ROOT}/stable-wm/${ARM_TAG}"
VIDEO_DIR="${DATA_ROOT}/abl_videos/${ARM_TAG}"
EVAL_LOG="${DATA_ROOT}/${ARM_TAG}_eval_videos.log"

NUM_EPISODES=20
MAX_STEPS=300

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [ ! -d "$CKPT_DIR" ]; then
    echo "[ERR] ckpt dir not found: $CKPT_DIR" >&2; exit 1
fi
mkdir -p "$VIDEO_DIR"

log()  { echo -e "\n$(date '+%H:%M:%S') [INFO] $*"; }
err()  { echo -e "\n$(date '+%H:%M:%S') [ERR ] $*" >&2; }

# -----------------------------------------------------------------------------
# Pick best-val ckpt — same logic as run_ablation.sh step 3, but read-only.
# Walks the run's TB scalars, finds the lowest validate/total_loss_epoch (or
# ce_loss_epoch fallback), maps to nearest existing epoch ckpt.
# -----------------------------------------------------------------------------
log "===== finding best-val ckpt for $ARM_TAG ====="
best_epoch=$(python << PYEOF
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
versions = sorted(glob.glob("$CKPT_DIR/tb_logs/vla_baseline/version_*"))
if not versions:
    print(""); raise SystemExit
acc = EventAccumulator(versions[-1], size_guidance={"scalars": 0})
acc.Reload()
key = "validate/total_loss_epoch"
if key not in acc.Tags()["scalars"]:
    key = "validate/ce_loss_epoch"
    if key not in acc.Tags()["scalars"]:
        print(""); raise SystemExit
evs = acc.Scalars(key)
if not evs:
    print(""); raise SystemExit
best = min(evs, key=lambda e: e.value)
ckpts = sorted([int(os.path.basename(p).split("_")[2])
                for p in glob.glob("$CKPT_DIR/lewm_epoch_*_object.ckpt")])
if not ckpts:
    print(""); raise SystemExit
step_per_epoch = max(1, evs[-1].step // max(ckpts))
best_epoch_guess = max(1, best.step // step_per_epoch)
best_epoch = min(ckpts, key=lambda e: abs(e - best_epoch_guess))
print(best_epoch)
PYEOF
)
if [ -z "$best_epoch" ]; then
    err "could not pick best-val epoch from TB events for $CKPT_DIR"
    exit 1
fi
best_ckpt="$CKPT_DIR/lewm_epoch_${best_epoch}_object.ckpt"
if [ ! -f "$best_ckpt" ]; then
    err "best-val ckpt not on disk: $best_ckpt"
    exit 1
fi
log "  using best-val epoch=$best_epoch  ($best_ckpt)"
log "  videos → $VIDEO_DIR"

# -----------------------------------------------------------------------------
# Run eval with video saving. Output mp4 filenames carry the task name so we
# can post-organize into per-task subdirs after eval finishes.
# -----------------------------------------------------------------------------
log "===== eval ($NUM_EPISODES ep × 10 tasks  +  videos) ====="
if ! python eval_libero.py \
    --checkpoint "$best_ckpt" \
    --tokenizer "$TOKENIZER" \
    --processed-dir "$PROCESSED_DIR" \
    --suite libero_spatial \
    --num-episodes "$NUM_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --device cuda \
    --save-videos \
    --video-dir "$VIDEO_DIR" 2>&1 | tee "$EVAL_LOG" | tail -50; then
    err "eval FAILED — see $EVAL_LOG"
    exit 1
fi

# -----------------------------------------------------------------------------
# Post-organize: regroup the flat list of mp4s into per-task subfolders so the
# directory structure matches the existing main PPT layout
# (sp_videos/seed3072/<task_dir>/<file>.mp4).
# -----------------------------------------------------------------------------
log "===== organizing videos into per-task subdirs ====="
n_total=0; n_moved=0
for mp4 in "$VIDEO_DIR"/*.mp4; do
    [ -f "$mp4" ] || continue
    n_total=$((n_total + 1))
    base=$(basename "$mp4")
    # Strip "_epN_(success|fail).mp4" suffix to recover the task dir name.
    task_dir=$(echo "$base" | sed -E 's/_ep[0-9]+_(success|fail)\.mp4$//')
    [ -n "$task_dir" ] || continue
    mkdir -p "$VIDEO_DIR/$task_dir"
    mv "$mp4" "$VIDEO_DIR/$task_dir/$base"
    n_moved=$((n_moved + 1))
done
log "  organized: $n_moved / $n_total mp4 moved into per-task dirs"

n_dirs=$(find "$VIDEO_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)
n_files=$(find "$VIDEO_DIR" -mindepth 2 -name "*.mp4" | wc -l)
log "===== done: $n_dirs task dirs · $n_files mp4 files in $VIDEO_DIR ====="

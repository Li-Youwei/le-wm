#!/usr/bin/env bash
# =============================================================================
# run_ablation.sh — single-arm × single-seed ablation runner for libero_spatial
#
# Three ablation arms isolating which part of SP+SIGReg actually helps and
# which part causes T5 collapse vs the LN-baseline (frozen 7008f15):
#
#   ARM=bn_only       projector=batch  pred_w=0      sigreg_w=0     (LN→BN switch alone)
#   ARM=sp_only_ln    projector=layer  pred_w=1.0    sigreg_w=0     (SP without BN/SIGReg)
#   ARM=sigreg_only   projector=batch  pred_w=0      sigreg_w=0.1   (SIGReg + BN, no SP)
#
# Companion arms already on disk (don't re-run):
#   baseline         /Data/lyw/checkpoints/multitask_ln_100ep/      (frozen)
#   sp_sigreg s=1234 /Data/lyw/stable-wm/sp_libero_spatial_joint_seed1234/
#   sp_sigreg s=3072 /Data/lyw/stable-wm/sp_libero_spatial_joint_seed3072/
#
# Each run: preprocess (skip if exists) → train 100 ep on all 10 spatial tasks
# → keep best-val + last → eval 20 episodes × 10 tasks → log per-task results.
#
# Usage:
#   ARM=bn_only       SEED=42   CUDA_VISIBLE_DEVICES=1  bash run_ablation.sh
#   ARM=sp_only_ln    SEED=1234 CUDA_VISIBLE_DEVICES=3  bash run_ablation.sh
#   ARM=sigreg_only   SEED=3072 CUDA_VISIBLE_DEVICES=1  bash run_ablation.sh
# =============================================================================
set -uo pipefail

# -----------------------------------------------------------------------------
# Required inputs
# -----------------------------------------------------------------------------
ARM="${ARM:?must set ARM=bn_only|sp_only_ln|sigreg_only}"
SEED="${SEED:?must set SEED=<int>}"

case "$ARM" in
    bn_only)      NORM_TYPE=batch; PRED_WEIGHT=0.0; SIGREG_WEIGHT=0.0 ;;
    sp_only_ln)   NORM_TYPE=layer; PRED_WEIGHT=1.0; SIGREG_WEIGHT=0.0 ;;
    sigreg_only)  NORM_TYPE=batch; PRED_WEIGHT=0.0; SIGREG_WEIGHT=0.1 ;;
    *) echo "ARM must be one of: bn_only | sp_only_ln | sigreg_only (got '$ARM')" >&2; exit 2 ;;
esac

# -----------------------------------------------------------------------------
# Paths (mirror run_sp_libero_spatial.sh — same v4 preprocessing, same
# tokenizer; only the output dirs are arm/seed-specific).
# -----------------------------------------------------------------------------
RAW_ROOT="/nas_data_new/caz/data_ssd/libero/libero_spatial"
DATA_ROOT="/Data/lyw"
TOKENIZER="${DATA_ROOT}/fast_tokenizer"
PROCESSED_DIR="${DATA_ROOT}/libero_processed_v4/libero_spatial"

ARM_TAG="abl_${ARM}_seed${SEED}"
CKPT_DIR="${DATA_ROOT}/stable-wm/${ARM_TAG}"
RESULTS_LOG="${DATA_ROOT}/${ARM_TAG}_results.txt"
EVAL_LOG="${DATA_ROOT}/${ARM_TAG}_eval.log"

NUM_EPISODES=20
MAX_STEPS=300
CHUNK_SIZE=20
CHUNK_STRIDE=1
MAX_EPOCHS=100
BATCH_SIZE=128

# Default GPU 1; override with CUDA_VISIBLE_DEVICES=N.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

mkdir -p "$CKPT_DIR"
echo "==== ABLATION RUN ====" | tee "$RESULTS_LOG"
echo "arm=$ARM  seed=$SEED  GPU=$CUDA_VISIBLE_DEVICES" | tee -a "$RESULTS_LOG"
echo "norm=$NORM_TYPE  pred_w=$PRED_WEIGHT  sigreg_w=$SIGREG_WEIGHT" | tee -a "$RESULTS_LOG"
echo "ckpt_dir=$CKPT_DIR" | tee -a "$RESULTS_LOG"
echo "started: $(date)" | tee -a "$RESULTS_LOG"

log()  { echo -e "\n$(date '+%H:%M:%S') [INFO] $*"  | tee -a "$RESULTS_LOG"; }
err()  { echo -e "\n$(date '+%H:%M:%S') [ERR ] $*"  | tee -a "$RESULTS_LOG" >&2; }

# -----------------------------------------------------------------------------
# Step 1: confirm v4 preprocessing already exists (run_sp_libero_spatial.sh
# was the producer; we don't re-run it from this script — fail loudly if
# data is missing so we don't silently re-preprocess in an ablation run).
# -----------------------------------------------------------------------------
n_proc=$(ls "$PROCESSED_DIR"/*.h5 2>/dev/null | wc -l)
if [ "$n_proc" -lt 10 ]; then
    err "expected 10 preprocessed files in $PROCESSED_DIR, got $n_proc"
    err "run \`bash run_sp_libero_spatial.sh\` once to populate it (preprocessing is shared)"
    exit 1
fi
log "found $n_proc preprocessed files in $PROCESSED_DIR (shared across arms)"

# -----------------------------------------------------------------------------
# Step 2: train (skip if final ckpt already exists)
# -----------------------------------------------------------------------------
log "===== Step 2: train  arm=$ARM  seed=$SEED ====="
if [ -f "$CKPT_DIR/lewm_weights.ckpt" ]; then
    log "  ckpt already exists, skipping training: $CKPT_DIR/lewm_weights.ckpt"
else
    export STABLEWM_HOME="$CKPT_DIR"
    if ! python train.py \
        data=libero \
        data.dataset.hdf5_dir="$PROCESSED_DIR" \
        loss.pred_weight="$PRED_WEIGHT" \
        loss.sigreg_weight="$SIGREG_WEIGHT" \
        projector.norm_type="$NORM_TYPE" \
        trainer.devices=1 \
        loader.batch_size="$BATCH_SIZE" \
        trainer.max_epochs="$MAX_EPOCHS" \
        seed="$SEED" \
        subdir="" \
        output_model_name=lewm 2>&1 | tail -40; then
        err "training FAILED — aborting"
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Step 3: pick best-val epoch from TB events; keep best + last only.
# (Same logic as run_sp_libero_spatial.sh.)
# -----------------------------------------------------------------------------
log "===== Step 3: pick best-val ckpt + cleanup ====="
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
ckpts = sorted([int(os.path.basename(p).split("_")[2]) for p in glob.glob("$CKPT_DIR/lewm_epoch_*_object.ckpt")])
if not ckpts:
    print(""); raise SystemExit
step_per_epoch = max(1, evs[-1].step // max(ckpts))
best_epoch_guess = max(1, best.step // step_per_epoch)
best_epoch = min(ckpts, key=lambda e: abs(e - best_epoch_guess))
print(best_epoch)
PYEOF
)
last_epoch=$(ls "$CKPT_DIR"/lewm_epoch_*_object.ckpt 2>/dev/null | \
    sed 's/.*lewm_epoch_\([0-9]*\)_object.ckpt/\1/' | sort -n | tail -1)
if [ -z "$best_epoch" ]; then
    err "  could not infer best-val epoch; falling back to last_epoch=$last_epoch"
    best_epoch="$last_epoch"
fi
log "  best-val epoch = $best_epoch  (last = $last_epoch)"
for ckpt in "$CKPT_DIR"/lewm_epoch_*_object.ckpt; do
    [ -f "$ckpt" ] || continue
    ep=$(echo "$ckpt" | sed 's/.*lewm_epoch_\([0-9]*\)_object.ckpt/\1/')
    if [ "$ep" != "$best_epoch" ] && [ "$ep" != "$last_epoch" ]; then
        rm -f "$ckpt"
    fi
done
best_ckpt="$CKPT_DIR/lewm_epoch_${best_epoch}_object.ckpt"

# -----------------------------------------------------------------------------
# Step 4: eval 20 episodes × 10 tasks on the best-val ckpt
# -----------------------------------------------------------------------------
log "===== Step 4: eval  ($NUM_EPISODES ep × 10 tasks) ====="
if [ ! -f "$best_ckpt" ]; then
    err "best ckpt $best_ckpt not found — eval skipped"
    exit 1
fi

if ! python eval_libero.py \
    --checkpoint "$best_ckpt" \
    --tokenizer "$TOKENIZER" \
    --processed-dir "$PROCESSED_DIR" \
    --suite libero_spatial \
    --num-episodes "$NUM_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --device cuda 2>&1 | tee "$EVAL_LOG" | tail -50; then
    err "eval FAILED — see $EVAL_LOG"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 5: per-task summary into RESULTS_LOG
# -----------------------------------------------------------------------------
echo "" | tee -a "$RESULTS_LOG"
echo "==== PER-TASK RESULTS  ($ARM seed=$SEED) ====" | tee -a "$RESULTS_LOG"
grep -E "^[[:space:]]+Task[[:space:]]+[0-9]+:" "$EVAL_LOG" | tee -a "$RESULTS_LOG"
echo "" | tee -a "$RESULTS_LOG"
echo "==== OVERALL ====" | tee -a "$RESULTS_LOG"
grep -E "Overall:" "$EVAL_LOG" | tee -a "$RESULTS_LOG"
echo "==== finished: $(date) ====" | tee -a "$RESULTS_LOG"

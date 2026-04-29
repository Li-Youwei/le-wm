#!/usr/bin/env bash
# =============================================================================
# run_sp_libero_spatial.sh — JOINT SP training + per-task eval on libero_spatial
#
# Mirrors how the frozen baseline at commit 7008f15 was actually produced:
#   - Single training run on the WHOLE libero_spatial suite (10 tasks, ~65k
#     chunks). LiberoDataset globs all .h5 in the dir, so chunks from all
#     tasks are mixed in one DataLoader. Language instruction is the only
#     per-task disambiguation signal.
#   - Single output checkpoint (no per-task isolation).
#   - Eval loops over all 10 tasks with that one checkpoint.
#
# Adds the SP method:
#   loss.pred_weight=1.0  loss.sigreg_weight=0.1  projector.norm_type=batch
#
# Usage:
#   bash run_sp_libero_spatial.sh
# Override the GPU:
#   CUDA_VISIBLE_DEVICES=3 bash run_sp_libero_spatial.sh
# =============================================================================
set -uo pipefail

RAW_ROOT="/nas_data_new/caz/data_ssd/libero/libero_spatial"
DATA_ROOT="/Data/lyw"
TOKENIZER="${DATA_ROOT}/fast_tokenizer"
SP_PROCESSED="${DATA_ROOT}/libero_processed_v4/libero_spatial"

# Seed control — pass via CLI: `SEED=42 bash run_sp_libero_spatial.sh`.
# Output dirs include the seed so multiple seed sweeps coexist; the
# default 3072 matches config/train/lewm.yaml so a no-arg run reproduces
# the original sweep.
SEED="${SEED:-3072}"
SP_CKPT_DIR="${DATA_ROOT}/stable-wm/sp_libero_spatial_joint_seed${SEED}"
RESULTS_LOG="${DATA_ROOT}/sp_libero_spatial_joint_seed${SEED}_results.txt"
EVAL_LOG="${DATA_ROOT}/sp_eval_joint_seed${SEED}.log"

NUM_EPISODES=20
MAX_STEPS=300
CHUNK_SIZE=20
CHUNK_STRIDE=1
MAX_EPOCHS=100         # match frozen baseline
BATCH_SIZE=128         # match frozen baseline; if SP path OOMs at this size,
                       # fall back to 64 (CLI override `loader.batch_size=64`)
PRED_WEIGHT=1.0
SIGREG_WEIGHT=0.1

# Pin to a free GPU. server-51 has 4× RTX 4090; GPU 0 is often busy with
# another user's process. CLI override:  CUDA_VISIBLE_DEVICES=3 bash ...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

mkdir -p "$SP_PROCESSED" "$SP_CKPT_DIR"
echo "==== SP joint libero_spatial run started $(date) ====" | tee "$RESULTS_LOG"
echo "GPU=$CUDA_VISIBLE_DEVICES, seed=$SEED, batch=$BATCH_SIZE, epochs=$MAX_EPOCHS, pred=$PRED_WEIGHT, sigreg=$SIGREG_WEIGHT" | tee -a "$RESULTS_LOG"
echo "ckpt_dir=$SP_CKPT_DIR" | tee -a "$RESULTS_LOG"

log()  { echo -e "\n$(date '+%H:%M:%S') [INFO] $*"  | tee -a "$RESULTS_LOG"; }
err()  { echo -e "\n$(date '+%H:%M:%S') [ERR ] $*"  | tee -a "$RESULTS_LOG" >&2; }
ok()   { echo -e "\n$(date '+%H:%M:%S') [ OK ] $*"  | tee -a "$RESULTS_LOG"; }

# -----------------------------------------------------------------------------
# Step 1: ensure all 10 tasks are preprocessed to v4 (with future fields)
# -----------------------------------------------------------------------------
log "===== Step 1: preprocessing (skip existing) ====="
for raw_hdf5 in "$RAW_ROOT"/*.hdf5; do
    [ -f "$raw_hdf5" ] || continue
    base=$(basename "$raw_hdf5" .hdf5)
    base="${base%_demo}"
    proc_h5="${SP_PROCESSED}/${base}.h5"

    if [ -f "$proc_h5" ]; then
        log "  skip (exists): $base"
        continue
    fi
    log "  preprocessing: $base"
    if ! python preprocess_libero.py \
        --input "$raw_hdf5" --output "$proc_h5" \
        --chunk-size "$CHUNK_SIZE" --stride "$CHUNK_STRIDE" \
        --image-key agentview_rgb --hand-image-key eye_in_hand_rgb \
        --max-action-tokens 80 \
        --load-tokenizer "$TOKENIZER" 2>&1 | tail -10; then
        err "  preprocess FAILED for $base, aborting"
        exit 1
    fi
done

n_proc=$(ls "$SP_PROCESSED"/*.h5 2>/dev/null | wc -l)
log "preprocess done: $n_proc files in $SP_PROCESSED"
if [ "$n_proc" -lt 10 ]; then
    err "expected 10 preprocessed files, got $n_proc — aborting"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 2: JOINT train one model on all 10 tasks
# -----------------------------------------------------------------------------
log "===== Step 2: joint training (all 10 tasks, 1 ckpt) ====="

if [ -f "$SP_CKPT_DIR/lewm_weights.ckpt" ]; then
    log "  ckpt exists, skipping training: $SP_CKPT_DIR/lewm_weights.ckpt"
else
    export STABLEWM_HOME="$SP_CKPT_DIR"
    if ! python train.py \
        data=libero \
        data.dataset.hdf5_dir="$SP_PROCESSED" \
        loss.pred_weight="$PRED_WEIGHT" \
        loss.sigreg_weight="$SIGREG_WEIGHT" \
        projector.norm_type=batch \
        trainer.devices=1 \
        loader.batch_size="$BATCH_SIZE" \
        trainer.max_epochs="$MAX_EPOCHS" \
        seed="$SEED" \
        subdir="" \
        output_model_name=lewm 2>&1 | tail -30; then
        err "training FAILED — aborting"
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Step 3: find best-val epoch from TB events, keep only that + last
# -----------------------------------------------------------------------------
log "===== Step 3: cleanup intermediate ckpts ====="
best_epoch=$(python << PYEOF
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
versions = sorted(glob.glob("$SP_CKPT_DIR/tb_logs/vla_baseline/version_*"))
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
ckpts = sorted([int(os.path.basename(p).split("_")[2]) for p in glob.glob("$SP_CKPT_DIR/lewm_epoch_*_object.ckpt")])
if not ckpts:
    print(""); raise SystemExit
step_per_epoch = max(1, evs[-1].step // max(ckpts))
best_epoch_guess = max(1, best.step // step_per_epoch)
best_epoch = min(ckpts, key=lambda e: abs(e - best_epoch_guess))
print(best_epoch)
PYEOF
)
last_epoch=$(ls "$SP_CKPT_DIR"/lewm_epoch_*_object.ckpt 2>/dev/null | \
    sed 's/.*lewm_epoch_\([0-9]*\)_object.ckpt/\1/' | sort -n | tail -1)
if [ -z "$best_epoch" ]; then
    err "  could not determine best-val epoch; using last_epoch=$last_epoch"
    best_epoch="$last_epoch"
fi
log "  best-val epoch = $best_epoch  (last = $last_epoch)"
log "  cleaning intermediate ckpts (keep $best_epoch + $last_epoch)"
for ckpt in "$SP_CKPT_DIR"/lewm_epoch_*_object.ckpt; do
    [ -f "$ckpt" ] || continue
    ep=$(echo "$ckpt" | sed 's/.*lewm_epoch_\([0-9]*\)_object.ckpt/\1/')
    if [ "$ep" != "$best_epoch" ] && [ "$ep" != "$last_epoch" ]; then
        rm -f "$ckpt"
    fi
done
best_ckpt="$SP_CKPT_DIR/lewm_epoch_${best_epoch}_object.ckpt"

# -----------------------------------------------------------------------------
# Step 4: eval the single joint ckpt on all 10 tasks
# -----------------------------------------------------------------------------
log "===== Step 4: eval (1 ckpt × 10 tasks × $NUM_EPISODES episodes) ====="
if [ ! -f "$best_ckpt" ]; then
    err "best ckpt $best_ckpt not found — eval skipped"
    exit 1
fi

if ! python eval_libero.py \
    --checkpoint "$best_ckpt" \
    --tokenizer "$TOKENIZER" \
    --processed-dir "$SP_PROCESSED" \
    --suite libero_spatial \
    --num-episodes "$NUM_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --device cuda 2>&1 | tee "$EVAL_LOG" | tail -50; then
    err "eval FAILED — see $EVAL_LOG"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 5: summary
# -----------------------------------------------------------------------------
echo "" | tee -a "$RESULTS_LOG"
echo "==== PER-TASK RESULTS ====" | tee -a "$RESULTS_LOG"
# Match the indented "Task N: NN/20 (NN.N%) — ..." rows printed by eval_libero.py
grep -E "^[[:space:]]+Task[[:space:]]+[0-9]+:" "$EVAL_LOG" | tee -a "$RESULTS_LOG"
echo "" | tee -a "$RESULTS_LOG"
echo "==== OVERALL ====" | tee -a "$RESULTS_LOG"
grep -E "Overall:" "$EVAL_LOG" | tee -a "$RESULTS_LOG"
echo "==== run finished $(date) ====" | tee -a "$RESULTS_LOG"

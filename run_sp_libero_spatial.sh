#!/usr/bin/env bash
# =============================================================================
# run_sp_libero_spatial.sh — full SP training + eval on libero_spatial (10 tasks)
#
# Per-task pipeline:
#   1. preprocess to v4 format (with image_*_future + proprio_future)
#   2. train with SP enabled (pred_weight=1.0, sigreg_weight=0.1, BN projector)
#   3. find best-val epoch from TensorBoard, keep only that _object.ckpt
#   4. eval 20 episodes using best-val _object.ckpt
#   5. emit summary line to results log
#
# Failure isolation: any step that fails just skips that task; later tasks
# still run. Final summary shows which tasks completed.
#
# Usage:
#   bash run_sp_libero_spatial.sh
#
# Edit only RAW_ROOT / DATA_ROOT / TOKENIZER if paths differ from defaults.
# =============================================================================
set -uo pipefail   # NOT -e, so a single task failure doesn't kill the loop

RAW_ROOT="/nas_data_new/caz/data_ssd/libero/libero_spatial"
DATA_ROOT="/Data/lyw"
TOKENIZER="${DATA_ROOT}/fast_tokenizer"
SP_PROCESSED="${DATA_ROOT}/libero_processed_v4/libero_spatial"
SP_CKPT_ROOT="${DATA_ROOT}/stable-wm/sp_libero_spatial"
RESULTS_LOG="${DATA_ROOT}/sp_libero_spatial_results.txt"

NUM_EPISODES=20
MAX_STEPS=300
CHUNK_SIZE=20
CHUNK_STRIDE=1
MAX_EPOCHS=50          # baseline used 100 but best-val was ~epoch 25-40
BATCH_SIZE=64          # conservative — bf16 + SP cat-ViT may OOM at 128
PRED_WEIGHT=1.0
SIGREG_WEIGHT=0.1
DEVICE="cuda"

mkdir -p "$SP_PROCESSED" "$SP_CKPT_ROOT"
echo "==== SP libero_spatial run started $(date) ====" | tee "$RESULTS_LOG"

log()  { echo -e "\n$(date '+%H:%M:%S') [INFO] $*"  | tee -a "$RESULTS_LOG"; }
err()  { echo -e "\n$(date '+%H:%M:%S') [ERR ] $*"  | tee -a "$RESULTS_LOG" >&2; }
ok()   { echo -e "\n$(date '+%H:%M:%S') [ OK ] $*"  | tee -a "$RESULTS_LOG"; }

success_count=0
total_tasks=0

for raw_hdf5 in "$RAW_ROOT"/*.hdf5; do
    [ -f "$raw_hdf5" ] || continue
    total_tasks=$((total_tasks + 1))

    base=$(basename "$raw_hdf5" .hdf5)
    base="${base%_demo}"
    proc_h5="${SP_PROCESSED}/${base}.h5"
    task_ckpt_dir="${SP_CKPT_ROOT}/${base}"
    task_data_dir="${SP_PROCESSED}/.train_${base}"

    log "===== Task $total_tasks: $base ====="

    # --- Step 1: preprocess (skip if already done) ---
    if [ -f "$proc_h5" ]; then
        log "  preprocess: $proc_h5 already exists, skipping"
    else
        log "  preprocessing → $proc_h5"
        if ! python preprocess_libero.py \
            --input "$raw_hdf5" \
            --output "$proc_h5" \
            --chunk-size "$CHUNK_SIZE" --stride "$CHUNK_STRIDE" \
            --image-key agentview_rgb \
            --hand-image-key eye_in_hand_rgb \
            --max-action-tokens 80 \
            --load-tokenizer "$TOKENIZER" 2>&1 | tail -20; then
            err "  preprocess FAILED for $base, skipping task"
            continue
        fi
    fi

    # --- Step 2: train (single-task, SP enabled, BN projector) ---
    if [ -f "$task_ckpt_dir/lewm_weights.ckpt" ]; then
        log "  train: $task_ckpt_dir/lewm_weights.ckpt already exists, skipping training"
    else
        log "  training (max_epochs=$MAX_EPOCHS, batch=$BATCH_SIZE, pred=$PRED_WEIGHT, sigreg=$SIGREG_WEIGHT)"

        # Single-task data dir = symlink dir with one .h5
        rm -rf "$task_data_dir"
        mkdir -p "$task_data_dir"
        ln -sf "$proc_h5" "$task_data_dir/$(basename "$proc_h5")"

        export STABLEWM_HOME="$task_ckpt_dir"
        mkdir -p "$task_ckpt_dir"

        # `trainer.devices=1` REQUIRED — train.py M3 guard rejects 'auto'
        # when projector.norm_type=batch to prevent silent multi-GPU
        # BatchNorm divergence. Explicit single-GPU is the safe path.
        if ! python train.py \
            data=libero \
            data.dataset.hdf5_dir="$task_data_dir" \
            loss.pred_weight="$PRED_WEIGHT" \
            loss.sigreg_weight="$SIGREG_WEIGHT" \
            projector.norm_type=batch \
            trainer.devices=1 \
            loader.batch_size="$BATCH_SIZE" \
            trainer.max_epochs="$MAX_EPOCHS" \
            subdir="" \
            output_model_name=lewm 2>&1 | tail -20; then
            err "  training FAILED for $base, skipping task"
            rm -rf "$task_data_dir"
            continue
        fi

        rm -rf "$task_data_dir"
    fi

    # --- Step 3: find best-val epoch, keep only that + last _object.ckpt ---
    log "  finding best-val epoch from TB events"
    best_epoch=$(python << PYEOF
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
versions = sorted(glob.glob("$task_ckpt_dir/tb_logs/vla_baseline/version_*"))
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
# Lightning logs validate/* on the global step counter, not epoch — convert
# back via ModelObjectCallBack epoch_<N>_object.ckpt files we have on disk.
ckpts = sorted([int(os.path.basename(p).split("_")[2]) for p in glob.glob("$task_ckpt_dir/lewm_epoch_*_object.ckpt")])
if not ckpts:
    print(""); raise SystemExit
# Map step → epoch index by ratio (step_max / epoch_max)
step_per_epoch = max(1, evs[-1].step // max(ckpts))
best_epoch_guess = max(1, best.step // step_per_epoch)
# Pick the closest existing ckpt epoch
best_epoch = min(ckpts, key=lambda e: abs(e - best_epoch_guess))
print(best_epoch)
PYEOF
)
    if [ -z "$best_epoch" ]; then
        err "  could not determine best-val epoch for $base, keeping last only"
        last_epoch=$(ls "$task_ckpt_dir"/lewm_epoch_*_object.ckpt 2>/dev/null | \
            sed 's/.*lewm_epoch_\([0-9]*\)_object.ckpt/\1/' | sort -n | tail -1)
        best_epoch="$last_epoch"
    fi
    log "  best-val epoch = $best_epoch"

    last_epoch=$(ls "$task_ckpt_dir"/lewm_epoch_*_object.ckpt 2>/dev/null | \
        sed 's/.*lewm_epoch_\([0-9]*\)_object.ckpt/\1/' | sort -n | tail -1)
    log "  cleaning intermediate ckpts (keep epoch $best_epoch + $last_epoch)"
    for ckpt in "$task_ckpt_dir"/lewm_epoch_*_object.ckpt; do
        [ -f "$ckpt" ] || continue
        ep=$(echo "$ckpt" | sed 's/.*lewm_epoch_\([0-9]*\)_object.ckpt/\1/')
        if [ "$ep" != "$best_epoch" ] && [ "$ep" != "$last_epoch" ]; then
            rm -f "$ckpt"
        fi
    done

    best_ckpt="$task_ckpt_dir/lewm_epoch_${best_epoch}_object.ckpt"

    # --- Step 4: eval ---
    if [ ! -f "$best_ckpt" ]; then
        err "  best ckpt $best_ckpt not found, skipping eval"
        continue
    fi

    log "  evaluating $best_ckpt"
    eval_data_dir=$(mktemp -d "${SP_PROCESSED}/.eval_${base}_XXXXXX")
    ln -sf "$proc_h5" "$eval_data_dir/$(basename "$proc_h5")"

    if ! python eval_libero.py \
        --checkpoint "$best_ckpt" \
        --tokenizer "$TOKENIZER" \
        --processed-dir "$eval_data_dir" \
        --suite libero_spatial \
        --num-episodes "$NUM_EPISODES" \
        --max-steps "$MAX_STEPS" \
        --device "$DEVICE" 2>&1 | tee /tmp/sp_eval_${base}.log | tail -30; then
        err "  eval FAILED for $base"
        rm -rf "$eval_data_dir"
        continue
    fi
    rm -rf "$eval_data_dir"

    # Parse last "Result:" line for this task
    result_line=$(grep "Result:" /tmp/sp_eval_${base}.log | tail -1)
    ok "  ${base}: ${result_line:-no result parsed}"
    success_count=$((success_count + 1))
done

echo "" | tee -a "$RESULTS_LOG"
echo "==== SUMMARY ====" | tee -a "$RESULTS_LOG"
echo "Tasks completed: ${success_count}/${total_tasks}" | tee -a "$RESULTS_LOG"
grep "\[ OK \]" "$RESULTS_LOG" | tee -a "$RESULTS_LOG"
echo "==== run finished $(date) ====" | tee -a "$RESULTS_LOG"

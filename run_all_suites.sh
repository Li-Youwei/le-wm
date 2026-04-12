#!/usr/bin/env bash
# =============================================================================
# run_all_suites.sh — Preprocess, train (per-task), and evaluate on LIBERO suites
#
# Single-task training: one model per task (10 models per suite).
# Each task gets its own checkpoint and is evaluated independently.
#
# Usage:
#   bash run_all_suites.sh                    # run all 4 suites
#   bash run_all_suites.sh libero_object      # start from a specific suite
#   bash run_all_suites.sh --eval-only        # skip preprocess+train, only eval
#
# Paths (edit these):
#   RAW_ROOT    — read-only LIBERO raw data
#   DATA_ROOT   — writable data disk for processed data + checkpoints
#   TOKENIZER   — shared FAST tokenizer (already fitted)
# =============================================================================
set -euo pipefail

# ----------------------------- Configuration ---------------------------------
RAW_ROOT="/nas_data_new/caz/data_ssd/libero"
DATA_ROOT="/Data/lyw"
TOKENIZER="${DATA_ROOT}/fast_tokenizer"

PROCESSED_ROOT="${DATA_ROOT}/libero_processed"
RESULTS_ROOT="${DATA_ROOT}/eval_results"

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
NUM_EPISODES=20
MAX_STEPS=300
CHUNK_SIZE=20
MAX_EPOCHS=100
DEVICE="cuda"

# ----------------------------- Parse arguments -------------------------------
START_FROM=""
EVAL_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --eval-only) EVAL_ONLY=true ;;
        libero_*)    START_FROM="$arg" ;;
        *)           echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ----------------------------- Helper functions ------------------------------
log() { echo -e "\n$(date '+%Y-%m-%d %H:%M:%S') [INFO] $*"; }
err() { echo -e "\n$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $*" >&2; }

# ----------------------------- Main loop -------------------------------------
started=false
if [ -z "$START_FROM" ]; then
    started=true
fi

mkdir -p "$RESULTS_ROOT"

for SUITE in "${SUITES[@]}"; do
    # Skip until we reach the start-from suite
    if [ "$started" = false ]; then
        if [ "$SUITE" = "$START_FROM" ]; then
            started=true
        else
            log "Skipping $SUITE (waiting for $START_FROM)"
            continue
        fi
    fi

    log "========== $SUITE =========="

    RAW_DIR="${RAW_ROOT}/${SUITE}"
    PROC_DIR="${PROCESSED_ROOT}/${SUITE}"
    EVAL_LOG="${RESULTS_ROOT}/${SUITE}.txt"

    # ---- Step 1: Preprocess all tasks in this suite ----
    if [ "$EVAL_ONLY" = false ]; then
        log "[$SUITE] Step 1/3: Preprocessing"
        mkdir -p "$PROC_DIR"

        for hdf5_file in "${RAW_DIR}"/*.hdf5; do
            [ -f "$hdf5_file" ] || continue

            base=$(basename "$hdf5_file" .hdf5)
            base="${base%_demo}"
            out="${PROC_DIR}/${base}.h5"

            if [ -f "$out" ]; then
                log "  Skip (exists): $(basename "$out")"
                continue
            fi

            log "  Processing: $(basename "$hdf5_file") -> $(basename "$out")"
            python preprocess_libero.py \
                --input "$hdf5_file" \
                --output "$out" \
                --chunk-size "$CHUNK_SIZE" \
                --image-key agentview_rgb \
                --hand-image-key eye_in_hand_rgb \
                --max-action-tokens 80 \
                --load-tokenizer "$TOKENIZER"
        done
        log "[$SUITE] Preprocessing done"

        # ---- Step 2: Train per-task (one model per H5 file) ----
        log "[$SUITE] Step 2/3: Training per-task (${MAX_EPOCHS} epochs each)"

        for h5_file in "${PROC_DIR}"/*.h5; do
            [ -f "$h5_file" ] || continue

            task_name=$(basename "$h5_file" .h5)
            task_ckpt_dir="${DATA_ROOT}/stable-wm/${SUITE}/${task_name}"
            task_ckpt="${task_ckpt_dir}/lewm_weights.ckpt"

            if [ -f "$task_ckpt" ]; then
                log "  Skip (checkpoint exists): ${task_name}"
                continue
            fi

            log "  Training: ${task_name}"

            # Create a temp dir with only this one H5 so LiberoDataset loads single task
            task_data_dir=$(mktemp -d "${PROC_DIR}/.train_${task_name}_XXXXXX")
            ln -s "$(realpath "$h5_file")" "${task_data_dir}/$(basename "$h5_file")"

            export STABLEWM_HOME="$task_ckpt_dir"
            mkdir -p "$task_ckpt_dir"

            if ! python train.py \
                data=libero \
                data.dataset.hdf5_dir="$task_data_dir" \
                subdir="" \
                output_model_name=lewm \
                trainer.max_epochs="$MAX_EPOCHS"; then
                err "  Training FAILED for ${task_name} — aborting suite"
                rm -rf "$task_data_dir"
                exit 1
            fi

            # Clean up temp dir
            rm -rf "$task_data_dir"

            log "  Done: ${task_name} → ${task_ckpt}"
        done

        log "[$SUITE] Per-task training done"
    fi

    # ---- Step 3: Evaluate per-task ----
    log "[$SUITE] Step 3/3: Evaluating (${NUM_EPISODES} episodes per task)"

    suite_successes=0
    suite_total=0
    suite_n_tasks=0
    > "$EVAL_LOG"  # clear log for this suite

    for h5_file in "${PROC_DIR}"/*.h5; do
        [ -f "$h5_file" ] || continue

        task_name=$(basename "$h5_file" .h5)
        task_ckpt="${DATA_ROOT}/stable-wm/${SUITE}/${task_name}/lewm_weights.ckpt"

        if [ ! -f "$task_ckpt" ]; then
            err "  Checkpoint not found for ${task_name}: ${task_ckpt} — skipping"
            continue
        fi

        log "  Evaluating: ${task_name}"

        # eval_libero.py with --processed-dir pointing to a temp dir with one H5
        task_eval_dir=$(mktemp -d "${PROC_DIR}/.eval_${task_name}_XXXXXX")
        ln -s "$(realpath "$h5_file")" "${task_eval_dir}/$(basename "$h5_file")"

        if ! python eval_libero.py \
            --checkpoint "$task_ckpt" \
            --tokenizer "$TOKENIZER" \
            --processed-dir "$task_eval_dir" \
            --suite "$SUITE" \
            --num-episodes "$NUM_EPISODES" \
            --max-steps "$MAX_STEPS" \
            --device "$DEVICE" \
            2>&1 | tee -a "$EVAL_LOG"; then
            err "  Eval FAILED for ${task_name} — aborting suite"
            rm -rf "$task_eval_dir"
            exit 1
        fi

        rm -rf "$task_eval_dir"

        # Parse result from eval log (last "Result:" line for this task)
        result_line=$(grep "Result:" "$EVAL_LOG" | tail -1)
        if [[ "$result_line" =~ ([0-9]+)/([0-9]+) ]]; then
            suite_successes=$((suite_successes + BASH_REMATCH[1]))
            suite_total=$((suite_total + BASH_REMATCH[2]))
        fi
        suite_n_tasks=$((suite_n_tasks + 1))
    done

    # Print suite-level aggregate
    if [ "$suite_total" -gt 0 ]; then
        suite_rate=$(python3 -c "print(f'{100.0 * $suite_successes / $suite_total:.1f}')")
        log "[$SUITE] SUITE TOTAL: ${suite_successes}/${suite_total} (${suite_rate}%) across ${suite_n_tasks} tasks"
        echo "SUITE TOTAL: ${suite_successes}/${suite_total} (${suite_rate}%) across ${suite_n_tasks} tasks" >> "$EVAL_LOG"
    fi

    log "[$SUITE] Eval results saved to: $EVAL_LOG"
done

log "========== ALL SUITES COMPLETE =========="
log "Results:"
for SUITE in "${SUITES[@]}"; do
    log_file="${RESULTS_ROOT}/${SUITE}.txt"
    if [ -f "$log_file" ]; then
        suite_line=$(grep "SUITE TOTAL:" "$log_file" 2>/dev/null | tail -1)
        log "  $SUITE: ${suite_line:-no results}"
    fi
done

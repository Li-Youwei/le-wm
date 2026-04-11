#!/usr/bin/env bash
# =============================================================================
# run_all_suites.sh — Preprocess, train, and evaluate on 4 LIBERO suites
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
DATA_ROOT="/data/lyl"
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
    CKPT_DIR="${DATA_ROOT}/stable-wm/${SUITE}"
    CKPT_PATH="${CKPT_DIR}/lewm_weights.ckpt"
    EVAL_LOG="${RESULTS_ROOT}/${SUITE}.txt"

    # ---- Step 1: Preprocess ----
    if [ "$EVAL_ONLY" = false ]; then
        log "[$SUITE] Step 1/3: Preprocessing"
        mkdir -p "$PROC_DIR"

        for hdf5_file in "${RAW_DIR}"/*.hdf5; do
            [ -f "$hdf5_file" ] || continue  # skip if no files match

            # Output filename: strip _demo.hdf5 suffix, use .h5
            base=$(basename "$hdf5_file" .hdf5)
            base="${base%_demo}"  # remove _demo suffix if present
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
                --load-tokenizer "$TOKENIZER"
        done
        log "[$SUITE] Preprocessing done"

        # ---- Step 2: Train ----
        log "[$SUITE] Step 2/3: Training (${MAX_EPOCHS} epochs)"

        export STABLEWM_HOME="$CKPT_DIR"
        mkdir -p "$CKPT_DIR"

        python train.py \
            data=libero \
            data.dataset.hdf5_dir="$PROC_DIR" \
            subdir="" \
            output_model_name=lewm \
            trainer.max_epochs="$MAX_EPOCHS"

        log "[$SUITE] Training done — checkpoint: $CKPT_PATH"
    fi

    # ---- Step 3: Evaluate ----
    log "[$SUITE] Step 3/3: Evaluating (${NUM_EPISODES} episodes × 10 tasks)"

    if [ ! -f "$CKPT_PATH" ]; then
        err "[$SUITE] Checkpoint not found: $CKPT_PATH — skipping eval"
        continue
    fi

    python eval_libero.py \
        --checkpoint "$CKPT_PATH" \
        --tokenizer "$TOKENIZER" \
        --processed-dir "$PROC_DIR" \
        --suite "$SUITE" \
        --num-episodes "$NUM_EPISODES" \
        --max-steps "$MAX_STEPS" \
        --device "$DEVICE" \
        2>&1 | tee "$EVAL_LOG"

    log "[$SUITE] Eval results saved to: $EVAL_LOG"
done

log "========== ALL SUITES COMPLETE =========="
log "Results:"
for SUITE in "${SUITES[@]}"; do
    log_file="${RESULTS_ROOT}/${SUITE}.txt"
    if [ -f "$log_file" ]; then
        # Print the last "Overall:" line from each log
        overall=$(grep "Overall:" "$log_file" 2>/dev/null | tail -1)
        log "  $SUITE: ${overall:-no results}"
    fi
done

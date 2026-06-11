#!/usr/bin/env bash
# launch_paper_baseline_grid.sh — Launch the 4 paper baseline experiments.
#
# Matrix, one experiment per GPU:
#   SPLIT_MODE=demo_90_10 ARM=all4_sp_sigreg_dinov2_frozen_visual17_patch_sp     CUDA_VISIBLE_DEVICES=0
#   SPLIT_MODE=demo_90_10 ARM=all4_sp_sigreg_dinov2_frozen_visual17_patch_sp_mot CUDA_VISIBLE_DEVICES=1
#   SPLIT_MODE=full       ARM=all4_sp_sigreg_dinov2_frozen_visual17_patch_sp     CUDA_VISIBLE_DEVICES=2
#   SPLIT_MODE=full       ARM=all4_sp_sigreg_dinov2_frozen_visual17_patch_sp_mot CUDA_VISIBLE_DEVICES=3
#
# Run from the fresh code directory after data preparation has produced
# FLAT_DIR, TOKENIZER, and PROCESSED_ROOT. The script waits for all four child
# runs and exits non-zero if any experiment fails.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
FLAT_DIR="${FLAT_DIR:-$DATA_ROOT/libero_processed_v5/all4_flat}"
TOKENIZER="${TOKENIZER:-$DATA_ROOT/fast_tokenizer_all4}"
PROCESSED_ROOT="${PROCESSED_ROOT:-$DATA_ROOT/libero_processed_v5}"
VISION_ENCODER="${VISION_ENCODER:-facebook/dinov2-base}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/grid}"

SEED="${SEED:-3072}"
MAX_STEPS="${MAX_STEPS:-100000}"
VAL_INTERVAL="${VAL_INTERVAL:-4000}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CKPT_TOP_K="${CKPT_TOP_K:-6}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
CAMERA_SIZE="${CAMERA_SIZE:-224}"
PATCH_SP_WEIGHT="${PATCH_SP_WEIGHT:-1.0}"
PROBE_ENABLED="${PROBE_ENABLED:-false}"

BASE_ARM="all4_sp_sigreg_dinov2_frozen_visual17_patch_sp"
MOT_ARM="all4_sp_sigreg_dinov2_frozen_visual17_patch_sp_mot"

mkdir -p "$LOG_DIR" "$CKPT_ROOT"

[[ -d "$FLAT_DIR" ]] || { echo "ERROR: FLAT_DIR missing: $FLAT_DIR" >&2; exit 1; }
[[ -d "$TOKENIZER" ]] || { echo "ERROR: TOKENIZER missing: $TOKENIZER" >&2; exit 1; }
[[ -d "$PROCESSED_ROOT" ]] || { echo "ERROR: PROCESSED_ROOT missing: $PROCESSED_ROOT" >&2; exit 1; }
if [[ -d "$VISION_ENCODER" ]]; then
    echo "[grid] using local vision encoder dir: $VISION_ENCODER"
else
    python - "$VISION_ENCODER" <<'PY'
import sys
from transformers import AutoConfig

AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)
PY
fi

PID_FILE="$LOG_DIR/experiment_pids.tsv"
: > "$PID_FILE"

launch_one() {
    local name="$1"
    local split_mode="$2"
    local arm="$3"
    local gpu="$4"
    local log_file="$LOG_DIR/${name}.log"

    echo "[grid] launch $name split=$split_mode arm=$arm gpu=$gpu log=$log_file"
    (
        cd "$ROOT"
        ARM="$arm" \
        SPLIT_MODE="$split_mode" \
        CUDA_VISIBLE_DEVICES="$gpu" \
        SEED="$SEED" \
        MAX_STEPS="$MAX_STEPS" \
        VAL_INTERVAL="$VAL_INTERVAL" \
        WARMUP_STEPS="$WARMUP_STEPS" \
        BATCH_SIZE="$BATCH_SIZE" \
        CKPT_TOP_K="$CKPT_TOP_K" \
        EVAL_EPISODES="$EVAL_EPISODES" \
        CAMERA_SIZE="$CAMERA_SIZE" \
        PATCH_SP_WEIGHT="$PATCH_SP_WEIGHT" \
        FLAT_DIR="$FLAT_DIR" \
        TOKENIZER="$TOKENIZER" \
        PROCESSED_ROOT="$PROCESSED_ROOT" \
        VISION_ENCODER="$VISION_ENCODER" \
        CKPT_ROOT="$CKPT_ROOT" \
        PROBE_ENABLED="$PROBE_ENABLED" \
        bash run_all4_pretrained_vision.sh
    ) > "$log_file" 2>&1 &
    local pid=$!
    printf "%s\t%s\t%s\t%s\t%s\n" "$pid" "$name" "$split_mode" "$arm" "$gpu" >> "$PID_FILE"
}

launch_one "demo_90_10_nomot" "demo_90_10" "$BASE_ARM" "0"
launch_one "demo_90_10_mot" "demo_90_10" "$MOT_ARM" "1"
launch_one "full_nomot" "full" "$BASE_ARM" "2"
launch_one "full_mot" "full" "$MOT_ARM" "3"

status=0
while IFS=$'\t' read -r pid name split_mode arm gpu; do
    if wait "$pid"; then
        echo "[grid] done $name split=$split_mode arm=$arm gpu=$gpu"
    else
        rc=$?
        echo "[grid] FAILED $name split=$split_mode arm=$arm gpu=$gpu rc=$rc" >&2
        status=1
    fi
done < "$PID_FILE"

exit "$status"

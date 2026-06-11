#!/usr/bin/env bash
# server_paper_baseline_pipeline.sh — Fresh-server paper baseline pipeline.
#
# Run this from a fresh code directory under /Data/lyw. It never writes to the
# official LIBERO root; all regenerated data, processed data, tokenizer cache,
# checkpoints, and logs live under this directory unless overridden.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RAW_ROOT="${RAW_ROOT:-/nas_data_new/caz/data_ssd/libero}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
FILTERED_ROOT="${FILTERED_ROOT:-$DATA_ROOT/libero_filtered_224}"
PROCESSED_ROOT="${PROCESSED_ROOT:-$DATA_ROOT/libero_processed_v5}"
TOKENIZER="${TOKENIZER:-$DATA_ROOT/fast_tokenizer_all4}"
FLAT_DIR="${FLAT_DIR:-$PROCESSED_ROOT/all4_flat}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
HF_HOME="${HF_HOME:-$ROOT/hf_home}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
VISION_ENCODER="${VISION_ENCODER:-$DATA_ROOT/hf_models/facebook-dinov2-base}"
LOCAL_VISION_ENCODER_SOURCE="${LOCAL_VISION_ENCODER_SOURCE:-/Data/lyw/hf_models/facebook-dinov2-base}"
CONDA_ENV="${CONDA_ENV:-vla}"
PARALLEL="${PARALLEL:-1}"
CAMERA_SIZE="${CAMERA_SIZE:-224}"
WAIT_STEPS="${WAIT_STEPS:-10}"
NOOP_THRESHOLD="${NOOP_THRESHOLD:-1e-4}"
SPLIT_MODE="${SPLIT_MODE:-demo_90_10}"
ARM="${ARM:-sp_sigreg}"
SEED="${SEED:-3072}"
MAX_STEPS="${MAX_STEPS:-100000}"
VAL_INTERVAL="${VAL_INTERVAL:-4000}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PROBE_ENABLED="${PROBE_ENABLED:-false}"
HF_FALLBACK_HOME="${HF_FALLBACK_HOME:-$HOME/.cache/huggingface}"

refuse_raw_write_path() {
    local name="$1" path="$2"
    case "$path" in
        "$RAW_ROOT"|"$RAW_ROOT"/*)
            echo "ERROR: $name=$path is under RAW_ROOT=$RAW_ROOT; refusing to write." >&2
            exit 1
            ;;
    esac
}

refuse_raw_write_path DATA_ROOT "$DATA_ROOT"
refuse_raw_write_path FILTERED_ROOT "$FILTERED_ROOT"
refuse_raw_write_path PROCESSED_ROOT "$PROCESSED_ROOT"
refuse_raw_write_path TOKENIZER "$TOKENIZER"
refuse_raw_write_path CKPT_ROOT "$CKPT_ROOT"
refuse_raw_write_path LOG_DIR "$LOG_DIR"
refuse_raw_write_path HF_HOME "$HF_HOME"

mkdir -p "$DATA_ROOT" "$FILTERED_ROOT" "$PROCESSED_ROOT" "$CKPT_ROOT" "$LOG_DIR" "$HF_HOME"

export HF_ENDPOINT HF_HOME VISION_ENCODER
export MUJOCO_GL="${MUJOCO_GL:-egl}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[baseline] ROOT=$ROOT"
echo "[baseline] RAW_ROOT=$RAW_ROOT (read-only)"
echo "[baseline] DATA_ROOT=$DATA_ROOT"
echo "[baseline] HF_ENDPOINT=$HF_ENDPOINT"
echo "[baseline] VISION_ENCODER=$VISION_ENCODER"
echo "[baseline] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

prepare_vision_encoder() {
    if [[ -d "$VISION_ENCODER" ]]; then
        echo "[baseline] Using vision encoder: $VISION_ENCODER"
        return
    fi
    case "$VISION_ENCODER" in
        "$DATA_ROOT"/hf_models/*)
            if [[ ! -d "$LOCAL_VISION_ENCODER_SOURCE" ]]; then
                echo "ERROR: missing local DINOv2 source: $LOCAL_VISION_ENCODER_SOURCE" >&2
                exit 1
            fi
            local vision_parent
            vision_parent="$(dirname "$VISION_ENCODER")"
            mkdir -p "$vision_parent"
            echo "[baseline] Copy DINOv2 model $LOCAL_VISION_ENCODER_SOURCE -> $vision_parent/"
            cp -a "$LOCAL_VISION_ENCODER_SOURCE" "$vision_parent/"
            ;;
        *)
            echo "[baseline] VISION_ENCODER is not a local path; will resolve via HF cache: $VISION_ENCODER"
            ;;
    esac
}

copy_available_hf_fallback_cache() {
    local src_hub="$HF_FALLBACK_HOME/hub"
    local dst_hub="$HF_HOME/hub"
    mkdir -p "$dst_hub"
    for repo in models--physical-intelligence--fast models--t5-small models--facebook--dinov2-base; do
        if [[ -d "$dst_hub/$repo" ]]; then
            continue
        fi
        if [[ -d "$src_hub/$repo" ]]; then
            echo "[baseline] Copy fallback cache $src_hub/$repo -> $dst_hub/"
            cp -a "$src_hub/$repo" "$dst_hub/"
        else
            echo "[baseline] Fallback cache not found for $repo"
        fi
    done
}

validate_hf_cache() {
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
import os
from transformers import AutoModel, AutoProcessor, T5EncoderModel, T5Tokenizer

AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
T5Tokenizer.from_pretrained("t5-small")
T5EncoderModel.from_pretrained("t5-small")
AutoModel.from_pretrained(os.environ["VISION_ENCODER"], local_files_only=True)
print("hf-cache-ok")
PY
}

prepare_vision_encoder

echo "[baseline] Copy any existing server HF cache from $HF_FALLBACK_HOME" | tee "$LOG_DIR/00_hf_precache.log"
copy_available_hf_fallback_cache 2>&1 | tee -a "$LOG_DIR/00_hf_precache.log"

set +e
validate_hf_cache 2>&1 | tee -a "$LOG_DIR/00_hf_precache.log"
hf_status=${PIPESTATUS[0]}
set -e
if [[ "$hf_status" == "0" ]]; then
    echo "[baseline] HF cache validation succeeded." | tee -a "$LOG_DIR/00_hf_precache.log"
else
    echo "[baseline] HF cache incomplete; pre-caching missing dependencies via mirror ..." | tee -a "$LOG_DIR/00_hf_precache.log"
    set +e
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - <<'PY' 2>&1 | tee -a "$LOG_DIR/00_hf_precache.log"
from transformers import AutoModel, AutoProcessor, T5EncoderModel, T5Tokenizer

AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
T5Tokenizer.from_pretrained("t5-small")
T5EncoderModel.from_pretrained("t5-small")
AutoModel.from_pretrained("facebook/dinov2-base")
print("hf-precache-ok")
PY
    hf_status=${PIPESTATUS[0]}
    set -e
    if [[ "$hf_status" != "0" ]]; then
        echo "ERROR: mirror pre-cache failed and local fallback cache is incomplete." >&2
        exit "$hf_status"
    fi
    validate_hf_cache 2>&1 | tee -a "$LOG_DIR/00_hf_precache.log"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[baseline] Regenerate/filter official LIBERO HDF5 ..."
set +e
RAW_ROOT="$RAW_ROOT" \
FILTERED_ROOT="$FILTERED_ROOT" \
OUT_ROOT="$PROCESSED_ROOT" \
TOKENIZER="$TOKENIZER" \
PARALLEL="$PARALLEL" \
CAMERA_SIZE="$CAMERA_SIZE" \
WAIT_STEPS="$WAIT_STEPS" \
NOOP_THRESHOLD="$NOOP_THRESHOLD" \
bash preprocess_all4.sh 2>&1 | tee "$LOG_DIR/01_regenerate_filter.log"
regen_status=${PIPESTATUS[0]}
set -e
if [[ "$regen_status" != "0" && "$regen_status" != "2" ]]; then
    echo "ERROR: regenerate/filter step failed with status $regen_status" >&2
    exit "$regen_status"
fi

if [[ ! -d "$TOKENIZER" ]]; then
    echo "[baseline] Fit shared FAST tokenizer from filtered data ..."
    python fit_tokenizer_all4.py \
        --raw-root "$FILTERED_ROOT" \
        --tokenizer-out "$TOKENIZER" \
        --audit-out "$PROCESSED_ROOT/audit_token_length.json" \
        --chunk-size 20 \
        --stride 1 \
        2>&1 | tee "$LOG_DIR/02_fit_tokenizer.log"
else
    echo "[baseline] Reusing tokenizer in this fresh root: $TOKENIZER"
fi

echo "[baseline] Preprocess filtered HDF5 into training HDF5 ..."
RAW_ROOT="$RAW_ROOT" \
FILTERED_ROOT="$FILTERED_ROOT" \
OUT_ROOT="$PROCESSED_ROOT" \
TOKENIZER="$TOKENIZER" \
PARALLEL="$PARALLEL" \
CAMERA_SIZE="$CAMERA_SIZE" \
WAIT_STEPS="$WAIT_STEPS" \
NOOP_THRESHOLD="$NOOP_THRESHOLD" \
bash preprocess_all4.sh 2>&1 | tee "$LOG_DIR/03_preprocess.log"

echo "[baseline] Build flat 40-task symlink directory ..."
mkdir -p "$FLAT_DIR"
shopt -s nullglob
for suite_dir in "$PROCESSED_ROOT"/libero_spatial "$PROCESSED_ROOT"/libero_object "$PROCESSED_ROOT"/libero_goal "$PROCESSED_ROOT"/libero_10; do
    suite="$(basename "$suite_dir")"
    for h5 in "$suite_dir"/*.h5; do
        ln -sfn "$h5" "$FLAT_DIR/${suite}__$(basename "$h5")"
    done
done
flat_count="$(find "$FLAT_DIR" -maxdepth 1 -type l -name '*.h5' | wc -l)"
echo "[baseline] FLAT_DIR=$FLAT_DIR count=$flat_count"
if [[ "$flat_count" -lt 40 ]]; then
    echo "ERROR: expected 40 flat H5 symlinks, got $flat_count" >&2
    exit 1
fi

echo "[baseline] Launch paper baseline grid (4 experiments × 1 GPU) ..."
SEED="$SEED" \
MAX_STEPS="$MAX_STEPS" \
VAL_INTERVAL="$VAL_INTERVAL" \
WARMUP_STEPS="$WARMUP_STEPS" \
BATCH_SIZE="$BATCH_SIZE" \
EVAL_EPISODES="$EVAL_EPISODES" \
CAMERA_SIZE="$CAMERA_SIZE" \
FLAT_DIR="$FLAT_DIR" \
TOKENIZER="$TOKENIZER" \
PROCESSED_ROOT="$PROCESSED_ROOT" \
VISION_ENCODER="$VISION_ENCODER" \
CKPT_ROOT="$CKPT_ROOT" \
HF_HOME="$HF_HOME" \
HF_ENDPOINT="$HF_ENDPOINT" \
PROBE_ENABLED="$PROBE_ENABLED" \
bash launch_paper_baseline_grid.sh 2>&1 | tee "$LOG_DIR/04_train_eval_grid.log"

echo "[baseline] Pipeline complete. Logs: $LOG_DIR"

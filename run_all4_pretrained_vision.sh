#!/usr/bin/env bash
# run_all4_pretrained_vision.sh - 4-suite LIBERO training with a frozen
# HuggingFace vision backbone. Default is a clean action-only baseline with
# frozen DINOv2, used to isolate whether object failures come from visual
# representation rather than SP/SIGReg/MoT.
set -euo pipefail

ARM="${ARM:-all4_dinov2_frozen}"
SEED="${SEED:-3072}"
MAX_STEPS="${MAX_STEPS:-100000}"
VAL_INTERVAL="${VAL_INTERVAL:-4000}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"

FLAT_DIR="${FLAT_DIR:-/Data/lyw/libero_processed_v5/all4_flat}"
TOKENIZER="${TOKENIZER:-/Data/lyw/fast_tokenizer_all4}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/Data/lyw/libero_processed_v5}"
VISION_ENCODER="${VISION_ENCODER:-/Data/lyw/hf_models/facebook-dinov2-base}"
CKPT_ROOT="${CKPT_ROOT:-/Data/lyw/stable-wm}"
CKPT_DIR="${CKPT_ROOT}/${ARM}_seed${SEED}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export STABLEWM_HOME="$CKPT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vla

echo "=========================================================="
echo "[all4_pretrained_vision] ARM=$ARM SEED=$SEED MAX_STEPS=$MAX_STEPS"
echo "[all4_pretrained_vision] FLAT_DIR=$FLAT_DIR"
echo "[all4_pretrained_vision] TOKENIZER=$TOKENIZER"
echo "[all4_pretrained_vision] PROCESSED_ROOT=$PROCESSED_ROOT"
echo "[all4_pretrained_vision] VISION_ENCODER=$VISION_ENCODER"
echo "[all4_pretrained_vision] CKPT_DIR=$CKPT_DIR"
echo "[all4_pretrained_vision] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================================="

[[ -d "$FLAT_DIR" ]] || { echo "ERROR: FLAT_DIR missing: $FLAT_DIR" >&2; exit 1; }
[[ -d "$TOKENIZER" ]] || { echo "ERROR: TOKENIZER missing: $TOKENIZER" >&2; exit 1; }
[[ -d "$VISION_ENCODER" ]] || {
    echo "ERROR: VISION_ENCODER missing: $VISION_ENCODER" >&2
    echo "Download/copy a HuggingFace vision model there first." >&2
    exit 1
}
N_FLAT=$(find "$FLAT_DIR" -maxdepth 1 -name "*.h5" | wc -l)
if [[ "$N_FLAT" -lt 40 ]]; then
    echo "ERROR: expected 40 .h5 in $FLAT_DIR, got $N_FLAT" >&2
    exit 1
fi

mkdir -p "$CKPT_DIR"
TRAIN_LOG="${CKPT_DIR}/train.log"

python train.py \
    data=libero \
    data.dataset.hdf5_dir="$FLAT_DIR" \
    vision_encoder.source=hf \
    vision_encoder.model_name_or_path="$VISION_ENCODER" \
    vision_encoder.freeze=true \
    vision_encoder.local_files_only=true \
    loss.pred_weight=0 \
    loss.sigreg_weight=0 \
    predictor.state_prediction_arch=shared \
    projector.norm_type=layer \
    scheduler.warmup_steps="$WARMUP_STEPS" \
    trainer.devices=1 \
    +trainer.max_steps="$MAX_STEPS" \
    trainer.max_epochs=999 \
    +trainer.val_check_interval="$VAL_INTERVAL" \
    +trainer.check_val_every_n_epoch=null \
    loader.batch_size="$BATCH_SIZE" \
    seed="$SEED" \
    subdir="" \
    output_model_name=lewm \
    2>&1 | tee "$TRAIN_LOG"

PICK_OUT=$(python pick_best_ckpt.py --ckpt-dir "$CKPT_DIR" --top-k 3)
echo "[all4_pretrained_vision] pick_best_ckpt output:"
echo "$PICK_OUT"
BEST_CKPT=$(echo "$PICK_OUT" | python -c "import sys, json; d=json.load(sys.stdin); print(d.get('top_1') or '')")
if [[ -z "$BEST_CKPT" || ! -f "$BEST_CKPT" ]]; then
    echo "ERROR: pick_best_ckpt returned no usable ckpt" >&2
    exit 2
fi
echo "[all4_pretrained_vision] best ckpt: $BEST_CKPT"

for suite in libero_spatial libero_object libero_goal libero_10; do
    EVAL_LOG="${CKPT_DIR}/eval_${suite}.log"
    PROC_DIR="${PROCESSED_ROOT}/${suite}"
    [[ -d "$PROC_DIR" ]] || { echo "ERROR: processed suite dir missing: $PROC_DIR" >&2; exit 1; }
    echo "[all4_pretrained_vision] eval $suite -> $EVAL_LOG"
    python eval_libero.py \
        --checkpoint "$BEST_CKPT" \
        --tokenizer "$TOKENIZER" \
        --processed-dir "$PROC_DIR" \
        --suite "$suite" \
        --num-episodes 20 \
        --max-steps 300 \
        --device cuda \
        --seed 42 \
        2>&1 | tee "$EVAL_LOG"
done

echo "[all4_pretrained_vision] ALL DONE - see logs under $CKPT_DIR/"

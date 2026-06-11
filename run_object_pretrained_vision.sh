#!/usr/bin/env bash
# run_object_pretrained_vision.sh — object-only diagnostic with a frozen
# pretrained HuggingFace vision backbone. T5-small stays unchanged.
set -euo pipefail

ARM="${ARM:-object_dinov2_frozen}"
SEED="${SEED:-3072}"
MAX_STEPS="${MAX_STEPS:-50000}"
VAL_INTERVAL="${VAL_INTERVAL:-2000}"
BATCH_SIZE="${BATCH_SIZE:-96}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
CAMERA_SIZE="${CAMERA_SIZE:-224}"

OBJECT_DIR="${OBJECT_DIR:-/Data/lyw/libero_processed_v5/libero_object}"
TOKENIZER="${TOKENIZER:-/Data/lyw/fast_tokenizer_all4}"
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
echo "[obj_pretrained_vision] ARM=$ARM SEED=$SEED MAX_STEPS=$MAX_STEPS"
echo "[obj_pretrained_vision] OBJECT_DIR=$OBJECT_DIR"
echo "[obj_pretrained_vision] TOKENIZER=$TOKENIZER"
echo "[obj_pretrained_vision] VISION_ENCODER=$VISION_ENCODER"
echo "[obj_pretrained_vision] CKPT_DIR=$CKPT_DIR"
echo "[obj_pretrained_vision] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================================="

[[ -d "$OBJECT_DIR" ]] || { echo "ERROR: OBJECT_DIR missing: $OBJECT_DIR" >&2; exit 1; }
[[ -d "$TOKENIZER" ]] || { echo "ERROR: TOKENIZER missing: $TOKENIZER" >&2; exit 1; }
[[ -d "$VISION_ENCODER" ]] || {
    echo "ERROR: VISION_ENCODER missing: $VISION_ENCODER" >&2
    echo "Download/copy a HuggingFace vision model there first." >&2
    exit 1
}

mkdir -p "$CKPT_DIR"
TRAIN_LOG="${CKPT_DIR}/train.log"

python train.py \
    data=libero \
    data.dataset.hdf5_dir="$OBJECT_DIR" \
    vision_encoder.source=hf \
    vision_encoder.model_name_or_path="$VISION_ENCODER" \
    vision_encoder.freeze=true \
    vision_encoder.local_files_only=true \
    loss.pred_weight=0 \
    loss.sigreg_weight=0 \
    projector.norm_type=layer \
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
echo "[obj_pretrained_vision] pick_best_ckpt output:"
echo "$PICK_OUT"
BEST_CKPT=$(echo "$PICK_OUT" | python -c "import sys, json; d=json.load(sys.stdin); print(d.get('top_1') or '')")
if [[ -z "$BEST_CKPT" || ! -f "$BEST_CKPT" ]]; then
    echo "ERROR: pick_best_ckpt returned no usable ckpt" >&2
    exit 2
fi

EVAL_LOG="${CKPT_DIR}/eval_libero_object.log"
python eval_libero.py \
    --checkpoint "$BEST_CKPT" \
    --tokenizer "$TOKENIZER" \
    --processed-dir "$OBJECT_DIR" \
    --suite libero_object \
    --num-episodes "$EVAL_EPISODES" \
    --camera-size "$CAMERA_SIZE" \
    --device cuda \
    --seed "$SEED" \
    2>&1 | tee "$EVAL_LOG"

echo "[obj_pretrained_vision] ALL DONE — see logs under $CKPT_DIR/"

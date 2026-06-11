#!/usr/bin/env bash
# run_object_baseline.sh — libero_object-only baseline (diagnostic).
#
# Tests whether libero_object's 0% across all 4-suite variants is caused by
# joint-training distribution shift, or by something more fundamental to
# the object suite (data / tokenizer / OSC execution).
#
# Config matches the spatial-only frozen 65% baseline exactly:
#   LN projector, pred=0, sigreg=0, CLS-only visual, label_smoothing=0.1.
# Step budget = ~100 epoch over object's ~65K chunks at batch=128 ≈ 50K step.
#
# Interpretation:
#   - object-only > 50% → joint training is killing object
#   - object-only ~ 0%  → deeper issue (data layer or tokenizer or eval)
#
set -euo pipefail

ARM="${ARM:-object_baseline}"
SEED="${SEED:-3072}"

MAX_STEPS="${MAX_STEPS:-50000}"      # ~100 epoch over 65K object chunks at bs=128
VAL_INTERVAL="${VAL_INTERVAL:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
CAMERA_SIZE="${CAMERA_SIZE:-224}"

OBJECT_DIR="${OBJECT_DIR:-/Data/lyw/libero_processed_v5/libero_object}"
TOKENIZER="${TOKENIZER:-/Data/lyw/fast_tokenizer_all4}"
CKPT_ROOT="${CKPT_ROOT:-/Data/lyw/stable-wm}"
CKPT_DIR="${CKPT_ROOT}/${ARM}_seed${SEED}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export STABLEWM_HOME="$CKPT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vla

echo "=========================================================="
echo "[obj_baseline] ARM=$ARM SEED=$SEED MAX_STEPS=$MAX_STEPS"
echo "[obj_baseline] OBJECT_DIR=$OBJECT_DIR"
echo "[obj_baseline] TOKENIZER=$TOKENIZER"
echo "[obj_baseline] CKPT_DIR=$CKPT_DIR"
echo "[obj_baseline] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================================="

[[ -d "$OBJECT_DIR" ]] || { echo "ERROR: OBJECT_DIR missing: $OBJECT_DIR" >&2; exit 1; }
[[ -d "$TOKENIZER" ]]  || { echo "ERROR: TOKENIZER missing: $TOKENIZER" >&2; exit 1; }
N=$(ls "$OBJECT_DIR"/*.h5 2>/dev/null | wc -l)
[[ "$N" -ge 10 ]] || { echo "ERROR: expected 10 .h5 in $OBJECT_DIR, got $N" >&2; exit 1; }

mkdir -p "$CKPT_DIR"

TRAIN_LOG="${CKPT_DIR}/train.log"
echo "[obj_baseline] starting training; logs → $TRAIN_LOG"

# LN / pred=0 / sigreg=0 = identical config to the 65% spatial-only baseline.
python train.py \
    data=libero \
    data.dataset.hdf5_dir="$OBJECT_DIR" \
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

echo "[obj_baseline] training done"

PICK_OUT=$(python pick_best_ckpt.py --ckpt-dir "$CKPT_DIR" --top-k 3)
echo "[obj_baseline] pick_best_ckpt output:"
echo "$PICK_OUT"
BEST_CKPT=$(echo "$PICK_OUT" | python -c "import sys, json; d=json.load(sys.stdin); print(d.get('top_1') or '')")
if [[ -z "$BEST_CKPT" || ! -f "$BEST_CKPT" ]]; then
    echo "ERROR: pick_best_ckpt returned no usable ckpt" >&2
    exit 2
fi
echo "[obj_baseline] best ckpt: $BEST_CKPT"

EVAL_LOG="${CKPT_DIR}/eval_libero_object.log"
echo "[obj_baseline] eval libero_object → $EVAL_LOG"
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

echo "[obj_baseline] ALL DONE — see logs under $CKPT_DIR/"

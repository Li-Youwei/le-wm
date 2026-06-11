#!/usr/bin/env bash
# run_suite_policy.sh — Train/evaluate one LIBERO suite as one policy.
#
# This is the OpenVLA/WorldVLA-aligned ablation interface. The main project
# path remains run_all4.sh (4 suites × 1 model); this script trains
# 1 suite × 1 policy.
#
# Usage:
#   SUITE=libero_spatial bash run_suite_policy.sh
#   SUITE=libero_object ARM=baseline SPLIT_MODE=full bash run_suite_policy.sh
set -euo pipefail

SUITE="${SUITE:-libero_spatial}"
case "$SUITE" in
  libero_spatial|libero_object|libero_goal|libero_10) ;;
  *) echo "Unknown SUITE: $SUITE" >&2; exit 1 ;;
esac

ARM="${ARM:-sp_sigreg}"
case "$ARM" in
  baseline)      NORM=layer; PRED=0;   SIGREG=0;   STATE_ARCH_DEFAULT=shared ;;
  sigreg_only)   NORM=batch; PRED=0;   SIGREG=0.1; STATE_ARCH_DEFAULT=shared ;;
  sp_only_ln)    NORM=layer; PRED=1.0; SIGREG=0;   STATE_ARCH_DEFAULT=shared ;;
  sp_sigreg)     NORM=batch; PRED=1.0; SIGREG=0.1; STATE_ARCH_DEFAULT=shared ;;
  sp_sigreg_mot) NORM=batch; PRED=1.0; SIGREG=0.1; STATE_ARCH_DEFAULT=mot    ;;
  *) echo "Unknown ARM: $ARM (expected baseline|sigreg_only|sp_only_ln|sp_sigreg|sp_sigreg_mot)" >&2; exit 1 ;;
esac

SEED="${SEED:-3072}"
MAX_STEPS="${MAX_STEPS:-50000}"
VAL_INTERVAL="${VAL_INTERVAL:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SPLIT_MODE="${SPLIT_MODE:-demo_90_10}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
CAMERA_SIZE="${CAMERA_SIZE:-224}"
STATE_ARCH="${STATE_ARCH:-$STATE_ARCH_DEFAULT}"

PROCESSED_ROOT="${PROCESSED_ROOT:-/Data/lyw/libero_processed_v5}"
PROC_DIR="${PROCESSED_ROOT}/${SUITE}"
TOKENIZER="${TOKENIZER:-/Data/lyw/fast_tokenizer_all4}"
CKPT_ROOT="${CKPT_ROOT:-/Data/lyw/stable-wm}"
CKPT_DIR="${CKPT_ROOT}/${SUITE}_${ARM}_${STATE_ARCH}_split${SPLIT_MODE}_seed${SEED}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export STABLEWM_HOME="$CKPT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vla

echo "=========================================================="
echo "[run_suite_policy] SUITE=$SUITE ARM=$ARM STATE_ARCH=$STATE_ARCH"
echo "[run_suite_policy] SPLIT_MODE=$SPLIT_MODE SEED=$SEED MAX_STEPS=$MAX_STEPS"
echo "[run_suite_policy] PROC_DIR=$PROC_DIR"
echo "[run_suite_policy] TOKENIZER=$TOKENIZER"
echo "[run_suite_policy] CKPT_DIR=$CKPT_DIR"
echo "[run_suite_policy] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================================="

[[ -d "$PROC_DIR" ]] || { echo "ERROR: PROC_DIR missing: $PROC_DIR" >&2; exit 1; }
[[ -d "$TOKENIZER" ]] || { echo "ERROR: TOKENIZER missing: $TOKENIZER" >&2; exit 1; }
N=$(ls "$PROC_DIR"/*.h5 2>/dev/null | wc -l)
if [[ "$N" -lt 10 ]]; then
    echo "ERROR: expected 10 .h5 in $PROC_DIR, got $N" >&2
    exit 1
fi

mkdir -p "$CKPT_DIR"

TRAIN_LOG="${CKPT_DIR}/train.log"
echo "[run_suite_policy] starting training; logs -> $TRAIN_LOG"

python train.py \
    data=libero \
    data.dataset.hdf5_dir="$PROC_DIR" \
    loss.pred_weight="$PRED" \
    loss.sigreg_weight="$SIGREG" \
    predictor.state_prediction_arch="$STATE_ARCH" \
    projector.norm_type="$NORM" \
    scheduler.warmup_steps="$WARMUP_STEPS" \
    split_mode="$SPLIT_MODE" \
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

echo "[run_suite_policy] training done"

if [[ "$SPLIT_MODE" == "full" ]]; then
    BEST_CKPT="${CKPT_DIR}/lewm_final_object.ckpt"
    if [[ ! -f "$BEST_CKPT" ]]; then
        BEST_CKPT="${CKPT_DIR}/lewm_latest_object.ckpt"
    fi
else
    PICK_OUT=$(python pick_best_ckpt.py --ckpt-dir "$CKPT_DIR" --top-k 3)
    echo "[run_suite_policy] pick_best_ckpt output:"
    echo "$PICK_OUT"
    BEST_CKPT=$(echo "$PICK_OUT" | python -c "import sys, json; d=json.load(sys.stdin); print(d.get('top_1') or '')")
fi
if [[ -z "$BEST_CKPT" || ! -f "$BEST_CKPT" ]]; then
    echo "ERROR: no usable ckpt found for SPLIT_MODE=$SPLIT_MODE" >&2
    exit 2
fi
echo "[run_suite_policy] best ckpt: $BEST_CKPT"

EVAL_LOG="${CKPT_DIR}/eval_${SUITE}.log"
echo "[run_suite_policy] eval $SUITE -> $EVAL_LOG"
python eval_libero.py \
    --checkpoint "$BEST_CKPT" \
    --tokenizer "$TOKENIZER" \
    --processed-dir "$PROC_DIR" \
    --suite "$SUITE" \
    --num-episodes "$EVAL_EPISODES" \
    --camera-size "$CAMERA_SIZE" \
    --device cuda \
    --seed "$SEED" \
    2>&1 | tee "$EVAL_LOG"

echo "[run_suite_policy] ALL DONE — see logs under $CKPT_DIR/"

#!/usr/bin/env bash
# run_all4_grip.sh — 4-suite joint training + gripper-aware aux head.
#
# Targets the libero_object 0% failure observed under sp_sigreg / baseline
# 4-suite joint training. Diagnostic (diag_object_actions.py) showed FAST
# joint BPE diluting the gripper signal — predicted gripper command
# oscillated ±1 while GT stayed at -1 for early chunks. The aux head
# bypasses FAST for dim 6 with a direct (H,) MLP regression from the BOS
# hidden state.
#
# Base arm is sp_sigreg (BN projector, pred=1.0, sigreg=0.1) so we can
# diff against the existing all4_sp_sigreg run. Override ARM env var
# to switch base; gripper_aux_weight is always added.
#
# Usage (server):
#   bash run_all4_grip.sh                    # sp_sigreg + grip, seed=3072
#   CUDA_VISIBLE_DEVICES=1 SEED=2024 bash run_all4_grip.sh
#   ARM=baseline bash run_all4_grip.sh       # baseline + grip
#
set -euo pipefail

ARM="${ARM:-sp_sigreg}"
case "$ARM" in
  baseline)    NORM=layer; PRED=0;   SIGREG=0   ;;
  sigreg_only) NORM=batch; PRED=0;   SIGREG=0.1 ;;
  sp_only_ln)  NORM=layer; PRED=1.0; SIGREG=0   ;;
  sp_sigreg)   NORM=batch; PRED=1.0; SIGREG=0.1 ;;
  *) echo "Unknown ARM: $ARM" >&2; exit 1 ;;
esac

GRIP_WEIGHT="${GRIP_WEIGHT:-1.0}"
SEED="${SEED:-3072}"
MAX_STEPS="${MAX_STEPS:-100000}"
VAL_INTERVAL="${VAL_INTERVAL:-4000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
CAMERA_SIZE="${CAMERA_SIZE:-224}"

FLAT_DIR="${FLAT_DIR:-/Data/lyw/libero_processed_v5/all4_flat}"
TOKENIZER="${TOKENIZER:-/Data/lyw/fast_tokenizer_all4}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/Data/lyw/libero_processed_v5}"
CKPT_ROOT="${CKPT_ROOT:-/Data/lyw/stable-wm}"
CKPT_DIR="${CKPT_ROOT}/all4_${ARM}_grip${GRIP_WEIGHT}_seed${SEED}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export STABLEWM_HOME="$CKPT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vla

echo "=========================================================="
echo "[run_all4_grip] ARM=$ARM GRIP_WEIGHT=$GRIP_WEIGHT SEED=$SEED"
echo "[run_all4_grip] FLAT_DIR=$FLAT_DIR"
echo "[run_all4_grip] TOKENIZER=$TOKENIZER"
echo "[run_all4_grip] CKPT_DIR=$CKPT_DIR"
echo "[run_all4_grip] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================================="

[[ -d "$FLAT_DIR" ]]  || { echo "ERROR: FLAT_DIR missing: $FLAT_DIR"  >&2; exit 1; }
[[ -d "$TOKENIZER" ]] || { echo "ERROR: TOKENIZER missing: $TOKENIZER" >&2; exit 1; }
N_FLAT=$(ls "$FLAT_DIR"/*.h5 2>/dev/null | wc -l)
if [[ "$N_FLAT" -lt 40 ]]; then
    echo "ERROR: expected 40 .h5 in $FLAT_DIR, got $N_FLAT" >&2
    exit 1
fi

mkdir -p "$CKPT_DIR"

TRAIN_LOG="${CKPT_DIR}/train.log"
echo "[run_all4_grip] starting training; logs → $TRAIN_LOG"

python train.py \
    data=libero \
    data.dataset.hdf5_dir="$FLAT_DIR" \
    loss.pred_weight="$PRED" \
    loss.sigreg_weight="$SIGREG" \
    loss.gripper_aux_weight="$GRIP_WEIGHT" \
    projector.norm_type="$NORM" \
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

echo "[run_all4_grip] training done"

PICK_OUT=$(python pick_best_ckpt.py --ckpt-dir "$CKPT_DIR" --top-k 3)
echo "[run_all4_grip] pick_best_ckpt output:"
echo "$PICK_OUT"
BEST_CKPT=$(echo "$PICK_OUT" | python -c "import sys, json; d=json.load(sys.stdin); print(d.get('top_1') or '')")
if [[ -z "$BEST_CKPT" || ! -f "$BEST_CKPT" ]]; then
    echo "ERROR: pick_best_ckpt returned no usable ckpt" >&2
    exit 2
fi
echo "[run_all4_grip] best ckpt: $BEST_CKPT"

for suite in libero_spatial libero_object libero_goal libero_10; do
    EVAL_LOG="${CKPT_DIR}/eval_${suite}.log"
    PROC_DIR="${PROCESSED_ROOT}/${suite}"
    echo "[run_all4_grip] eval $suite → $EVAL_LOG"
    python eval_libero.py \
        --checkpoint "$BEST_CKPT" \
        --tokenizer "$TOKENIZER" \
        --processed-dir "$PROC_DIR" \
        --suite "$suite" \
        --num-episodes "$EVAL_EPISODES" \
        --camera-size "$CAMERA_SIZE" \
        --device cuda \
        --seed "$SEED" \
        2>&1 | tee "$EVAL_LOG"
done

echo "[run_all4_grip] ALL DONE — see logs under $CKPT_DIR/"

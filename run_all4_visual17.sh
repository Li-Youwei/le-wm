#!/usr/bin/env bash
# run_all4_visual17.sh — 4-suite joint training × V17 visual prefix × sp_sigreg.
#
# Tests whether the V17 (CLS + 4×4 spatially-pooled patches = 17 tokens/view)
# visual prefix's overfit signature on spatial-only (train CE 4.41→2.64 while
# val CE 3.61@12K→4.26@50K, gap +1.62) dissolves under 4× data from 4-suite
# joint training. Base arm is sp_sigreg (BN projector, pred=1.0, sigreg=0.1)
# to match the baseline all4_sp_sigreg run for direct comparison.
#
# Requires the jepa.py reshape adapter to handle (B, N=17, D) → (B*N, D) → BN
# → (B, 17, D) — wired Apr 2026 alongside the gripper-aux work.
#
# Usage (server):
#   bash run_all4_visual17.sh                       # sp_sigreg + V17, seed=3072
#   CUDA_VISIBLE_DEVICES=3 bash run_all4_visual17.sh
#   ARM=baseline bash run_all4_visual17.sh          # V17 + LN baseline
#
set -euo pipefail

ARM="${ARM:-sp_sigreg}"
case "$ARM" in
  baseline)      NORM=layer; PRED=0;   SIGREG=0;   STATE_ARCH_DEFAULT=shared ;;
  sigreg_only)   NORM=batch; PRED=0;   SIGREG=0.1; STATE_ARCH_DEFAULT=shared ;;
  sp_only_ln)    NORM=layer; PRED=1.0; SIGREG=0;   STATE_ARCH_DEFAULT=shared ;;
  sp_sigreg)     NORM=batch; PRED=1.0; SIGREG=0.1; STATE_ARCH_DEFAULT=shared ;;
  sp_sigreg_mot) NORM=batch; PRED=1.0; SIGREG=0.1; STATE_ARCH_DEFAULT=mot    ;;
  *) echo "Unknown ARM: $ARM" >&2; exit 1 ;;
esac

POOL_GRID="${POOL_GRID:-4}"
SEED="${SEED:-3072}"
MAX_STEPS="${MAX_STEPS:-100000}"
VAL_INTERVAL="${VAL_INTERVAL:-4000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
STATE_ARCH="${STATE_ARCH:-$STATE_ARCH_DEFAULT}"
ARCH_SUFFIX=""
if [[ "$STATE_ARCH" != "shared" && "$ARM" != *"_${STATE_ARCH}"* ]]; then
    ARCH_SUFFIX="_${STATE_ARCH}"
fi

FLAT_DIR="${FLAT_DIR:-/Data/lyw/libero_processed_v5/all4_flat}"
TOKENIZER="${TOKENIZER:-/Data/lyw/fast_tokenizer_all4}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/Data/lyw/libero_processed_v5}"
CKPT_ROOT="${CKPT_ROOT:-/Data/lyw/stable-wm}"
CKPT_DIR="${CKPT_ROOT}/all4_${ARM}_v17${ARCH_SUFFIX}_seed${SEED}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export STABLEWM_HOME="$CKPT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vla

echo "=========================================================="
echo "[run_all4_v17] ARM=$ARM STATE_ARCH=$STATE_ARCH POOL_GRID=$POOL_GRID SEED=$SEED"
echo "[run_all4_v17] FLAT_DIR=$FLAT_DIR"
echo "[run_all4_v17] TOKENIZER=$TOKENIZER"
echo "[run_all4_v17] CKPT_DIR=$CKPT_DIR"
echo "[run_all4_v17] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
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
echo "[run_all4_v17] starting training; logs → $TRAIN_LOG"

python train.py \
    data=libero \
    data.dataset.hdf5_dir="$FLAT_DIR" \
    loss.pred_weight="$PRED" \
    loss.sigreg_weight="$SIGREG" \
    predictor.state_prediction_arch="$STATE_ARCH" \
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
    +visual_tokens.pool_grid="$POOL_GRID" \
    2>&1 | tee "$TRAIN_LOG"

echo "[run_all4_v17] training done"

PICK_OUT=$(python pick_best_ckpt.py --ckpt-dir "$CKPT_DIR" --top-k 3)
echo "[run_all4_v17] pick_best_ckpt output:"
echo "$PICK_OUT"
BEST_CKPT=$(echo "$PICK_OUT" | python -c "import sys, json; d=json.load(sys.stdin); print(d.get('top_1') or '')")
if [[ -z "$BEST_CKPT" || ! -f "$BEST_CKPT" ]]; then
    echo "ERROR: pick_best_ckpt returned no usable ckpt" >&2
    exit 2
fi
echo "[run_all4_v17] best ckpt: $BEST_CKPT"

for suite in libero_spatial libero_object libero_goal libero_10; do
    EVAL_LOG="${CKPT_DIR}/eval_${suite}.log"
    PROC_DIR="${PROCESSED_ROOT}/${suite}"
    echo "[run_all4_v17] eval $suite → $EVAL_LOG"
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

echo "[run_all4_v17] ALL DONE — see logs under $CKPT_DIR/"

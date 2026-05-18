#!/usr/bin/env bash
# run_visual17.sh — Spatial-only baseline (LN / pred=0 / sigreg=0) with the
# CLS + 4x4 spatial-pool visual token prefix (17 tokens/view) to test whether
# richer visual context helps precision-placement tasks (spatial T5/T8/T9).
#
# Compare against the existing spatial-only baseline 65% (CLS-only, same
# everything else), and watch per-task — especially T4 / T5 / T8 / T9 which
# are the "next_to_ramekin / on_plate" precision tasks.
set -euo pipefail

ARM="${ARM:-visual17_baseline}"
SEED="${SEED:-3072}"

# Step budget matched to spatial-only 100 epoch baseline (~50K step).
MAX_STEPS="${MAX_STEPS:-50000}"
VAL_INTERVAL="${VAL_INTERVAL:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
POOL_GRID="${POOL_GRID:-4}"

SPATIAL_DIR="${SPATIAL_DIR:-/Data/lyw/libero_processed_v5/libero_spatial}"
TOKENIZER="${TOKENIZER:-/Data/lyw/fast_tokenizer_all4}"
CKPT_ROOT="${CKPT_ROOT:-/Data/lyw/stable-wm}"
CKPT_DIR="${CKPT_ROOT}/${ARM}_seed${SEED}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export STABLEWM_HOME="$CKPT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vla

echo "=========================================================="
echo "[run_visual17] ARM=$ARM SEED=$SEED MAX_STEPS=$MAX_STEPS POOL_GRID=$POOL_GRID"
echo "[run_visual17] SPATIAL_DIR=$SPATIAL_DIR"
echo "[run_visual17] TOKENIZER=$TOKENIZER"
echo "[run_visual17] CKPT_DIR=$CKPT_DIR"
echo "[run_visual17] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================================="

[[ -d "$SPATIAL_DIR" ]] || { echo "ERROR: SPATIAL_DIR missing: $SPATIAL_DIR" >&2; exit 1; }
[[ -d "$TOKENIZER" ]]   || { echo "ERROR: TOKENIZER missing: $TOKENIZER"   >&2; exit 1; }
N=$(ls "$SPATIAL_DIR"/*.h5 2>/dev/null | wc -l)
[[ "$N" -ge 10 ]] || { echo "ERROR: expected 10 .h5 in $SPATIAL_DIR, got $N" >&2; exit 1; }

mkdir -p "$CKPT_DIR"

TRAIN_LOG="${CKPT_DIR}/train.log"
echo "[run_visual17] starting training; logs → $TRAIN_LOG"

# Hydra strict struct: keys ABSENT from lewm.yaml need `+`. visual_tokens is
# new top-level → `+`. loss.pred_weight, loss.sigreg_weight, projector.norm_type,
# trainer.devices, trainer.max_epochs, loader.batch_size, seed all already exist.
python train.py \
    data=libero \
    data.dataset.hdf5_dir="$SPATIAL_DIR" \
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
    +visual_tokens.pool_grid="$POOL_GRID" \
    2>&1 | tee "$TRAIN_LOG"

echo "[run_visual17] training done"

PICK_OUT=$(python pick_best_ckpt.py --ckpt-dir "$CKPT_DIR" --top-k 3)
echo "[run_visual17] pick_best_ckpt output:"
echo "$PICK_OUT"
BEST_CKPT=$(echo "$PICK_OUT" | python -c "import sys, json; d=json.load(sys.stdin); print(d.get('top_1') or '')")
if [[ -z "$BEST_CKPT" || ! -f "$BEST_CKPT" ]]; then
    echo "ERROR: pick_best_ckpt returned no usable ckpt" >&2
    exit 2
fi
echo "[run_visual17] best ckpt: $BEST_CKPT"

# Eval only libero_spatial (this experiment is single-suite).
EVAL_LOG="${CKPT_DIR}/eval_libero_spatial.log"
echo "[run_visual17] eval libero_spatial → $EVAL_LOG"
python eval_libero.py \
    --checkpoint "$BEST_CKPT" \
    --tokenizer "$TOKENIZER" \
    --processed-dir "$SPATIAL_DIR" \
    --suite libero_spatial \
    --num-episodes 20 \
    --max-steps 300 \
    --device cuda \
    --seed "$SEED" \
    2>&1 | tee "$EVAL_LOG"

echo "[run_visual17] ALL DONE — see logs under $CKPT_DIR/"

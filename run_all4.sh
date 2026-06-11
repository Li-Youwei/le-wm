#!/usr/bin/env bash
# run_all4.sh — Single-arm 4-suite LIBERO joint training driver.
#
# Trains sp_sigreg (BN projector, pred_weight=1.0, sigreg_weight=0.1) on the
# flat 40-task dir for 100K steps, then evaluates the best ckpt on each of the
# 4 suites separately. Phase B (multi-arm baseline / sigreg_only / sp_only_ln)
# is left as a commented `case/esac` block at the top — uncomment to revive.
#
# Usage (on the GPU server):
#   # default: sp_sigreg, seed=3072
#   bash run_all4.sh
#
#   # different seed:
#   SEED=2024 bash run_all4.sh
#
#   # restore multi-arm: edit the ARM env var below before launching.
set -euo pipefail

# ====================================================================
# Phase B (commented out — uncomment + adjust ARM to switch arms)
# ====================================================================
# ARM=${ARM:-sp_sigreg}
# case "$ARM" in
#   baseline)    NORM=layer; PRED=0;   SIGREG=0   ;;
#   sigreg_only) NORM=batch; PRED=0;   SIGREG=0.1 ;;
#   sp_only_ln)  NORM=layer; PRED=1.0; SIGREG=0   ;;
#   sp_sigreg)   NORM=batch; PRED=1.0; SIGREG=0.1 ;;
#   *) echo "Unknown ARM: $ARM" >&2; exit 1 ;;
# esac
# ====================================================================

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
MAX_STEPS="${MAX_STEPS:-100000}"
VAL_INTERVAL="${VAL_INTERVAL:-4000}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SPLIT_MODE="${SPLIT_MODE:-demo_90_10}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
CAMERA_SIZE="${CAMERA_SIZE:-224}"
STATE_ARCH="${STATE_ARCH:-$STATE_ARCH_DEFAULT}"
ARCH_SUFFIX=""
if [[ "$STATE_ARCH" != "shared" && "$ARM" != *"_${STATE_ARCH}"* ]]; then
    ARCH_SUFFIX="_${STATE_ARCH}"
fi

FLAT_DIR="${FLAT_DIR:-/Data/lyw/libero_processed_v5/all4_flat}"
TOKENIZER="${TOKENIZER:-/Data/lyw/fast_tokenizer_all4}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/Data/lyw/libero_processed_v5}"
CKPT_ROOT="${CKPT_ROOT:-/Data/lyw/stable-wm}"
CKPT_DIR="${CKPT_ROOT}/all4_${ARM}${ARCH_SUFFIX}_split${SPLIT_MODE}_seed${SEED}"

# Set GPU explicitly via CUDA_VISIBLE_DEVICES; default GPU 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export STABLEWM_HOME="$CKPT_DIR"

# Conda init (bash -lc / non-interactive shells don't auto-init).
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vla

echo "=========================================================="
echo "[run_all4] ARM=$ARM STATE_ARCH=$STATE_ARCH SEED=$SEED MAX_STEPS=$MAX_STEPS"
echo "[run_all4] SPLIT_MODE=$SPLIT_MODE EVAL_EPISODES=$EVAL_EPISODES CAMERA_SIZE=$CAMERA_SIZE"
echo "[run_all4] FLAT_DIR=$FLAT_DIR"
echo "[run_all4] TOKENIZER=$TOKENIZER"
echo "[run_all4] CKPT_DIR=$CKPT_DIR"
echo "[run_all4] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# Sanity: required inputs exist.
[[ -d "$FLAT_DIR" ]]  || { echo "ERROR: FLAT_DIR missing: $FLAT_DIR"  >&2; exit 1; }
[[ -d "$TOKENIZER" ]] || { echo "ERROR: TOKENIZER missing: $TOKENIZER" >&2; exit 1; }
N_FLAT=$(ls "$FLAT_DIR"/*.h5 2>/dev/null | wc -l)
if [[ "$N_FLAT" -lt 40 ]]; then
    echo "ERROR: expected 40 .h5 in $FLAT_DIR, got $N_FLAT" >&2
    exit 1
fi

mkdir -p "$CKPT_DIR"

# ====================================================================
# Phase 5a — Training (100K steps, step-based, with Stage A probe at 20K)
# ====================================================================
TRAIN_LOG="${CKPT_DIR}/train.log"
echo "[run_all4] starting training; logs → $TRAIN_LOG"

PROBE_SCRIPT="${PROBE_SCRIPT:-$(pwd)/quick_probe_eval.py}"
PROBE_TRIGGER="${PROBE_TRIGGER:-20000}"
PROBE_ENABLED="${PROBE_ENABLED:-true}"

# Hydra strict struct: only keys ABSENT from lewm.yaml need `+` prefix.
# Existing keys in lewm.yaml (loss.pred_weight, loss.sigreg_weight,
# projector.norm_type, scheduler.warmup_steps, loader.batch_size, seed,
# trainer.{devices,max_epochs}) get plain overrides. New keys
# (trainer.max_steps, trainer.val_check_interval, probe.*) get `+`.
python train.py \
    data=libero \
    data.dataset.hdf5_dir="$FLAT_DIR" \
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
    +probe.enabled="$PROBE_ENABLED" \
    +probe.tokenizer_path="$TOKENIZER" \
    +probe.processed_root="$PROCESSED_ROOT" \
    +probe.script="$PROBE_SCRIPT" \
    +probe.trigger_steps="[$PROBE_TRIGGER]" \
    2>&1 | tee "$TRAIN_LOG"

echo "[run_all4] training done"

# ====================================================================
# Phase 5b — Select ckpt
# ====================================================================
if [[ "$SPLIT_MODE" == "full" ]]; then
    BEST_CKPT="${CKPT_DIR}/lewm_final_object.ckpt"
    if [[ ! -f "$BEST_CKPT" ]]; then
        BEST_CKPT="${CKPT_DIR}/lewm_latest_object.ckpt"
    fi
else
    PICK_OUT=$(python pick_best_ckpt.py --ckpt-dir "$CKPT_DIR" --top-k 3)
    echo "[run_all4] pick_best_ckpt output:"
    echo "$PICK_OUT"
    BEST_CKPT=$(echo "$PICK_OUT" | python -c "import sys, json; d=json.load(sys.stdin); print(d.get('top_1') or '')")
fi
if [[ -z "$BEST_CKPT" || ! -f "$BEST_CKPT" ]]; then
    echo "ERROR: no usable ckpt found for SPLIT_MODE=$SPLIT_MODE" >&2
    exit 2
fi
echo "[run_all4] best ckpt: $BEST_CKPT"

# ====================================================================
# Phase 5c — Per-suite eval × 4 (10 tasks × 50 rollouts = 500 trials/suite)
# ====================================================================
for suite in libero_spatial libero_object libero_goal libero_10; do
    EVAL_LOG="${CKPT_DIR}/eval_${suite}.log"
    PROC_DIR="${PROCESSED_ROOT}/${suite}"
    echo "[run_all4] eval $suite → $EVAL_LOG"
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

echo "[run_all4] ALL DONE — see logs under $CKPT_DIR/"

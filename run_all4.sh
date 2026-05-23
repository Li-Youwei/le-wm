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
  baseline)    NORM=layer; PRED=0;   SIGREG=0   ;;
  sigreg_only) NORM=batch; PRED=0;   SIGREG=0.1 ;;
  sp_only_ln)  NORM=layer; PRED=1.0; SIGREG=0   ;;
  sp_sigreg)   NORM=batch; PRED=1.0; SIGREG=0.1 ;;
  *) echo "Unknown ARM: $ARM (expected baseline|sigreg_only|sp_only_ln|sp_sigreg)" >&2; exit 1 ;;
esac

SEED="${SEED:-3072}"
MAX_STEPS="${MAX_STEPS:-100000}"
VAL_INTERVAL="${VAL_INTERVAL:-4000}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"

# Architecture toggles. Defaults reproduce plain sp_sigreg (CLS-only visual,
# shared predictor, random-init trainable ViT-Tiny) — no behavior change unless
# overridden.
POOL_GRID="${POOL_GRID:-0}"                 # >0 enables V17 multi-token visual (G*G pooled patches)
USE_MOT="${USE_MOT:-false}"                 # true enables full Mixture-of-Transformers
VISION_SOURCE="${VISION_SOURCE:-spt_vit}"   # spt_vit | hf (frozen DINOv2 etc. via AutoModel)
VISION_MODEL="${VISION_MODEL:-}"            # required when VISION_SOURCE=hf (e.g. facebook/dinov2-small)
VISION_FREEZE="${VISION_FREEZE:-false}"     # hf path: eval() + requires_grad_(False)
VISION_LOCAL_ONLY="${VISION_LOCAL_ONLY:-true}"  # set false to allow first download

FLAT_DIR="${FLAT_DIR:-/Data/lyw/libero_processed_v5/all4_flat}"
TOKENIZER="${TOKENIZER:-/Data/lyw/fast_tokenizer_all4}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/Data/lyw/libero_processed_v5}"
CKPT_ROOT="${CKPT_ROOT:-/Data/lyw/stable-wm}"
CKPT_DIR="${CKPT_ROOT}/all4_${ARM}_seed${SEED}"

# Set GPU explicitly via CUDA_VISIBLE_DEVICES; default GPU 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export STABLEWM_HOME="$CKPT_DIR"

# Conda init (bash -lc / non-interactive shells don't auto-init).
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vla

echo "=========================================================="
echo "[run_all4] ARM=$ARM SEED=$SEED MAX_STEPS=$MAX_STEPS"
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

PROBE_SCRIPT="${PROBE_SCRIPT:-/Data/lyw/le-wm/quick_probe_eval.py}"
PROBE_TRIGGER="${PROBE_TRIGGER:-20000}"
PROBE_ENABLED="${PROBE_ENABLED:-true}"

# Hydra strict struct: only keys ABSENT from lewm.yaml need `+` prefix.
# Existing keys in lewm.yaml (loss.pred_weight, loss.sigreg_weight,
# projector.norm_type, loader.batch_size, seed, trainer.{devices,max_epochs})
# get plain overrides. New keys (trainer.max_steps, trainer.val_check_interval,
# probe.*) get `+`.
VISION_OVERRIDES=(
    vision_encoder.source="$VISION_SOURCE"
    vision_encoder.freeze="$VISION_FREEZE"
    vision_encoder.local_files_only="$VISION_LOCAL_ONLY"
)
if [[ "$VISION_SOURCE" == "hf" ]]; then
    [[ -n "$VISION_MODEL" ]] || { echo "ERROR: VISION_SOURCE=hf requires VISION_MODEL" >&2; exit 1; }
    VISION_OVERRIDES+=(vision_encoder.model_name_or_path="$VISION_MODEL")
fi

python train.py \
    data=libero \
    data.dataset.hdf5_dir="$FLAT_DIR" \
    predictor.use_mot="$USE_MOT" \
    +visual_tokens.pool_grid="$POOL_GRID" \
    "${VISION_OVERRIDES[@]}" \
    loss.pred_weight="$PRED" \
    loss.sigreg_weight="$SIGREG" \
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
    +probe.enabled="$PROBE_ENABLED" \
    +probe.tokenizer_path="$TOKENIZER" \
    +probe.processed_root="$PROCESSED_ROOT" \
    +probe.script="$PROBE_SCRIPT" \
    +probe.trigger_steps="[$PROBE_TRIGGER]" \
    2>&1 | tee "$TRAIN_LOG"

echo "[run_all4] training done"

# ====================================================================
# Phase 5b — Pick best ckpt by validate/ce_loss_taskbal
# ====================================================================
PICK_OUT=$(python pick_best_ckpt.py --ckpt-dir "$CKPT_DIR" --top-k 3)
echo "[run_all4] pick_best_ckpt output:"
echo "$PICK_OUT"
BEST_CKPT=$(echo "$PICK_OUT" | python -c "import sys, json; d=json.load(sys.stdin); print(d.get('top_1') or '')")
if [[ -z "$BEST_CKPT" || ! -f "$BEST_CKPT" ]]; then
    echo "ERROR: pick_best_ckpt returned no usable ckpt" >&2
    exit 2
fi
echo "[run_all4] best ckpt: $BEST_CKPT"

# ====================================================================
# Phase 5c — Per-suite eval × 4 (~30 min/suite for 10 task × 20 ep)
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
        --num-episodes 20 \
        --max-steps 300 \
        --device cuda \
        --seed 42 \
        2>&1 | tee "$EVAL_LOG"
done

echo "[run_all4] ALL DONE — see logs under $CKPT_DIR/"

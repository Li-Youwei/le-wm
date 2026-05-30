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
WARMUP_STEPS="${WARMUP_STEPS:-4000}"       # informational only — effective warmup is computed in train.py as min(4000, 0.04*max_steps)
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-6}"            # dataloader workers; raise on many-core hosts to keep the GPU fed

# Architecture toggles. Defaults: full 16x16 visual patch grid, shared
# predictor (override USE_MOT=true for MoT), frozen DINOv2-base backbone.
POOL_GRID="${POOL_GRID:-16}"                # multi-token visual: 1 CLS + G*G patches/view. 16 = full 16x16 DINOv2 grid (no pooling, current default); 4 = old V17 (4x4); 0 = CLS-only baseline
USE_MOT="${USE_MOT:-false}"                 # true enables full Mixture-of-Transformers
VISION_SOURCE="${VISION_SOURCE:-hf}"        # hf (frozen DINOv2, default) | spt_vit (legacy ViT-Tiny)
VISION_MODEL="${VISION_MODEL:-facebook/dinov2-base}"  # used when VISION_SOURCE=hf
# freeze default: BOTH sources default to TRAINABLE now. hf (DINOv2) is
# finetuned end-to-end (the #1 LIBERO lever; encoder uses optimizer.encoder_lr).
# spt_vit (random-init ViT-Tiny) must also be trainable (a frozen random-init
# encoder emits garbage). Set VISION_FREEZE=true to keep DINOv2 frozen.
VISION_FREEZE="${VISION_FREEZE:-false}"
VISION_LOCAL_ONLY="${VISION_LOCAL_ONLY:-true}"  # set false on the FIRST run to download DINOv2
# Normalize bool-ish vars to lowercase so True/TRUE/False also work (the offline
# gate below + Hydra bool overrides expect lowercase true/false).
VISION_FREEZE="${VISION_FREEZE,,}"
VISION_LOCAL_ONLY="${VISION_LOCAL_ONLY,,}"

FLAT_DIR="${FLAT_DIR:-/data/lyw/libero_processed_v5/all4_flat}"
TOKENIZER="${TOKENIZER:-/data/lyw/fast_tokenizer_all4}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/data/lyw/libero_processed_v5}"
CKPT_ROOT="${CKPT_ROOT:-/data/lyw/stable-wm}"
CKPT_DIR="${CKPT_ROOT}/all4_${ARM}_seed${SEED}"

# Set GPU explicitly via CUDA_VISIBLE_DEVICES; default GPU 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# HF offline: forced ON when VISION_LOCAL_ONLY=true (cache hit); OFF on the
# first run (VISION_LOCAL_ONLY=false) so DINOv2-base — and, if not yet cached,
# T5-small — can download. After the first run both are cached; keep it true.
if [[ "$VISION_LOCAL_ONLY" == "true" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
else
    export HF_HUB_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
fi
export STABLEWM_HOME="$CKPT_DIR"

# Conda init (bash -lc / non-interactive shells don't auto-init).
source /data/lyw/miniconda3/etc/profile.d/conda.sh
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

PROBE_SCRIPT="${PROBE_SCRIPT:-/data/lyw/le-wm/quick_probe_eval.py}"
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
    num_workers="$NUM_WORKERS" \
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
# Phase 5c — Per-suite eval × 4. Per-suite policy horizon (LIBERO convention:
# spatial 220 / object 280 / goal 300 / 10-long 520) + a fixed no-op warmup to
# let the scene settle (NOT counted toward the horizon). Eval seed = $SEED.
# ====================================================================
EVAL_WARMUP="${EVAL_WARMUP:-10}"
# Receding-horizon eval (optional): execute only the first N steps of each
# predicted chunk, then re-observe + re-predict. Empty = legacy full-chunk exec.
EVAL_EXEC_STEPS="${EVAL_EXEC_STEPS:-}"
EXEC_ARG=()
[[ -n "$EVAL_EXEC_STEPS" ]] && EXEC_ARG=(--exec-steps "$EVAL_EXEC_STEPS")
for suite in libero_spatial libero_object libero_goal libero_10; do
    case "$suite" in
        libero_spatial) SUITE_MAX_STEPS=220 ;;
        libero_object)  SUITE_MAX_STEPS=280 ;;
        libero_goal)    SUITE_MAX_STEPS=300 ;;
        libero_10)      SUITE_MAX_STEPS=520 ;;
        *)              SUITE_MAX_STEPS=300 ;;
    esac
    EVAL_LOG="${CKPT_DIR}/eval_${suite}.log"
    PROC_DIR="${PROCESSED_ROOT}/${suite}"
    echo "[run_all4] eval $suite (max_steps=$SUITE_MAX_STEPS warmup=$EVAL_WARMUP seed=$SEED) → $EVAL_LOG"
    python eval_libero.py \
        --checkpoint "$BEST_CKPT" \
        --tokenizer "$TOKENIZER" \
        --processed-dir "$PROC_DIR" \
        --suite "$suite" \
        --num-episodes 50 \
        --max-steps "$SUITE_MAX_STEPS" \
        --num-warmup-steps "$EVAL_WARMUP" \
        "${EXEC_ARG[@]}" \
        --device cuda \
        --seed "$SEED" \
        2>&1 | tee "$EVAL_LOG"
done

echo "[run_all4] ALL DONE — see logs under $CKPT_DIR/"

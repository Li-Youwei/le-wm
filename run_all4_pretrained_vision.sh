#!/usr/bin/env bash
# run_all4_pretrained_vision.sh - 4-suite LIBERO training with a frozen
# HuggingFace vision backbone. Default runs SP+SIGReg with frozen DINOv2.
set -euo pipefail

ARM="${ARM:-all4_sp_sigreg_dinov2_frozen}"
case "$ARM" in
  all4_dinov2_frozen)
    PRED=0
    SIGREG=0
    NORM=layer
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=0
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_sigreg_dinov2_frozen)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=0
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_sigreg_dinov2_frozen_visual17)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=4
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_sigreg_dinov2_frozen_visual17_sep_proj)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=4
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_sigreg_dinov2_frozen_visual17_patch_sp)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=4
    PATCH_SP_DEFAULT=true
    ;;
  all4_sp_sigreg_dinov2_frozen_visual17_patch_sp_mot)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=mot
    POOL_GRID_DEFAULT=4
    PATCH_SP_DEFAULT=true
    ;;
  all4_sp_sigreg_dinov2_frozen_mot)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=mot
    POOL_GRID_DEFAULT=0
    PATCH_SP_DEFAULT=false
    ;;
  *)
    echo "Unknown ARM: $ARM" >&2
    echo "Expected all4_dinov2_frozen|all4_sp_sigreg_dinov2_frozen|all4_sp_sigreg_dinov2_frozen_visual17|all4_sp_sigreg_dinov2_frozen_visual17_sep_proj|all4_sp_sigreg_dinov2_frozen_visual17_patch_sp|all4_sp_sigreg_dinov2_frozen_visual17_patch_sp_mot|all4_sp_sigreg_dinov2_frozen_mot" >&2
    exit 1
    ;;
esac

SEED="${SEED:-3072}"
MAX_STEPS="${MAX_STEPS:-100000}"
VAL_INTERVAL="${VAL_INTERVAL:-4000}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
STATE_ARCH="${STATE_ARCH:-$STATE_ARCH_DEFAULT}"
POOL_GRID="${POOL_GRID:-$POOL_GRID_DEFAULT}"
PATCH_SP="${PATCH_SP:-$PATCH_SP_DEFAULT}"
PATCH_SP_WEIGHT="${PATCH_SP_WEIGHT:-1.0}"
CKPT_SELECT_TOP_K="${CKPT_SELECT_TOP_K:-3}"
LIGHT_EVAL_EPISODES="${LIGHT_EVAL_EPISODES:-5}"
LIGHT_EVAL_MAX_STEPS="${LIGHT_EVAL_MAX_STEPS:-300}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-20}"
ACTION_CODEC="${ACTION_CODEC:-fast}"
NUM_ACTION_BINS="${NUM_ACTION_BINS:-256}"
ACTION_DIM="${ACTION_DIM:-7}"
if [[ -z "${MAX_ACTION_TOKENS+x}" ]]; then
    if [[ "$ACTION_CODEC" == "worldvla_bins" ]]; then
        MAX_ACTION_TOKENS=$((20 * ACTION_DIM))
    else
        MAX_ACTION_TOKENS=80
    fi
fi

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
echo "[all4_pretrained_vision] ARM=$ARM STATE_ARCH=$STATE_ARCH SEED=$SEED MAX_STEPS=$MAX_STEPS"
echo "[all4_pretrained_vision] PRED=$PRED SIGREG=$SIGREG NORM=$NORM"
echo "[all4_pretrained_vision] POOL_GRID=$POOL_GRID PATCH_SP=$PATCH_SP PATCH_SP_WEIGHT=$PATCH_SP_WEIGHT"
echo "[all4_pretrained_vision] FLAT_DIR=$FLAT_DIR"
echo "[all4_pretrained_vision] TOKENIZER=$TOKENIZER"
echo "[all4_pretrained_vision] PROCESSED_ROOT=$PROCESSED_ROOT"
echo "[all4_pretrained_vision] VISION_ENCODER=$VISION_ENCODER"
echo "[all4_pretrained_vision] CKPT_DIR=$CKPT_DIR"
echo "[all4_pretrained_vision] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[all4_pretrained_vision] CKPT_SELECT_TOP_K=$CKPT_SELECT_TOP_K LIGHT_EVAL_EPISODES=$LIGHT_EVAL_EPISODES FINAL_EVAL_EPISODES=$FINAL_EVAL_EPISODES"
echo "[all4_pretrained_vision] ACTION_CODEC=$ACTION_CODEC NUM_ACTION_BINS=$NUM_ACTION_BINS MAX_ACTION_TOKENS=$MAX_ACTION_TOKENS"
echo "=========================================================="

[[ -d "$FLAT_DIR" ]] || { echo "ERROR: FLAT_DIR missing: $FLAT_DIR" >&2; exit 1; }
if [[ "$ACTION_CODEC" != "fast" && "$ACTION_CODEC" != "worldvla_bins" ]]; then
    echo "ERROR: ACTION_CODEC must be fast or worldvla_bins, got: $ACTION_CODEC" >&2
    exit 1
fi
if [[ "$ACTION_CODEC" == "fast" ]]; then
    [[ -d "$TOKENIZER" ]] || { echo "ERROR: TOKENIZER missing: $TOKENIZER" >&2; exit 1; }
fi
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

EVAL_CODEC_ARGS=(--action-codec "$ACTION_CODEC" --num-action-bins "$NUM_ACTION_BINS")
if [[ "$ACTION_CODEC" == "fast" ]]; then
    EVAL_CODEC_ARGS+=(--tokenizer "$TOKENIZER")
fi

python train.py \
    data=libero \
    data.dataset.hdf5_dir="$FLAT_DIR" \
    data.dataset.max_action_tokens="$MAX_ACTION_TOKENS" \
    data.dataset.action_dim="$ACTION_DIM" \
    data.dataset.action_codec.type="$ACTION_CODEC" \
    data.dataset.action_codec.num_bins="$NUM_ACTION_BINS" \
    vision_encoder.source=hf \
    vision_encoder.model_name_or_path="$VISION_ENCODER" \
    vision_encoder.freeze=true \
    vision_encoder.local_files_only=true \
    loss.pred_weight="$PRED" \
    loss.sigreg_weight="$SIGREG" \
    predictor.state_prediction_arch="$STATE_ARCH" \
    projector.norm_type="$NORM" \
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
    +ckpt_top_k="$CKPT_SELECT_TOP_K" \
    +visual_tokens.pool_grid="$POOL_GRID" \
    +visual_tokens.patch_sp="$PATCH_SP" \
    +visual_tokens.patch_sp_weight="$PATCH_SP_WEIGHT" \
    2>&1 | tee "$TRAIN_LOG"

PICK_JSON="${CKPT_DIR}/ckpt_ce_topk.json"
PICK_OUT=$(python pick_best_ckpt.py --ckpt-dir "$CKPT_DIR" --top-k "$CKPT_SELECT_TOP_K")
echo "[all4_pretrained_vision] pick_best_ckpt output:"
echo "$PICK_OUT"
echo "$PICK_OUT" > "$PICK_JSON"

mapfile -t CANDIDATE_CKPTS < <(python - "$PICK_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
seen = set()
for row in data.get("all", []):
    ckpt = row.get("ckpt")
    if ckpt and ckpt not in seen:
        seen.add(ckpt)
        print(ckpt)
PY
)
if [[ "${#CANDIDATE_CKPTS[@]}" -eq 0 ]]; then
    echo "ERROR: pick_best_ckpt returned no candidate checkpoints" >&2
    exit 2
fi

for candidate_ckpt in "${CANDIDATE_CKPTS[@]}"; do
    if [[ ! -f "$candidate_ckpt" ]]; then
        echo "ERROR: candidate ckpt missing: $candidate_ckpt" >&2
        exit 2
    fi
    candidate_stem="$(basename "$candidate_ckpt" .ckpt)"
    echo "[all4_pretrained_vision] light eval candidate: $candidate_ckpt"
    for suite in libero_spatial libero_object libero_goal libero_10; do
        EVAL_LOG="${CKPT_DIR}/light_eval_${candidate_stem}_${suite}.log"
        PROC_DIR="${PROCESSED_ROOT}/${suite}"
        [[ -d "$PROC_DIR" ]] || { echo "ERROR: processed suite dir missing: $PROC_DIR" >&2; exit 1; }
        echo "[all4_pretrained_vision] light eval $suite -> $EVAL_LOG"
        python eval_libero.py \
            --checkpoint "$candidate_ckpt" \
            --processed-dir "$PROC_DIR" \
            --suite "$suite" \
            --num-episodes "$LIGHT_EVAL_EPISODES" \
            --max-steps "$LIGHT_EVAL_MAX_STEPS" \
            --device cuda \
            --seed "$SEED" \
            "${EVAL_CODEC_ARGS[@]}" \
            2>&1 | tee "$EVAL_LOG"
    done
done

SELECT_JSON="${CKPT_DIR}/ckpt_light_eval_selection.json"
SELECT_OUT=$(python select_light_eval_ckpt.py \
    --ckpt-dir "$CKPT_DIR" \
    --candidates-json "$PICK_JSON" \
    --out "$SELECT_JSON")
echo "[all4_pretrained_vision] select_light_eval_ckpt output:"
echo "$SELECT_OUT"
BEST_CKPT=$(echo "$SELECT_OUT" | python -c "import sys, json; d=json.load(sys.stdin); print(d.get('top_1') or '')")
if [[ -z "$BEST_CKPT" || ! -f "$BEST_CKPT" ]]; then
    echo "ERROR: light rollout selection returned no usable ckpt" >&2
    exit 2
fi
echo "[all4_pretrained_vision] best ckpt by light rollout: $BEST_CKPT"

for suite in libero_spatial libero_object libero_goal libero_10; do
    EVAL_LOG="${CKPT_DIR}/eval_${suite}.log"
    PROC_DIR="${PROCESSED_ROOT}/${suite}"
    [[ -d "$PROC_DIR" ]] || { echo "ERROR: processed suite dir missing: $PROC_DIR" >&2; exit 1; }
    echo "[all4_pretrained_vision] eval $suite -> $EVAL_LOG"
    python eval_libero.py \
        --checkpoint "$BEST_CKPT" \
        --processed-dir "$PROC_DIR" \
        --suite "$suite" \
        --num-episodes "$FINAL_EVAL_EPISODES" \
        --max-steps 300 \
        --device cuda \
        --seed "$SEED" \
        "${EVAL_CODEC_ARGS[@]}" \
        2>&1 | tee "$EVAL_LOG"
done

echo "[all4_pretrained_vision] ALL DONE - see logs under $CKPT_DIR/"

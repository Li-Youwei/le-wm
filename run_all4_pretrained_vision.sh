#!/usr/bin/env bash
# run_all4_pretrained_vision.sh - 4-suite LIBERO training with a frozen
# HuggingFace vision backbone. Default runs the paper baseline arm:
# frozen DINOv2-base, CLS+4x4 patch visual tokens, patch SP, SP, and SIGReg.
set -euo pipefail

ARM="${ARM:-all4_sp_sigreg_dinov2_frozen_visual17_patch_sp}"
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
SPLIT_MODE="${SPLIT_MODE:-demo_90_10}"
CKPT_TOP_K="${CKPT_TOP_K:-6}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
CAMERA_SIZE="${CAMERA_SIZE:-224}"
STATE_ARCH="${STATE_ARCH:-$STATE_ARCH_DEFAULT}"
POOL_GRID="${POOL_GRID:-$POOL_GRID_DEFAULT}"
PATCH_SP="${PATCH_SP:-$PATCH_SP_DEFAULT}"
PATCH_SP_WEIGHT="${PATCH_SP_WEIGHT:-1.0}"

FLAT_DIR="${FLAT_DIR:-/Data/lyw/libero_processed_v5/all4_flat}"
TOKENIZER="${TOKENIZER:-/Data/lyw/fast_tokenizer_all4}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/Data/lyw/libero_processed_v5}"
VISION_ENCODER="${VISION_ENCODER:-facebook/dinov2-base}"
CKPT_ROOT="${CKPT_ROOT:-/Data/lyw/stable-wm}"
CKPT_DIR="${CKPT_ROOT}/${ARM}_split${SPLIT_MODE}_seed${SEED}"

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
echo "[all4_pretrained_vision] SPLIT_MODE=$SPLIT_MODE CKPT_TOP_K=$CKPT_TOP_K"
echo "[all4_pretrained_vision] FLAT_DIR=$FLAT_DIR"
echo "[all4_pretrained_vision] TOKENIZER=$TOKENIZER"
echo "[all4_pretrained_vision] PROCESSED_ROOT=$PROCESSED_ROOT"
echo "[all4_pretrained_vision] VISION_ENCODER=$VISION_ENCODER"
echo "[all4_pretrained_vision] CKPT_DIR=$CKPT_DIR"
echo "[all4_pretrained_vision] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================================="

[[ -d "$FLAT_DIR" ]] || { echo "ERROR: FLAT_DIR missing: $FLAT_DIR" >&2; exit 1; }
[[ -d "$TOKENIZER" ]] || { echo "ERROR: TOKENIZER missing: $TOKENIZER" >&2; exit 1; }
if [[ -d "$VISION_ENCODER" ]]; then
    echo "[all4_pretrained_vision] using local vision encoder dir: $VISION_ENCODER"
else
    python - "$VISION_ENCODER" <<'PY'
import sys
from transformers import AutoConfig

AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)
PY
fi
N_FLAT=$(find "$FLAT_DIR" -maxdepth 1 -name "*.h5" | wc -l)
if [[ "$N_FLAT" -lt 40 ]]; then
    echo "ERROR: expected 40 .h5 in $FLAT_DIR, got $N_FLAT" >&2
    exit 1
fi

mkdir -p "$CKPT_DIR"
TRAIN_LOG="${CKPT_DIR}/train.log"

python train.py \
    data=libero \
    data.dataset.hdf5_dir="$FLAT_DIR" \
    vision_encoder.source=hf \
    vision_encoder.model_name_or_path="$VISION_ENCODER" \
    vision_encoder.freeze=true \
    vision_encoder.local_files_only=true \
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
    +ckpt_top_k="$CKPT_TOP_K" \
    +visual_tokens.pool_grid="$POOL_GRID" \
    +visual_tokens.patch_sp="$PATCH_SP" \
    +visual_tokens.patch_sp_weight="$PATCH_SP_WEIGHT" \
    2>&1 | tee "$TRAIN_LOG"

eval_checkpoint() {
    local rank="$1"
    local ckpt="$2"
    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
        echo "ERROR: no usable ckpt for rank=$rank: $ckpt" >&2
        exit 2
    fi
    echo "[all4_pretrained_vision] eval rank=$rank ckpt=$ckpt"
    for suite in libero_spatial libero_object libero_goal libero_10; do
        EVAL_LOG="${CKPT_DIR}/eval_rank${rank}_${suite}.log"
        PROC_DIR="${PROCESSED_ROOT}/${suite}"
        [[ -d "$PROC_DIR" ]] || { echo "ERROR: processed suite dir missing: $PROC_DIR" >&2; exit 1; }
        echo "[all4_pretrained_vision] eval rank=$rank $suite -> $EVAL_LOG"
        python eval_libero.py \
            --checkpoint "$ckpt" \
            --tokenizer "$TOKENIZER" \
            --processed-dir "$PROC_DIR" \
            --suite "$suite" \
            --num-episodes "$EVAL_EPISODES" \
            --camera-size "$CAMERA_SIZE" \
            --device cuda \
            --seed "$SEED" \
            2>&1 | tee "$EVAL_LOG"
    done
}

list_full_ckpts() {
    python - "$CKPT_DIR" <<'PY'
import re
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
step_re = re.compile(r"lewm_step_(\d+)_object\.ckpt$")
step_ckpts = []
for path in ckpt_dir.glob("lewm_step_*_object.ckpt"):
    match = step_re.match(path.name)
    if match:
        step_ckpts.append((int(match.group(1)), path))
for step, path in sorted(step_ckpts):
    print(f"step{step}\t{path}")
final = ckpt_dir / "lewm_final_object.ckpt"
if final.exists():
    print(f"final\t{final}")
PY
}

if [[ "$SPLIT_MODE" == "full" ]]; then
    found_ckpt=0
    while IFS=$'\t' read -r RANK CKPT; do
        [[ -n "${RANK:-}" && -n "${CKPT:-}" ]] || continue
        found_ckpt=1
        eval_checkpoint "$RANK" "$CKPT"
    done < <(list_full_ckpts)
    if [[ "$found_ckpt" -eq 0 ]]; then
        BEST_CKPT="${CKPT_DIR}/lewm_latest_object.ckpt"
        eval_checkpoint "final" "$BEST_CKPT"
    fi
else
    PICK_OUT=$(python pick_best_ckpt.py --ckpt-dir "$CKPT_DIR" --top-k "$CKPT_TOP_K")
    PICK_JSON="${CKPT_DIR}/pick_best_top${CKPT_TOP_K}.json"
    printf "%s\n" "$PICK_OUT" > "$PICK_JSON"
    echo "[all4_pretrained_vision] pick_best_ckpt output:"
    echo "$PICK_OUT"
    found_ckpt=0
    while IFS=$'\t' read -r RANK CKPT; do
        [[ -n "${RANK:-}" && -n "${CKPT:-}" ]] || continue
        found_ckpt=1
        eval_checkpoint "$RANK" "$CKPT"
    done < <(python - "$PICK_JSON" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
for entry in data.get("all", []):
    ckpt = entry.get("ckpt")
    if ckpt:
        print(f"{entry.get('rank')}\t{ckpt}")
PY
)
    if [[ "$found_ckpt" -eq 0 ]]; then
        echo "ERROR: pick_best_ckpt returned no usable ckpts" >&2
        exit 2
    fi
fi

echo "[all4_pretrained_vision] ALL DONE - see logs under $CKPT_DIR/"

#!/usr/bin/env bash
# run_all4_pretrained_vision.sh - suitewise LIBERO training with a frozen
# HuggingFace vision backbone. Default trains 4 independent SP+SIGReg policies.
set -euo pipefail

ARM="${ARM:-all4_sp_sigreg_dinov2_frozen}"
case "$ARM" in
  all4_dinov2_frozen)
    PRED=0
    SIGREG=0
    NORM=layer
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=0
    PREDICTOR_DEPTH_DEFAULT=6
    PATCH_SP_DEFAULT=false
    ;;
  all4_dinov2_frozen_visual17)
    PRED=0
    SIGREG=0
    NORM=layer
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=4
    PREDICTOR_DEPTH_DEFAULT=6
    PATCH_SP_DEFAULT=false
    ;;
  all4_dinov2_frozen_visual257)
    PRED=0
    SIGREG=0
    NORM=layer
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=16
    PREDICTOR_DEPTH_DEFAULT=12
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_sigreg_dinov2_frozen)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=0
    PREDICTOR_DEPTH_DEFAULT=6
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_sigreg_dinov2_frozen_visual17)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=4
    PREDICTOR_DEPTH_DEFAULT=6
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_sigreg_dinov2_frozen_visual257)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=16
    PREDICTOR_DEPTH_DEFAULT=12
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_sigreg_dinov2_frozen_visual17_sep_proj)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=4
    PREDICTOR_DEPTH_DEFAULT=6
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_sigreg_dinov2_frozen_visual257_sep_proj)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=16
    PREDICTOR_DEPTH_DEFAULT=12
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_sigreg_dinov2_frozen_visual17_patch_sp)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=4
    PREDICTOR_DEPTH_DEFAULT=6
    PATCH_SP_DEFAULT=true
    ;;
  all4_sp_sigreg_dinov2_frozen_visual257_patch_sp)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=16
    PREDICTOR_DEPTH_DEFAULT=12
    PATCH_SP_DEFAULT=true
    ;;
  all4_sp_sigreg_dinov2_frozen_visual17_patch_sp_mot)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=mot
    POOL_GRID_DEFAULT=4
    PREDICTOR_DEPTH_DEFAULT=6
    PATCH_SP_DEFAULT=true
    ;;
  all4_sp_sigreg_dinov2_frozen_visual257_patch_sp_mot)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=mot
    POOL_GRID_DEFAULT=16
    PREDICTOR_DEPTH_DEFAULT=12
    PATCH_SP_DEFAULT=true
    ;;
  all4_sp_sigreg_dinov2_frozen_mot)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=mot
    POOL_GRID_DEFAULT=0
    PREDICTOR_DEPTH_DEFAULT=6
    PATCH_SP_DEFAULT=false
    ;;
  *)
    echo "Unknown ARM: $ARM" >&2
    echo "Expected all4_dinov2_frozen|all4_dinov2_frozen_visual17|all4_dinov2_frozen_visual257|all4_sp_sigreg_dinov2_frozen|all4_sp_sigreg_dinov2_frozen_visual17|all4_sp_sigreg_dinov2_frozen_visual257|all4_sp_sigreg_dinov2_frozen_visual17_sep_proj|all4_sp_sigreg_dinov2_frozen_visual257_sep_proj|all4_sp_sigreg_dinov2_frozen_visual17_patch_sp|all4_sp_sigreg_dinov2_frozen_visual257_patch_sp|all4_sp_sigreg_dinov2_frozen_visual17_patch_sp_mot|all4_sp_sigreg_dinov2_frozen_visual257_patch_sp_mot|all4_sp_sigreg_dinov2_frozen_mot" >&2
    exit 1
    ;;
esac

SEED="${SEED:-3072}"
MAX_STEPS="${MAX_STEPS:-60000}"
VAL_INTERVAL="${VAL_INTERVAL:-4000}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
DDP_DEVICES="${DDP_DEVICES:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
if [[ -n "${PER_DEVICE_BATCH_SIZE:-}" ]]; then
    :
elif [[ -n "${BATCH_SIZE:-}" ]]; then
    PER_DEVICE_BATCH_SIZE="$BATCH_SIZE"
else
    PER_DEVICE_BATCH_SIZE=16
fi
BATCH_SIZE="${BATCH_SIZE:-$PER_DEVICE_BATCH_SIZE}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
TRAINER_STRATEGY="${TRAINER_STRATEGY:-ddp}"
SYNC_BATCHNORM="${SYNC_BATCHNORM:-true}"
STATE_ARCH="${STATE_ARCH:-$STATE_ARCH_DEFAULT}"
POOL_GRID="${POOL_GRID:-$POOL_GRID_DEFAULT}"
PREDICTOR_DEPTH="${PREDICTOR_DEPTH:-$PREDICTOR_DEPTH_DEFAULT}"
PATCH_SP="${PATCH_SP:-$PATCH_SP_DEFAULT}"
PATCH_SP_WEIGHT="${PATCH_SP_WEIGHT:-1.0}"
CKPT_SELECT_TOP_K="${CKPT_SELECT_TOP_K:-3}"
LIGHT_EVAL_EPISODES="${LIGHT_EVAL_EPISODES:-5}"
LIGHT_EVAL_MAX_STEPS="${LIGHT_EVAL_MAX_STEPS:-}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-50}"
FINAL_EVAL_MAX_STEPS="${FINAL_EVAL_MAX_STEPS:-}"
EVAL_MAX_STEPS_LIBERO_SPATIAL="${EVAL_MAX_STEPS_LIBERO_SPATIAL:-220}"
EVAL_MAX_STEPS_LIBERO_OBJECT="${EVAL_MAX_STEPS_LIBERO_OBJECT:-280}"
EVAL_MAX_STEPS_LIBERO_GOAL="${EVAL_MAX_STEPS_LIBERO_GOAL:-300}"
EVAL_MAX_STEPS_LIBERO_10="${EVAL_MAX_STEPS_LIBERO_10:-520}"
TRAIN_SPLIT="${TRAIN_SPLIT:-1.0}"
FULL_TRAIN_TARGET_STEP="${FULL_TRAIN_TARGET_STEP:-60000}"

TOKENIZER="${TOKENIZER:-/Data/lyw/fast_tokenizer_all4}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/Data/lyw/libero_processed_v5}"
VISION_ENCODER="${VISION_ENCODER:-/Data/lyw/hf_models/facebook-dinov2-base}"
CKPT_ROOT="${CKPT_ROOT:-/Data/lyw/stable-wm}"
DEFAULT_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
if [[ -n "${SUITES:-}" ]]; then
    read -r -a RUN_SUITES <<< "$SUITES"
else
    RUN_SUITES=("${DEFAULT_SUITES[@]}")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vla

echo "=========================================================="
echo "[all4_pretrained_vision] ARM=$ARM STATE_ARCH=$STATE_ARCH SEED=$SEED MAX_STEPS=$MAX_STEPS"
echo "[all4_pretrained_vision] PRED=$PRED SIGREG=$SIGREG NORM=$NORM"
echo "[all4_pretrained_vision] POOL_GRID=$POOL_GRID PREDICTOR_DEPTH=$PREDICTOR_DEPTH PATCH_SP=$PATCH_SP PATCH_SP_WEIGHT=$PATCH_SP_WEIGHT"
echo "[all4_pretrained_vision] TOKENIZER=$TOKENIZER"
echo "[all4_pretrained_vision] PROCESSED_ROOT=$PROCESSED_ROOT"
echo "[all4_pretrained_vision] VISION_ENCODER=$VISION_ENCODER"
echo "[all4_pretrained_vision] SUITES=${RUN_SUITES[*]}"
echo "[all4_pretrained_vision] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[all4_pretrained_vision] DDP_DEVICES=$DDP_DEVICES TRAINER_STRATEGY=$TRAINER_STRATEGY SYNC_BATCHNORM=$SYNC_BATCHNORM"
echo "[all4_pretrained_vision] GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES"
echo "[all4_pretrained_vision] BATCH_SIZE alias=$BATCH_SIZE (per-device)"
echo "[all4_pretrained_vision] TRAIN_SPLIT=$TRAIN_SPLIT"
echo "[all4_pretrained_vision] FULL_TRAIN_TARGET_STEP=$FULL_TRAIN_TARGET_STEP"
echo "[all4_pretrained_vision] CKPT_SELECT_TOP_K=$CKPT_SELECT_TOP_K LIGHT_EVAL_EPISODES=$LIGHT_EVAL_EPISODES FINAL_EVAL_EPISODES=$FINAL_EVAL_EPISODES"
echo "[all4_pretrained_vision] EVAL_MAX_STEPS spatial=$EVAL_MAX_STEPS_LIBERO_SPATIAL object=$EVAL_MAX_STEPS_LIBERO_OBJECT goal=$EVAL_MAX_STEPS_LIBERO_GOAL long=$EVAL_MAX_STEPS_LIBERO_10"
echo "=========================================================="

[[ -d "$TOKENIZER" ]] || { echo "ERROR: TOKENIZER missing: $TOKENIZER" >&2; exit 1; }
[[ -d "$VISION_ENCODER" ]] || {
    echo "ERROR: VISION_ENCODER missing: $VISION_ENCODER" >&2
    echo "Download/copy a HuggingFace vision model there first." >&2
    exit 1
}

eval_max_steps_for_suite() {
    local suite="$1"
    case "$suite" in
        libero_spatial) echo "$EVAL_MAX_STEPS_LIBERO_SPATIAL" ;;
        libero_object) echo "$EVAL_MAX_STEPS_LIBERO_OBJECT" ;;
        libero_goal) echo "$EVAL_MAX_STEPS_LIBERO_GOAL" ;;
        libero_10) echo "$EVAL_MAX_STEPS_LIBERO_10" ;;
        *)
            echo "ERROR: unknown suite for eval max steps: $suite" >&2
            return 1
            ;;
    esac
}

light_eval_max_steps_for_suite() {
    local suite="$1"
    if [[ -n "$LIGHT_EVAL_MAX_STEPS" ]]; then
        echo "$LIGHT_EVAL_MAX_STEPS"
    else
        eval_max_steps_for_suite "$suite"
    fi
}

final_eval_max_steps_for_suite() {
    local suite="$1"
    if [[ -n "$FINAL_EVAL_MAX_STEPS" ]]; then
        echo "$FINAL_EVAL_MAX_STEPS"
    else
        eval_max_steps_for_suite "$suite"
    fi
}

write_step_candidates_json() {
    local ckpt_dir="$1"
    local out_json="$2"
    local target_step="$3"
    python - "$ckpt_dir" "$out_json" "$target_step" <<'PY'
import json
import re
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
target = int(float(sys.argv[3]))
rows = []
for path in sorted(ckpt_dir.glob("lewm_step_*_object.ckpt")):
    match = re.search(r"lewm_step_(\d+)_object\.ckpt$", path.name)
    if not match:
        continue
    step = int(match.group(1))
    rows.append(
        {
            "ckpt": str(path),
            "step": step,
            "value": abs(step - target),
        }
    )
rows.sort(key=lambda row: (row["value"], row["step"]))
text = json.dumps(
    {
        "selection_metric": "step_distance_to_full_train_target",
        "target_step": target,
        "all": rows,
    },
    indent=2,
)
print(text)
out_path.write_text(text + "\n")
PY
}

print_candidate_ckpts() {
    local pick_json="$1"
    python - "$pick_json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
for row in data.get("all", []):
    ckpt = row.get("ckpt")
    if ckpt:
        print(ckpt)
PY
}

run_suite_eval() {
    local phase="$1"
    local suite="$2"
    local ckpt_dir="$3"
    local checkpoint="$4"
    local proc_dir="${PROCESSED_ROOT}/${suite}"
    local episodes
    local max_steps
    local eval_log
    local candidate_stem

    [[ -d "$proc_dir" ]] || { echo "ERROR: processed suite dir missing: $proc_dir" >&2; exit 1; }
    if [[ "$phase" == "light" ]]; then
        episodes="$LIGHT_EVAL_EPISODES"
        max_steps="$(light_eval_max_steps_for_suite "$suite")"
        candidate_stem="$(basename "$checkpoint" .ckpt)"
        eval_log="${ckpt_dir}/light_eval_${candidate_stem}_${suite}.log"
    else
        episodes="$FINAL_EVAL_EPISODES"
        max_steps="$(final_eval_max_steps_for_suite "$suite")"
        eval_log="${ckpt_dir}/eval_${suite}.log"
    fi

    echo "[all4_pretrained_vision] $phase eval $suite episodes=$episodes max_steps=$max_steps -> $eval_log"
    python eval_libero.py \
        --checkpoint "$checkpoint" \
        --tokenizer "$TOKENIZER" \
        --processed-dir "$proc_dir" \
        --suite "$suite" \
        --num-episodes "$episodes" \
        --max-steps "$max_steps" \
        --device cuda \
        --seed "$SEED" \
        2>&1 | tee "$eval_log"
}

select_best_ckpt_for_suite() {
    local suite="$1"
    local ckpt_dir="$2"
    local pick_json
    local pick_out
    local candidate_ckpt
    local select_json
    local select_out
    local -a candidate_ckpts=()

    if [[ "$TRAIN_SPLIT" == "1" || "$TRAIN_SPLIT" == "1.0" ]]; then
        pick_json="${ckpt_dir}/ckpt_step_candidates.json"
        write_step_candidates_json "$ckpt_dir" "$pick_json" "$FULL_TRAIN_TARGET_STEP"
        echo "[all4_pretrained_vision] full-train mode: evaluating step candidates from $pick_json"
    else
        pick_json="${ckpt_dir}/ckpt_ce_topk.json"
        pick_out=$(python pick_best_ckpt.py --ckpt-dir "$ckpt_dir" --top-k "$CKPT_SELECT_TOP_K")
        echo "[all4_pretrained_vision] pick_best_ckpt output:"
        echo "$pick_out"
        printf '%s\n' "$pick_out" > "$pick_json"
    fi

    while IFS= read -r candidate_ckpt; do
        [[ -n "$candidate_ckpt" ]] && candidate_ckpts+=("$candidate_ckpt")
    done < <(print_candidate_ckpts "$pick_json")

    if [[ "${#candidate_ckpts[@]}" -eq 0 ]]; then
        echo "ERROR: no candidate checkpoints found from $pick_json" >&2
        exit 2
    fi
    for candidate_ckpt in "${candidate_ckpts[@]}"; do
        if [[ ! -f "$candidate_ckpt" ]]; then
            echo "ERROR: candidate checkpoint missing: $candidate_ckpt" >&2
            exit 2
        fi
        echo "[all4_pretrained_vision] light eval candidate for $suite: $candidate_ckpt"
        run_suite_eval light "$suite" "$ckpt_dir" "$candidate_ckpt"
    done

    select_json="${ckpt_dir}/ckpt_light_eval_selection.json"
    select_out=$(python select_light_eval_ckpt.py \
        --ckpt-dir "$ckpt_dir" \
        --candidates-json "$pick_json" \
        --suites "$suite" \
        --out "$select_json")
    echo "[all4_pretrained_vision] select_light_eval_ckpt output:"
    echo "$select_out"
    BEST_CKPT=$(printf '%s\n' "$select_out" | python -c "import sys, json; d=json.load(sys.stdin); print(d.get('top_1') or '')")
    if [[ -z "$BEST_CKPT" || ! -f "$BEST_CKPT" ]]; then
        echo "ERROR: light rollout selection returned no usable ckpt" >&2
        exit 2
    fi
    echo "[all4_pretrained_vision] best $suite ckpt by light rollout: $BEST_CKPT"
}

train_one_suite() {
    local suite="$1"
    local train_log
    local n_tasks

    TRAIN_HDF5_DIR="${PROCESSED_ROOT}/${suite}"
    CKPT_DIR="${CKPT_ROOT}/${ARM}_${suite}_seed${SEED}"
    export STABLEWM_HOME="$CKPT_DIR"

    [[ -d "$TRAIN_HDF5_DIR" ]] || { echo "ERROR: processed suite dir missing: $TRAIN_HDF5_DIR" >&2; exit 1; }
    n_tasks=$(find "$TRAIN_HDF5_DIR" -maxdepth 1 -name "*.h5" | wc -l | tr -d '[:space:]')
    if [[ "$n_tasks" -lt 10 ]]; then
        echo "ERROR: expected 10 .h5 in $TRAIN_HDF5_DIR, got $n_tasks" >&2
        exit 1
    fi

    mkdir -p "$CKPT_DIR"
    train_log="${CKPT_DIR}/train.log"

    echo "=========================================================="
    echo "[all4_pretrained_vision] train suite=$suite TRAIN_HDF5_DIR=$TRAIN_HDF5_DIR"
    echo "[all4_pretrained_vision] CKPT_DIR=$CKPT_DIR"
    echo "=========================================================="

    python train.py \
        data=libero \
        data.dataset.hdf5_dir="$TRAIN_HDF5_DIR" \
        vision_encoder.source=hf \
        vision_encoder.model_name_or_path="$VISION_ENCODER" \
        vision_encoder.freeze=true \
        vision_encoder.local_files_only=true \
        loss.pred_weight="$PRED" \
        loss.sigreg_weight="$SIGREG" \
        predictor.state_prediction_arch="$STATE_ARCH" \
        predictor.depth="$PREDICTOR_DEPTH" \
        projector.norm_type="$NORM" \
        scheduler.warmup_steps="$WARMUP_STEPS" \
        trainer.devices="$DDP_DEVICES" \
        trainer.strategy="$TRAINER_STRATEGY" \
        trainer.sync_batchnorm="$SYNC_BATCHNORM" \
        trainer.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
        trainer.use_distributed_sampler=false \
        +trainer.max_steps="$MAX_STEPS" \
        trainer.max_epochs=999 \
        +trainer.val_check_interval="$VAL_INTERVAL" \
        +trainer.check_val_every_n_epoch=null \
        loader.batch_size="$PER_DEVICE_BATCH_SIZE" \
        loader.global_batch_size="$GLOBAL_BATCH_SIZE" \
        train_split="$TRAIN_SPLIT" \
        seed="$SEED" \
        subdir="" \
        output_model_name=lewm \
        +ckpt_top_k="$CKPT_SELECT_TOP_K" \
        +visual_tokens.pool_grid="$POOL_GRID" \
        +visual_tokens.patch_sp="$PATCH_SP" \
        +visual_tokens.patch_sp_weight="$PATCH_SP_WEIGHT" \
        2>&1 | tee "$train_log"

    select_best_ckpt_for_suite "$suite" "$CKPT_DIR"
    run_suite_eval final "$suite" "$CKPT_DIR" "$BEST_CKPT"
    echo "[all4_pretrained_vision] suite done: $suite -> $CKPT_DIR"
}

for suite in "${RUN_SUITES[@]}"; do
    train_one_suite "$suite"
done

echo "[all4_pretrained_vision] ALL DONE - per-suite logs under $CKPT_ROOT/${ARM}_<suite>_seed${SEED}/"

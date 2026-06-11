#!/usr/bin/env bash
# preprocess_all4.sh — Regenerate/filter LIBERO then preprocess × 40.
#
# Reads from /nas_data_new/caz/data_ssd/libero/libero_{spatial,object,goal,10}/*.hdf5
# (STRICTLY READ-ONLY), writes OpenVLA-style no-noop filtered HDF5 to
# /Data/lyw/libero_filtered_224/libero_<suite>/<task>.hdf5, then writes
# per-task training .h5 outputs to /Data/lyw/libero_processed_v5/libero_<suite>/<task>.h5.
#
# Each invocation uses --load-tokenizer pointing at the unified
# /Data/lyw/fast_tokenizer_all4/ so all 40 outputs share BPE vocab. Per-task
# action_low/high are computed independently (Option B). Skips tasks whose
# output already exists, so the script is idempotent and resumable.
#
# Usage:
#   bash preprocess_all4.sh                       # first pass may only regenerate filtered data
#   python fit_tokenizer_all4.py --raw-root /Data/lyw/libero_filtered_224
#   bash preprocess_all4.sh                       # second pass preprocesses with tokenizer
#   PARALLEL=4 bash preprocess_all4.sh            # 4 invocations at a time
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/nas_data_new/caz/data_ssd/libero}"
FILTERED_ROOT="${FILTERED_ROOT:-/Data/lyw/libero_filtered_224}"
OUT_ROOT="${OUT_ROOT:-/Data/lyw/libero_processed_v5}"
TOKENIZER="${TOKENIZER:-/Data/lyw/fast_tokenizer_all4}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
STRIDE="${STRIDE:-1}"
MAX_TOKENS="${MAX_TOKENS:-80}"
CAMERA_SIZE="${CAMERA_SIZE:-224}"
WAIT_STEPS="${WAIT_STEPS:-10}"
NOOP_THRESHOLD="${NOOP_THRESHOLD:-1e-4}"
PARALLEL="${PARALLEL:-1}"

if [[ ! -d "$TOKENIZER" ]]; then
    echo "WARN: tokenizer dir not found: $TOKENIZER" >&2
    echo "      This pass will regenerate/filter HDF5 only. Then run:" >&2
    echo "      python fit_tokenizer_all4.py --raw-root $FILTERED_ROOT --tokenizer-out $TOKENIZER" >&2
    TOKENIZER_MISSING=1
else
    TOKENIZER_MISSING=0
fi
if [[ ! -d "$RAW_ROOT" ]]; then
    echo "ERROR: raw root not found: $RAW_ROOT" >&2
    exit 1
fi

# xargs spawns child shells via `bash -c` — they do NOT inherit the parent's
# local vars. Export so the per-task run_one() can see them after the fanout.
export CHUNK_SIZE STRIDE MAX_TOKENS TOKENIZER CAMERA_SIZE WAIT_STEPS NOOP_THRESHOLD TOKENIZER_MISSING

# Refuse to write under the read-only raw data root.
case "$OUT_ROOT" in
    /nas_data_new/caz/data_ssd/libero*|/nas_data_new/caz/data_ssd/libero)
        echo "ERROR: OUT_ROOT=$OUT_ROOT is inside the READ-ONLY raw LIBERO path. Aborting." >&2
        exit 1
        ;;
esac

mkdir -p "$OUT_ROOT"
mkdir -p "$FILTERED_ROOT"

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

processed_matches_filtered() {
    local out="$1" filtered="$2"
    python -c '
import h5py
import pathlib
import sys

try:
    with h5py.File(sys.argv[1], "r") as f:
        source_file = str(f.attrs.get("source_file", ""))
    if not source_file:
        sys.exit(1)
    source_path = pathlib.Path(source_file).resolve()
    filtered_path = pathlib.Path(sys.argv[2]).resolve()
    sys.exit(0 if source_path == filtered_path else 1)
except Exception:
    sys.exit(1)
' "$out" "$filtered"
}

run_one() {
    local raw="$1" suite="$2" filtered_suite_dir="$3" out_suite_dir="$4"
    local base
    base=$(basename "$raw" .hdf5)
    base="${base%_demo}"
    local filtered="${filtered_suite_dir}/${base}.hdf5"
    local out="${out_suite_dir}/${base}.h5"
    if [[ ! -f "$filtered" ]]; then
        echo "[regenerate] $raw -> $filtered"
        python regenerate_libero_filtered.py \
            --input "$raw" \
            --output "$filtered" \
            --suite "$suite" \
            --camera-size "$CAMERA_SIZE" \
            --wait-steps "$WAIT_STEPS" \
            --noop-threshold "$NOOP_THRESHOLD"
    else
        echo "[skip] $filtered exists"
    fi
    if [[ "$TOKENIZER_MISSING" == "1" ]]; then
        echo "[defer] tokenizer missing; filtered data ready at $filtered"
        return 0
    fi
    if [[ -f "$out" ]]; then
        if processed_matches_filtered "$out" "$filtered"; then
            echo "[skip] $out exists and matches $filtered"
            return 0
        fi
        echo "[stale] $out exists but does not match $filtered; reprocessing"
    fi
    echo "[preprocess] $filtered -> $out"
    python preprocess_libero.py \
        --input "$filtered" \
        --output "$out" \
        --chunk-size "$CHUNK_SIZE" \
        --stride "$STRIDE" \
        --image-key agentview_rgb \
        --hand-image-key eye_in_hand_rgb \
        --max-action-tokens "$MAX_TOKENS" \
        --load-tokenizer "$TOKENIZER"
}

# Build the full task list once, then optionally parallelize.
declare -a JOBS=()
for suite in "${SUITES[@]}"; do
    suite_dir="${RAW_ROOT}/${suite}"
    filtered_suite_dir="${FILTERED_ROOT}/${suite}"
    out_suite_dir="${OUT_ROOT}/${suite}"
    mkdir -p "$filtered_suite_dir"
    mkdir -p "$out_suite_dir"
    if [[ ! -d "$suite_dir" ]]; then
        echo "WARN: suite dir missing: $suite_dir" >&2
        continue
    fi
    for raw in "$suite_dir"/*.hdf5; do
        [[ -e "$raw" ]] || continue
        JOBS+=("$raw|$suite|$filtered_suite_dir|$out_suite_dir")
    done
done

echo "[preprocess_all4] ${#JOBS[@]} tasks to process (PARALLEL=$PARALLEL)"

if [[ "$PARALLEL" -le 1 ]]; then
    for entry in "${JOBS[@]}"; do
        IFS='|' read -r raw suite filtered_suite_dir out_suite_dir <<< "$entry"
        run_one "$raw" "$suite" "$filtered_suite_dir" "$out_suite_dir"
    done
else
    # Simple N-way fan-out via xargs.
    printf '%s\n' "${JOBS[@]}" | xargs -P "$PARALLEL" -I {} bash -c '
        entry="$1"
        IFS="|" read -r raw suite filtered_suite_dir out_suite_dir <<< "$entry"
        '"$(declare -f processed_matches_filtered)"'
        '"$(declare -f run_one)"'
        run_one "$raw" "$suite" "$filtered_suite_dir" "$out_suite_dir"
    ' _ {}
fi

if [[ "$TOKENIZER_MISSING" == "1" ]]; then
    echo "[preprocess_all4] filtered regeneration complete; tokenizer missing, preprocessing deferred" >&2
    exit 2
fi

echo "[preprocess_all4] all done"

#!/usr/bin/env bash
# preprocess_all4.sh — Drive preprocess_libero.py × 40 with the shared FAST tokenizer.
#
# Reads from /data/lyw/libero_{spatial,object,goal,10}/*.hdf5
# (STRICTLY READ-ONLY) and writes per-task .h5 outputs to
# /data/lyw/libero_processed_v5/libero_<suite>/<task>.h5.
#
# Each invocation uses --load-tokenizer pointing at the unified
# /data/lyw/fast_tokenizer_all4/ so all 40 outputs share BPE vocab. Per-task
# action_low/high are computed independently (Option B). Skips tasks whose
# output already exists, so the script is idempotent and resumable.
#
# Usage (on the GPU server, after fit_tokenizer_all4.py has produced the
# tokenizer):
#   bash preprocess_all4.sh                       # serial
#   PARALLEL=4 bash preprocess_all4.sh            # 4 invocations at a time
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/data/lyw}"
OUT_ROOT="${OUT_ROOT:-/data/lyw/libero_processed_v5}"
TOKENIZER="${TOKENIZER:-/data/lyw/fast_tokenizer_all4}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
STRIDE="${STRIDE:-1}"
MAX_TOKENS="${MAX_TOKENS:-80}"
PARALLEL="${PARALLEL:-1}"

# xargs spawns child shells via `bash -c` — they do NOT inherit the parent's
# local vars. Export so the per-task run_one() can see them after the fanout.
export CHUNK_SIZE STRIDE MAX_TOKENS TOKENIZER

if [[ ! -d "$TOKENIZER" ]]; then
    echo "ERROR: tokenizer dir not found: $TOKENIZER" >&2
    echo "Run fit_tokenizer_all4.py first." >&2
    exit 1
fi
if [[ ! -d "$RAW_ROOT" ]]; then
    echo "ERROR: raw root not found: $RAW_ROOT" >&2
    exit 1
fi

# Raw suites live at $RAW_ROOT/libero_<suite>; OUT_ROOT must differ from
# RAW_ROOT, else the per-task outputs would be written into (and pollute) the
# read-only raw suite dirs. (Raw + processed now coexist under /data/lyw.)
if [[ "$OUT_ROOT" == "$RAW_ROOT" ]]; then
    echo "ERROR: OUT_ROOT must differ from RAW_ROOT ($RAW_ROOT) — output would land inside the raw suite dirs. Use e.g. OUT_ROOT=\$RAW_ROOT/libero_processed_v5." >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

run_one() {
    local raw="$1" out_suite_dir="$2"
    local base
    base=$(basename "$raw" .hdf5)
    base="${base%_demo}"
    local out="${out_suite_dir}/${base}.h5"
    if [[ -f "$out" ]]; then
        echo "[skip] $out exists"
        return 0
    fi
    echo "[run] $raw -> $out"
    python preprocess_libero.py \
        --input "$raw" \
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
    out_suite_dir="${OUT_ROOT}/${suite}"
    mkdir -p "$out_suite_dir"
    if [[ ! -d "$suite_dir" ]]; then
        echo "WARN: suite dir missing: $suite_dir" >&2
        continue
    fi
    for raw in "$suite_dir"/*.hdf5; do
        [[ -e "$raw" ]] || continue
        JOBS+=("$raw|$out_suite_dir")
    done
done

echo "[preprocess_all4] ${#JOBS[@]} tasks to process (PARALLEL=$PARALLEL)"

if [[ "$PARALLEL" -le 1 ]]; then
    for entry in "${JOBS[@]}"; do
        raw="${entry%%|*}"; out_suite_dir="${entry##*|}"
        run_one "$raw" "$out_suite_dir"
    done
else
    # Simple N-way fan-out via xargs.
    printf '%s\n' "${JOBS[@]}" | xargs -P "$PARALLEL" -I {} bash -c '
        entry="$1"
        raw="${entry%%|*}"
        out_suite_dir="${entry##*|}"
        '"$(declare -f run_one)"'
        run_one "$raw" "$out_suite_dir"
    ' _ {}
fi

echo "[preprocess_all4] all done"

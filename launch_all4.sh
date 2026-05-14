#!/usr/bin/env bash
# launch_all4.sh — thin wrapper that launches run_all4.sh inside a tmux session.
#
# Single arm (sp_sigreg) by default — no fanout needed. Phase B (multi-arm
# follow-up) would loop over ARMS="baseline sigreg_only sp_only_ln" across
# multiple GPUs; that block is commented out below.
#
# Usage:
#   bash launch_all4.sh           # default GPU 0
#   GPU=3 bash launch_all4.sh
#   SEED=2024 GPU=1 bash launch_all4.sh
set -euo pipefail

ARM="${ARM:-sp_sigreg}"
SEED="${SEED:-3072}"
GPU="${GPU:-0}"
SESSION="${SESSION:-all4}"

# Pre-flight: confirm GPU is mostly free (helpful when sharing a server).
echo "[launch_all4] GPU $GPU status:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null | \
    awk -v g="$GPU" '$1 == g {printf "  GPU %s used=%sMiB total=%sMiB util=%s%%\n", $1, $2, $3, $4}' \
    || true

# Kill stale session of the same name.
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Forward all the relevant env vars to the tmux child shell.
PROBE_ENABLED="${PROBE_ENABLED:-true}"
tmux new-session -d -s "$SESSION" \
    "ARM=$ARM SEED=$SEED CUDA_VISIBLE_DEVICES=$GPU PROBE_ENABLED=$PROBE_ENABLED \
     bash /Data/lyw/le-wm/run_all4.sh 2>&1 | \
     tee /Data/lyw/le-wm/all4_${ARM}_seed${SEED}_gpu${GPU}.log; \
     echo \"=== EXIT \$? ===\"; bash"

echo "[launch_all4] launched tmux session '$SESSION' (ARM=$ARM SEED=$SEED GPU=$GPU)"
echo "Attach:   tmux attach -t $SESSION"
echo "Tail log: tail -f /Data/lyw/le-wm/all4_${ARM}_seed${SEED}_gpu${GPU}.log"

# ====================================================================
# Phase B fanout (commented out — for multi-arm follow-up)
# ====================================================================
# ARMS="baseline sigreg_only sp_only_ln sp_sigreg"
# GPUS="0 1"
# idx=0
# read -ra GPU_ARR <<< "$GPUS"
# for a in $ARMS; do
#     gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
#     win="arm_${a}_gpu${gpu}"
#     tmux new-window -t "$SESSION" -n "$win" \
#         "ARM=$a SEED=$SEED CUDA_VISIBLE_DEVICES=$gpu bash /Data/lyw/le-wm/run_all4.sh; bash"
#     idx=$((idx + 1))
# done

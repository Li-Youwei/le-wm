#!/usr/bin/env bash
# launch_visual17.sh — tmux wrapper around run_visual17.sh.
#
# Usage:
#   bash launch_visual17.sh                    # GPU 3, seed 3072
#   GPU=2 SEED=2024 bash launch_visual17.sh
set -euo pipefail

ARM="${ARM:-visual17_baseline}"
SEED="${SEED:-3072}"
GPU="${GPU:-3}"
SESSION="${SESSION:-v17}"

echo "[launch_visual17] GPU $GPU status:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null | \
    awk -v g="$GPU" '$1 == g {printf "  GPU %s used=%sMiB total=%sMiB util=%s%%\n", $1, $2, $3, $4}' \
    || true

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" \
    "ARM=$ARM SEED=$SEED CUDA_VISIBLE_DEVICES=$GPU \
     bash /Data/lyw/le-wm/run_visual17.sh 2>&1 | \
     tee /Data/lyw/le-wm/${ARM}_seed${SEED}_gpu${GPU}.log; \
     echo \"=== EXIT \$? ===\"; bash"

echo "[launch_visual17] launched tmux '$SESSION' (ARM=$ARM SEED=$SEED GPU=$GPU)"
echo "Attach:   tmux attach -t $SESSION"
echo "Tail log: tail -f /Data/lyw/le-wm/${ARM}_seed${SEED}_gpu${GPU}.log"

#!/usr/bin/env bash
# =============================================================================
# launch_eval_videos.sh — fan out 9 eval-only re-runs (with --save-videos)
# across the GPUs you have free. Mirrors launch_ablations.sh structure.
#
# Usage:
#   bash launch_eval_videos.sh                      # GPUs="1 3"
#   GPUS="0 1 2 3" bash launch_eval_videos.sh
#   SEEDS="2024 1234 3072" ARMS="bn_only sigreg_only sp_only_ln" bash launch_eval_videos.sh
#
# Watch:
#   tmux attach -t evalv
#   tail -f /Data/lyw/abl_eval_gpu<N>.tmuxlog
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/Data/lyw/le-wm}"
TMUX_SESS="${TMUX_SESS:-evalv}"
GPUS="${GPUS:-1 3}"
SEEDS="${SEEDS:-2024 1234 3072}"
ARMS="${ARMS:-bn_only sp_only_ln sigreg_only}"

# Build (arm:seed) work list.
WORK=()
for arm in $ARMS; do
    for seed in $SEEDS; do
        WORK+=("${arm}:${seed}")
    done
done

# Round-robin onto GPUs.
declare -A QUEUE_BY_GPU
for gpu in $GPUS; do QUEUE_BY_GPU[$gpu]=""; done
i=0
for item in "${WORK[@]}"; do
    gpu_arr=($GPUS)
    gpu="${gpu_arr[$((i % ${#gpu_arr[@]}))]}"
    QUEUE_BY_GPU[$gpu]+=" $item"
    i=$((i + 1))
done

echo "==================================================================="
echo "Eval-only (with --save-videos) launch plan"
echo "==================================================================="
echo "tmux session : $TMUX_SESS"
echo "repo on srv  : $REPO_DIR"
echo "arms         : $ARMS"
echo "seeds        : $SEEDS"
echo "gpus         : $GPUS"
echo "-------------------------------------------------------------------"
for gpu in $GPUS; do
    n=$(echo ${QUEUE_BY_GPU[$gpu]} | wc -w)
    echo "  GPU $gpu  ($n runs):"
    for item in ${QUEUE_BY_GPU[$gpu]}; do
        arm="${item%%:*}"; seed="${item##*:}"
        printf "    - %-12s seed=%s\n" "$arm" "$seed"
    done
done
echo "==================================================================="

tmux kill-session -t "$TMUX_SESS" 2>/dev/null || true

first_window=true
for gpu in $GPUS; do
    win="gpu${gpu}"
    queue="${QUEUE_BY_GPU[$gpu]}"
    log_file="/Data/lyw/abl_eval_gpu${gpu}.tmuxlog"

    # Same conda + offline-HF prelude as launch_ablations.sh.
    cmd="exec > >(tee -a $log_file) 2>&1; "
    cmd+="source $HOME/miniconda3/etc/profile.d/conda.sh && "
    cmd+="conda activate vla && cd $REPO_DIR && "
    cmd+="export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && "
    cmd+="echo '==== GPU $gpu eval queue start: '\$(date)' ===='; "
    for item in $queue; do
        arm="${item%%:*}"; seed="${item##*:}"
        cmd+="echo '---- start  arm=$arm  seed=$seed  GPU=$gpu  '\$(date)' ----'; "
        cmd+="ARM=$arm SEED=$seed CUDA_VISIBLE_DEVICES=$gpu bash eval_videos.sh; "
        cmd+="echo '---- finish arm=$arm  seed=$seed  GPU=$gpu  '\$(date)' ----'; "
    done
    cmd+="echo '==== GPU $gpu eval queue done: '\$(date)' ===='; bash"

    if $first_window; then
        tmux new-session -d -s "$TMUX_SESS" -n "$win" "bash -c \"$cmd\""
        first_window=false
    else
        tmux new-window -t "$TMUX_SESS" -n "$win" "bash -c \"$cmd\""
    fi
done

echo ""
echo "tmux session '$TMUX_SESS' running. Attach: tmux attach -t $TMUX_SESS"
echo "Per-GPU log: tail -f /Data/lyw/abl_eval_gpu<N>.tmuxlog"

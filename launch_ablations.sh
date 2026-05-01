#!/usr/bin/env bash
# =============================================================================
# launch_ablations.sh — fan out 9 ablation runs across the GPUs you have free.
#
# Schedules 3 arms × 3 seeds = 9 sequential runs, split across the GPUs
# listed in $GPUS (default "1 3"). Each GPU gets its own tmux window inside
# session $TMUX_SESS (default "abl") and processes its share serially.
#
# Usage:
#   bash launch_ablations.sh                            # GPUs = "1 3"
#   GPUS="1 3" bash launch_ablations.sh
#   GPUS="0 1 2 3" bash launch_ablations.sh             # 4-way fanout
#   SEEDS="42 1234 3072" bash launch_ablations.sh
#
# Watch:
#   tmux attach -t abl                                  # full session
#   tmux a -t abl \; select-window -t :gpu1             # one window
#
# Each window logs progress to /Data/lyw/abl_gpu<N>.tmuxlog.
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/Data/lyw/le-wm}"   # where run_ablation.sh lives on the server
TMUX_SESS="${TMUX_SESS:-abl}"
GPUS="${GPUS:-1 3}"
SEEDS="${SEEDS:-2024 1234 3072}"
ARMS="${ARMS:-bn_only sp_only_ln sigreg_only}"

# Build the 9-element work list (arm seed) pairs in a deterministic order.
WORK=()
for arm in $ARMS; do
    for seed in $SEEDS; do
        WORK+=("${arm}:${seed}")
    done
done

# Round-robin distribute across the GPUs in $GPUS.
declare -A QUEUE_BY_GPU
for gpu in $GPUS; do QUEUE_BY_GPU[$gpu]=""; done

i=0
for item in "${WORK[@]}"; do
    gpu_arr=($GPUS)
    gpu="${gpu_arr[$((i % ${#gpu_arr[@]}))]}"
    QUEUE_BY_GPU[$gpu]+=" $item"
    i=$((i + 1))
done

# Show plan first.
echo "==================================================================="
echo "Ablation launch plan"
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

# Kill any stale session with the same name.
tmux kill-session -t "$TMUX_SESS" 2>/dev/null || true

first_window=true
for gpu in $GPUS; do
    win="gpu${gpu}"
    queue="${QUEUE_BY_GPU[$gpu]}"
    log_file="/Data/lyw/abl_gpu${gpu}.tmuxlog"

    # Build a single shell command that walks the queue serially on this GPU.
    # `conda` is NOT available in non-interactive shells on this server, so we
    # source conda.sh explicitly before `conda activate`. Trailing `; bash`
    # keeps the tmux window open after the queue completes (so logs are
    # readable on attach instead of the window vanishing).
    # HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 — the GPU server cannot reach
    # huggingface.co, so without these flags `T5Tokenizer.from_pretrained(...)`
    # spends 5-15 minutes per file timing out HEAD probes before it falls back
    # to the cache at ~/.cache/huggingface/hub. Both vars are needed because
    # transformers <4.35 reads TRANSFORMERS_OFFLINE while newer reads HF_HUB_OFFLINE.
    cmd="exec > >(tee -a $log_file) 2>&1; "
    cmd+="source $HOME/miniconda3/etc/profile.d/conda.sh && "
    cmd+="conda activate vla && cd $REPO_DIR && "
    cmd+="export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && "
    cmd+="echo '==== GPU $gpu queue start: '\$(date)' ===='; "
    for item in $queue; do
        arm="${item%%:*}"; seed="${item##*:}"
        cmd+="echo '---- start  arm=$arm  seed=$seed  GPU=$gpu  '\$(date)' ----'; "
        cmd+="ARM=$arm SEED=$seed CUDA_VISIBLE_DEVICES=$gpu bash run_ablation.sh; "
        cmd+="echo '---- finish arm=$arm  seed=$seed  GPU=$gpu  '\$(date)' ----'; "
    done
    cmd+="echo '==== GPU $gpu queue done: '\$(date)' ===='; bash"

    if $first_window; then
        tmux new-session -d -s "$TMUX_SESS" -n "$win" "bash -c \"$cmd\""
        first_window=false
    else
        tmux new-window -t "$TMUX_SESS" -n "$win" "bash -c \"$cmd\""
    fi
done

echo ""
echo "tmux session '$TMUX_SESS' is running on the server."
echo "Attach:    tmux attach -t $TMUX_SESS"
echo "Per-GPU:   tail -f /Data/lyw/abl_gpu<N>.tmuxlog"

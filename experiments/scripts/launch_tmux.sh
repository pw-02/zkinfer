#!/usr/bin/env bash
# Local experiment launcher. All research settings live in experiments/config/config.yaml.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
SESSION="${TMUX_SESSION:-zkexp}"
CONDA_ENV="${CONDA_ENV:-zk}"
if (( $# > 1 )); then
    echo "Usage: $0 [workload]; edit experiments/config/config.yaml for other settings."
    exit 2
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already exists. Attach to it or close it explicitly first."
    exit 1
fi
ARGS=()
if (( $# == 1 )); then
    ARGS+=(--workload "$1")
fi
printf -v COMMAND '%q ' python -m experiments.run_local_exp "${ARGS[@]}"
printf -v ROOT_Q '%q' "$ROOT_DIR"
printf -v ENV_Q '%q' "$CONDA_ENV"
# Keep the completed window available for inspection, including failure output.
tmux new-session -d -s "$SESSION" -n experiment "bash"
tmux set-option -t "$SESSION" remain-on-exit on
tmux send-keys -t "$SESSION:experiment" "cd $ROOT_Q && source \$(conda info --base)/etc/profile.d/conda.sh && conda activate $ENV_Q && $COMMAND" C-m
tmux attach -t "$SESSION"

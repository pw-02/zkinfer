#!/usr/bin/env bash

set -euo pipefail

NUM_WORKERS="${1:-1}"
WORKLOAD="${2:-mnist_gan}"
COORDINATOR_HOST="${3:-127.0.0.1}"
COORDINATOR_PORT="${4:-50051}"

SESSION="zkexp"
CONDA_ENV="${CONDA_ENV:-zk}"
ROOT_DIR="$(pwd)"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_${WORKLOAD}"
RUN_DIR="${ROOT_DIR}/experiments/runs/${RUN_ID}"
LOGS_DIR="${RUN_DIR}/logs"
SHARED_DIR="${RUN_DIR}/shared"
CACHE_DIR="${RUN_DIR}/cache"

mkdir -p "$LOGS_DIR" "$SHARED_DIR" "$CACHE_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
fi

BASE_CMD="\
cd ${ROOT_DIR} && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate ${CONDA_ENV} && \
export PYTHONPATH=${ROOT_DIR}:\${PYTHONPATH:-}"

RUNTIME_CFG="\
coordinator.host=${COORDINATOR_HOST} \
coordinator.port=${COORDINATOR_PORT} \
coordinator.runs_dir=${RUN_DIR} \
coordinator.logs_dir=${LOGS_DIR} \
worker.runs_dir=${RUN_DIR} \
worker.logs_dir=${LOGS_DIR} \
storage.backend=filesystem \
storage.transfer_prefix=${SHARED_DIR} \
storage.proving_cache_prefix=${CACHE_DIR}"

tmux new-session -d -s "$SESSION" -n "coordinator"

tmux send-keys -t "$SESSION:coordinator" \
    "${BASE_CMD} && \
    python -m zkinfer.runtime.coordinator_grpc ${RUNTIME_CFG} \
    2>&1 | tee ${LOGS_DIR}/coordinator.tmux.log" \
    C-m

echo "Waiting for coordinator at ${COORDINATOR_HOST}:${COORDINATOR_PORT}..."

python - "$COORDINATOR_HOST" "$COORDINATOR_PORT" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.time() + 60

while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1):
            print(f"Coordinator ready at {host}:{port}")
            raise SystemExit(0)
    except OSError:
        time.sleep(0.5)

raise SystemExit(f"Coordinator not ready at {host}:{port}")
PY

for i in $(seq 1 "$NUM_WORKERS"); do
    tmux new-window -t "$SESSION" -n "worker${i}"

    tmux send-keys -t "$SESSION:worker${i}" \
        "${BASE_CMD} && \
        python -m zkinfer.runtime.worker \
        ${RUNTIME_CFG} \
        worker.worker_id=worker_${i} \
        2>&1 | tee ${LOGS_DIR}/worker_${i}.tmux.log" \
        C-m
done

tmux new-window -t "$SESSION" -n "submit"

tmux send-keys -t "$SESSION:submit" \
    "${BASE_CMD} && \
    echo 'Run directory: ${RUN_DIR}' && \
    python experiments/submit_job.py \
    +workload=${WORKLOAD} \
    launch.coordinator_host=${COORDINATOR_HOST} \
    launch.coordinator_port=${COORDINATOR_PORT} \
    2>&1 | tee ${LOGS_DIR}/submit.tmux.log" \
    C-m

echo "Started tmux session: ${SESSION}"
echo "Run directory: ${RUN_DIR}"
echo "Logs: ${LOGS_DIR}"

tmux attach -t "$SESSION"
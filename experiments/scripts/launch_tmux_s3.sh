#!/usr/bin/env bash

set -euo pipefail

NUM_WORKERS="${1:-1}"
WORKLOAD="${2:-mnist_gan}"
COORDINATOR_HOST="${3:-127.0.0.1}"
COORDINATOR_PORT="${4:-50051}"
OPS_PER_CHUNK="${5:-1}"

SESSION="${TMUX_SESSION:-zkexp}"
CONDA_ENV="${CONDA_ENV:-zk}"

S3_BUCKET="${ZKINFER_S3_BUCKET:-zkinfer}"
S3_PREFIX="${ZKINFER_S3_PREFIX:-zkinfer-reviewer}"

if [[ -z "$S3_BUCKET" ]]; then
    echo "Error: ZKINFER_S3_BUCKET is not set."
    echo "Example:"
    echo "  export ZKINFER_S3_BUCKET=zkinfer"
    exit 1
fi

if [[ ! "$OPS_PER_CHUNK" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: ops_per_chunk must be a positive integer."
    exit 1
fi

ROOT_DIR="$(pwd)"

if [[ ! -f "${ROOT_DIR}/experiments/submit_job.py" ]]; then
    echo "Error: run this script from the zkinfer repository root."
    exit 1
fi

RUN_ID="$(date -u +%Y-%m-%d_%H-%M-%S)_${WORKLOAD}_g${OPS_PER_CHUNK}"

CAMPAIGN_DIR="${ROOT_DIR}/experiments/runs/${RUN_ID}"
REQUESTS_DIR="${CAMPAIGN_DIR}/requests"
LOGS_DIR="${CAMPAIGN_DIR}/logs"

mkdir -p "$REQUESTS_DIR" "$LOGS_DIR"

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
coordinator.runs_dir=${REQUESTS_DIR} \
coordinator.logs_dir=${LOGS_DIR} \
worker.runs_dir=${REQUESTS_DIR} \
worker.logs_dir=${LOGS_DIR} \
storage.backend=s3 \
storage.s3_bucket=${S3_BUCKET} \
storage.s3_prefix=${S3_PREFIX} \
storage.transfer_prefix=transfer \
storage.proving_cache_enabled=true \
storage.proving_cache_prefix=cache \
storage.proving_cache_overwrite=false"

echo "Checking access to S3 bucket: ${S3_BUCKET}"

if command -v aws >/dev/null 2>&1; then
    if ! aws s3api head-bucket --bucket "$S3_BUCKET"; then
        echo "Error: cannot access S3 bucket ${S3_BUCKET}."
        echo "Check your AWS credentials, region, and bucket permissions."
        exit 1
    fi
else
    echo "Warning: AWS CLI not found; skipping S3 access check."
fi

tmux new-session -d -s "$SESSION" -n "coordinator"

tmux send-keys -t "$SESSION:coordinator" \
    "${BASE_CMD} && \
    python -m zkinfer.runtime.coordinator_grpc \
    ${RUNTIME_CFG} \
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

sleep 2

tmux new-window -t "$SESSION" -n "submit"

tmux send-keys -t "$SESSION:submit" \
    "${BASE_CMD} && \
    echo 'Campaign directory: ${CAMPAIGN_DIR}' && \
    echo 'Request results: ${REQUESTS_DIR}' && \
    echo 'Workload: ${WORKLOAD}' && \
    echo 'ops_per_chunk: ${OPS_PER_CHUNK}' && \
    echo 'S3 bucket: ${S3_BUCKET}' && \
    echo 'S3 prefix: ${S3_PREFIX}' && \
    python experiments/submit_job.py \
    +workload=${WORKLOAD} \
    execution.split_mode=fixed \
    execution.ops_per_chunk=${OPS_PER_CHUNK} \
    launch.coordinator_host=${COORDINATOR_HOST} \
    launch.coordinator_port=${COORDINATOR_PORT} \
    2>&1 | tee -a ${LOGS_DIR}/submit.tmux.log" \
    C-m

echo
echo "Started tmux session: ${SESSION}"
echo "Campaign directory: ${CAMPAIGN_DIR}"
echo "Request results: ${REQUESTS_DIR}"
echo "Logs: ${LOGS_DIR}"
echo "Workload: ${WORKLOAD}"
echo "Workers: ${NUM_WORKERS}"
echo "ops_per_chunk: ${OPS_PER_CHUNK}"
echo "S3 location: s3://${S3_BUCKET}/${S3_PREFIX}"
echo

tmux attach -t "$SESSION"
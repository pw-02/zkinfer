#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# The experiment YAML is the single user-facing configuration for this launcher.
# Positional arguments and existing environment overrides remain supported.
LAUNCH_SETTINGS="$(python experiments/scripts/launcher_config.py)"
eval "$LAUNCH_SETTINGS"
WORKLOAD="${1:-$CFG_WORKLOAD}"
COORDINATOR_HOST="${2:-$CFG_HOST}"
COORDINATOR_PORT="${3:-$CFG_PORT}"
STAGE_ADMISSION="${4:-$CFG_ADMISSION}"
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-$CFG_MAX_JOBS}"
KEEP_PROOF_ARTIFACTS="${KEEP_PROOF_ARTIFACTS:-$CFG_KEEP_PROOFS}"
ZKINFER_PROFILING_MODE="${ZKINFER_PROFILING_MODE:-$CFG_PROFILING_MODE}"

if [[ "$STAGE_ADMISSION" != "true" && "$STAGE_ADMISSION" != "false" ]]; then
    echo "Error: stage admission must be true or false."
    exit 1
fi
if [[ "$KEEP_PROOF_ARTIFACTS" != "true" && "$KEEP_PROOF_ARTIFACTS" != "false" ]]; then
    echo "Error: KEEP_PROOF_ARTIFACTS must be true or false."
    exit 1
fi
if [[ "$MAX_CONCURRENT_JOBS" != "null" && ! "$MAX_CONCURRENT_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: MAX_CONCURRENT_JOBS must be a positive integer or 'null'."
    exit 1
fi

SESSION="${TMUX_SESSION:-zkexp}"
CONDA_ENV="${CONDA_ENV:-zk}"
ADMISSION_TAG="admission-${STAGE_ADMISSION}"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_${WORKLOAD}_${ADMISSION_TAG}"
RUN_DIR="${CFG_RUNS_DIR}/${RUN_ID}"
REQUESTS_DIR="${RUN_DIR}/requests"
LOGS_DIR="${RUN_DIR}/logs"
CONFIG_DIR="${RUN_DIR}/config"
STATE_DB="${RUN_DIR}/coordinator_state.sqlite3"

REQUEST_DATA_DIR="$CFG_REQUEST_DATA"
REUSABLE_SETUP_DIR="$CFG_SETUP"
LOCAL_CACHE_DIR="$CFG_CACHE"
NODE_RUNTIME_DIR="${ROOT_DIR}/storage/node/runtime"

mkdir -p \
    "$REQUESTS_DIR" \
    "$LOGS_DIR" \
    "$CONFIG_DIR" \
    "$REQUEST_DATA_DIR" \
    "$REUSABLE_SETUP_DIR" \
    "$LOCAL_CACHE_DIR" \
    "$NODE_RUNTIME_DIR"

cat > "${CONFIG_DIR}/launch.txt" <<EOF
workload=${WORKLOAD}
coordinator_host=${COORDINATOR_HOST}
coordinator_port=${COORDINATOR_PORT}
stage_admission=${STAGE_ADMISSION}
max_concurrent_jobs=${MAX_CONCURRENT_JOBS}
keep_proof_artifacts=${KEEP_PROOF_ARTIFACTS}
profiling_mode=${ZKINFER_PROFILING_MODE}
sampling_interval_sec=${CFG_SAMPLING}
scheduling_policy=${CFG_POLICY}
state_db=${STATE_DB}
request_data=${REQUEST_DATA_DIR}
reusable_setup=${REUSABLE_SETUP_DIR}
local_cache=${LOCAL_CACHE_DIR}
node_runtime_dir=${NODE_RUNTIME_DIR}
EOF

if [[ "$MAX_CONCURRENT_JOBS" == "null" && "$STAGE_ADMISSION" == "false" ]]; then
    echo "Warning: unlimited concurrent jobs with stage admission disabled can start many provers at once."
    echo "Use this only for an intentional concurrency characterization run."
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
fi

BASE_CMD="\
cd ${ROOT_DIR} && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate ${CONDA_ENV} && \
export PYTHONPATH=${ROOT_DIR}:\${PYTHONPATH:-} && \
export ZKINFER_KEEP_PROOF_ARTIFACTS=${KEEP_PROOF_ARTIFACTS} && \
export ZKINFER_PROFILING_MODE=${ZKINFER_PROFILING_MODE}"

RUNTIME_CFG="\
coordinator.host=${COORDINATOR_HOST} \
coordinator.scheduling_policy=${CFG_POLICY} \
coordinator.port=${COORDINATOR_PORT} \
coordinator.runs_dir=${REQUESTS_DIR} \
coordinator.logs_dir=${LOGS_DIR} \
coordinator.state_db_path=${STATE_DB} \
node.runs_dir=${REQUESTS_DIR} \
node.logs_dir=${LOGS_DIR} \
node.local_cache.path=${LOCAL_CACHE_DIR} \
node.local_cache.max_gb=${CFG_CACHE_GB} \
node.resource_sampling_interval_sec=${CFG_SAMPLING} \
node.runtime_state_dir=${NODE_RUNTIME_DIR} \
node.max_concurrent_jobs=${MAX_CONCURRENT_JOBS} \
node.stage_admission.enabled=${STAGE_ADMISSION} \
node.stage_admission.mode=${CFG_ADMISSION_MODE} \
node.stage_admission.memory_safety_reserve_gb=${CFG_RESERVE} \
node.stage_admission.history_percentile=${CFG_PERCENTILE} \
storage.backend=filesystem \
storage.request_data.path=${REQUEST_DATA_DIR} \
storage.reusable_setup.path=${REUSABLE_SETUP_DIR} \
 storage.reusable_setup.enabled=${CFG_REUSE} \
 storage.resource_profiles.enabled=${CFG_PROFILES} \
 storage.resource_profiles.path=${CFG_PROFILE_PATH}"

tmux new-session -d -s "$SESSION" -n "coordinator"
tmux send-keys -t "$SESSION:coordinator" \
    "${BASE_CMD} && python -m zkinfer.runtime.coordinator_grpc ${RUNTIME_CFG} \
    2>&1 | tee ${LOGS_DIR}/coordinator.tmux.log" C-m

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
            raise SystemExit(0)
    except OSError:
        time.sleep(0.5)
raise SystemExit(f"Coordinator not ready at {host}:{port}")
PY

tmux new-window -t "$SESSION" -n "node"
tmux send-keys -t "$SESSION:node" \
    "${BASE_CMD} && python -m zkinfer.runtime.node.supervisor \
    ${RUNTIME_CFG} node.node_id=local-node \
    2>&1 | tee ${LOGS_DIR}/node.tmux.log" C-m

sleep 2

tmux new-window -t "$SESSION" -n "submit"
tmux send-keys -t "$SESSION:submit" \
    "${BASE_CMD} && \
    echo 'Run directory: ${RUN_DIR}' && \
    echo 'Stage admission: ${STAGE_ADMISSION}' && \
    echo 'Max concurrent jobs: ${MAX_CONCURRENT_JOBS}' && \
    echo 'Keep proof artifacts: ${KEEP_PROOF_ARTIFACTS}' && \
    python experiments/submit_job.py \
    +workload=${WORKLOAD} \
    launch.coordinator_host=${COORDINATOR_HOST} \
    launch.coordinator_port=${COORDINATOR_PORT} \
    2>&1 | tee ${LOGS_DIR}/submit.tmux.log; \
    if [ \${PIPESTATUS[0]} -eq 0 ]; then python -m zkinfer.profiling.collect_reports ${RUN_DIR}; fi" C-m

echo "Started tmux session: ${SESSION}"
echo "Run directory: ${RUN_DIR}"
echo "Requests: ${REQUESTS_DIR}"
echo "Persistent storage: ${ROOT_DIR}/storage"
echo "Stage admission: ${STAGE_ADMISSION}"
echo "Max concurrent jobs: ${MAX_CONCURRENT_JOBS}"
echo "Keep proof artifacts: ${KEEP_PROOF_ARTIFACTS}"
tmux attach -t "$SESSION"

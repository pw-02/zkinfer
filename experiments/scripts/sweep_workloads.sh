#!/usr/bin/env bash
set -euo pipefail
# export PYTHONPATH=.:$PYTHONPATH

NUM_WORKERS="${1:-1}"
COORDINATOR_HOST="${2:-127.0.0.1}"
COORDINATOR_PORT="${3:-50051}"

WORKLOADS=(
  # mnist_classifier
  mnist_gan
  # mobilenet_v2
  # nano_gpt_4_layers_64_embd
)

for WORKLOAD in "${WORKLOADS[@]}"; do
  echo "Running workload: ${WORKLOAD}"

  python experiments/run_local_exp.py \
    workload="${WORKLOAD}" \
    launch.num_workers="${NUM_WORKERS}" \
    launch.coordinator_host="${COORDINATOR_HOST}" \
    launch.coordinator_port="${COORDINATOR_PORT}" \
    launch.shutdown_when_done=true

  echo "Finished workload: ${WORKLOAD}"
done
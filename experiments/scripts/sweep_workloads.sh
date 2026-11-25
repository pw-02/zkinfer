#!/usr/bin/env bash
# Run the selected workloads with the research configuration.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
WORKLOADS=(mnist_gan)
for WORKLOAD in "${WORKLOADS[@]}"; do
  python -m experiments.run_local_exp --workload "$WORKLOAD" "$@"
done

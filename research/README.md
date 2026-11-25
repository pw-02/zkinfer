# zkInfer research results

This directory organizes evidence by research question. Raw data stays in
`experiments/runs/` or the original archived location. Existing `analysis/`
scripts provide analysis implementations; `experiments/` contains the active
launchers and configurations.

## Start here

Register one row per request and experiment in [experiment_index.csv](experiment_index.csv).
Use repository-relative paths where possible; external archives may use absolute
paths. The same run can support several experiments. Leave unknown fields blank;
never infer cold/warm state from a configuration flag alone. Use actual cache-hit
evidence. Record failures as well as successful requests.

| Experiment | Status (initial, provisional) |
| --- | --- |
| [E01 — Benefits of decomposition](E01_decomposition/README.md) | Existing evidence; verify provenance |
| [E02 — Proof job heterogeneity](E02_heterogeneity/README.md) | Existing evidence; extend to warm |
| [E03 — Temporal resource variation](E03_temporal_resources/README.md) | Partial; cold traces available |
| [E04 — Cold versus warm proving](E04_cold_vs_warm/README.md) | Priority now |
| [E05 — Resource repeatability](E05_repeatability/README.md) | Priority now |
| [E06 — Partition granularity](E06_granularity/README.md) | Planned |
| [E07 — Non-additivity](E07_non_additivity/README.md) | Planned; tooling needed |
| [E08 — Predictive cost model](E08_cost_prediction/README.md) | Conditional |
| [E09 — Cost-aware partition optimization](E09_partition_optimization/README.md) | Conditional on E06–E08 |
| [E10 — Concurrent proof serving](E10_concurrent_serving/README.md) | Runtime exists; evaluate |
| [E11 — Distributed scheduling and scaling](E11_distributed_scaling/README.md) | Preliminary; verify |
| [E12 — Combined end-to-end evaluation](E12_end_to_end/README.md) | Planned |

## Local run names

New runs launched through the small experiment YAML use descriptive directories:

```
2026-11-25_12-30-00_bert_p4_keys-off_c1_fifo_noadm_exact_streaming_circuits-off_history-off_learn-off_abcd1234
```

The name records the timestamp, workload, partition size (or `mono`), setup
reuse setting, concurrency cap (`cauto` for automatic admission), scheduling
policy, admission mode, witness preparation and dispatch, circuit reuse, history
reading/updating, and a unique suffix. `keys-on` and `circuits-on` enable reuse;
they do not imply a cache hit. Consult the effective configuration and measured
cache events for actual conditions. Existing run directories are unchanged.

## Workflow

1. Run the local experiment launcher or other explicitly documented runtime workflow; do not relocate raw runs.
2. Register its request ID, source path, experimental conditions and provenance in the master index.
3. Add analysis code to the relevant experiment directory (or reuse linked existing code).
4. Write derived tables to `results/` and publication plots to `figures/`.
5. Update the experiment README with findings, limitations and exact analysis commands.

Generated results and figures are ignored by default to avoid committing large
datasets. Deliberately selected publication outputs can be tracked explicitly.
The index is the single source of truth; no separate hand-maintained run lists.

## Measurement conventions

Use seconds and GiB (legacy reports may label GiB as GB). Distinguish job service
time, queue time, request latency and proof-generation makespan. A request is
complete only when every required sub-proof is available and consistent;
record verification status separately. Missing measurements are not zero.
Join job data using run/request/job/lease identities; distinguish retries.
Report cache warm-up separately and keep failed runs. For multi-node trials
record node hardware, thread budget and actual placement. Do not count nested
phase durations twice. Pin exact backend versions and record input shapes.

## Immediate priorities

E04/E03 warm isolated characterization, then E05 repeatability, then E06/E07
granularity and non-additivity. E08/E09 depend on predictive accuracy and cheap
feature availability. E10/E11 evaluate the existing runtime independently.
E12 combines only mechanisms supported by measured evidence.

Initial statuses describe prior project discussions, not validated results.
No run IDs or measurements have been invented.

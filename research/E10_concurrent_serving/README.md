# E10 — Concurrent proof serving

**Initial status:** Runtime exists; evaluate

## Research question

Can admission improve request-level performance safely?

## Protocol

Fixed concurrency, whole-job and stage admission; same partitions, cache condition, scheduling policy and arrivals.

## Measurements

Request throughput, median/p95 latency, makespan, admission waits, failures, memory and CPU utilization.

## Existing tooling / references

- [evaluation/02_concurrent_serving/README.md](../../evaluation/02_concurrent_serving/README.md)
- [evaluation/03_memory_admission/README.md](../../evaluation/03_memory_admission/README.md)

## Run registration

Add rows to [the master index](../experiment_index.csv) with experiment_id `E10`. Do not copy raw data into this directory.

## Findings

Pending verification and registration of source runs. Record measured findings here, with run IDs and reproducible analysis commands.

## Limitations and checks

Use request-level outcomes including failures; distinguish coarse EZKL admission from Halo2 instrumentation.

## Outputs

Derived tables in [results/](results/README.md); figures in [figures/](figures/README.md).

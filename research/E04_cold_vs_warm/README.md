# E04 — Cold versus warm proving

**Initial status:** Priority now

## Research question

What costs remain after setup reuse?

## Protocol

Same partitions and inputs; fresh setup versus verified setup hits; distinguish local and shared restore; one active job.

## Measurements

Job and request latency, stage timings, RSS, key sizes, load and transfer overhead.

## Existing tooling / references

- [evaluation/07_setup_reuse/README.md](../../evaluation/07_setup_reuse/README.md)

## Run registration

Add rows to [the master index](../experiment_index.csv) with experiment_id `E04`. Do not copy raw data into this directory.

## Findings

Pending verification and registration of source runs. Record measured findings here, with run IDs and reproducible analysis commands.

## Limitations and checks

Cache population runs are not measured warm runs; retain actual cache-hit evidence.

## Outputs

Derived tables in [results/](results/README.md); figures in [figures/](figures/README.md).

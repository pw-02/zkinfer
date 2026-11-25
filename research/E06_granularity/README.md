# E06 — Partition granularity

**Initial status:** Planned

## Research question

How does granularity affect work, feasibility and parallelism?

## Protocol

Sizes 1, 2, 4, 8, 16 and monolithic where feasible; start NanoGPT-4L and TinyBERT; isolated execution.

## Measurements

Number of jobs, sum of durations, longest job, max job RSS, key/proof sizes, boundary and verification costs.

## Existing tooling / references

- [evaluation/06_partition_granularity/README.md](../../evaluation/06_partition_granularity/README.md)

## Run registration

Add rows to [the master index](../experiment_index.csv) with experiment_id `E06`. Do not copy raw data into this directory.

## Findings

Pending verification and registration of source runs. Record measured findings here, with run IDs and reproducible analysis commands.

## Limitations and checks

Do not interpret longest job as actual makespan when jobs exceed worker slots.

## Outputs

Derived tables in [results/](results/README.md); figures in [figures/](figures/README.md).

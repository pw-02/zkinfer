# E11 — Distributed scheduling and scaling

**Initial status:** Preliminary; verify

## Research question

How do placement and worker count affect request completion?

## Protocol

FIFO, LPT and resource-aware; 1/2/4/8 workers where available; fixed workload and separate load scaling.

## Measurements

Request latency, throughput, speedup, transfer overhead, node utilization and fairness.

## Existing tooling / references

- [Runtime configuration](../../zkinfer/config/config.yaml)
- Dedicated distributed campaign tooling is not currently provided.

## Run registration

Add rows to [the master index](../experiment_index.csv) with experiment_id `E11`. Do not copy raw data into this directory.

## Findings

Pending verification and registration of source runs. Record measured findings here, with run IDs and reproducible analysis commands.

## Limitations and checks

Record physical nodes versus slots; actual scheduler semantics and shared-storage overhead; no assumed scalability.

## Outputs

Derived tables in [results/](results/README.md); figures in [figures/](figures/README.md).

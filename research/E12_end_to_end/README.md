# E12 — Combined end-to-end evaluation

**Initial status:** Planned

## Research question

Do partitioning and resource-aware execution deliver complementary improvements?

## Protocol

Baseline, partitioning only, scheduling/admission only, combined; matched hardware, inputs and proof semantics.

## Measurements

Verified request latency, throughput, failures, memory, total work, planning overhead.

## Existing tooling / references

- [Research experiment index](../experiment_index.csv)
- [Local experiment configuration](../../experiments/config/config.yaml)

## Run registration

Add rows to [the master index](../experiment_index.csv) with experiment_id `E12`. Do not copy raw data into this directory.

## Findings

Pending verification and registration of source runs. Record measured findings here, with run IDs and reproducible analysis commands.

## Limitations and checks

Include preparation, final proof composition and verification; report negative results and isolate caching effects.

## Outputs

Derived tables in [results/](results/README.md); figures in [figures/](figures/README.md).

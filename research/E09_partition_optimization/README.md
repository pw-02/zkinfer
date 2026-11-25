# E09 — Cost-aware partition optimization

**Initial status:** Conditional on E06–E08

## Research question

Do predicted costs improve feasible partition plans?

## Protocol

Uniform, layer/row-budget, additive DP, learned DP; measured-cost oracle on small graphs.

## Measurements

Measured request makespan, total work, longest job, peak RSS, planning overhead and proof verification.

## Existing tooling / references

- No dedicated optimizer verified.

## Run registration

Add rows to [the master index](../experiment_index.csv) with experiment_id `E09`. Do not copy raw data into this directory.

## Findings

Pending verification and registration of source runs. Record measured findings here, with run IDs and reproducible analysis commands.

## Limitations and checks

Minimax DP is exact only for its restricted objective; account for finite workers and actual schedule.

## Outputs

Derived tables in [results/](results/README.md); figures in [figures/](figures/README.md).

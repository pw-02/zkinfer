# E08 — Predictive cost model

**Initial status:** Conditional

## Research question

Can unseen candidate costs be estimated cheaply and conservatively?

## Protocol

Compare operator count, additive, domain size, analytical regression and tree-based models; cold/warm separately.

## Measurements

MAE, relative error, underprediction frequency/magnitude, feature extraction and prediction overhead.

## Existing tooling / references

- Use E05–E07 datasets
- no dedicated trainer verified.

## Run registration

Add rows to [the master index](../experiment_index.csv) with experiment_id `E08`. Do not copy raw data into this directory.

## Findings

Pending verification and registration of source runs. Record measured findings here, with run IDs and reproducible analysis commands.

## Limitations and checks

Group splits by partition identity and overlapping graph family; hold out models; avoid post-proving feature leakage.

## Outputs

Derived tables in [results/](results/README.md); figures in [figures/](figures/README.md).

# E05 — Resource repeatability

**Initial status:** Priority now

## Research question

How stable are repeated job costs?

## Protocol

Select 10–15 representative distinct partitions; 3 cold and 3 warm pilot repetitions, extend selected cases to 10; fixed threads.

## Measurements

Median, standard deviation, CV, observed peak RSS, memory underprediction magnitude/frequency.

## Existing tooling / references

- [experiments/microbench/README.md](../../experiments/microbench/README.md)
- [experiments/microbench/repeatability.py](../../experiments/microbench/repeatability.py)

## Run registration

Add rows to [the master index](../experiment_index.csv) with experiment_id `E05`. Do not copy raw data into this directory.

## Findings

Pending verification and registration of source runs. Record measured findings here, with run IDs and reproducible analysis commands.

## Limitations and checks

Separate identical-input and changed-input trials; do not mix synthetic and real inputs without labeling.

## Outputs

Derived tables in [results/](results/README.md); figures in [figures/](figures/README.md).

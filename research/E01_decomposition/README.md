# E01 — Benefits of decomposition

**Initial status:** Existing evidence; verify provenance

## Research question

Does decomposition reduce peak memory and enable larger models?

## Protocol

Monolithic versus partitioned sequential; same model, input, quantization, cache state and threads.

## Measurements

Request latency, aggregate job time, peak process RSS, verification outcome; OOM as feasibility outcome.

## Existing tooling / references

- [analysis/section_3/analyze_characterization.py](../../analysis/section_3/analyze_characterization.py)

## Run registration

Add rows to [the master index](../experiment_index.csv) with experiment_id `E01`. Do not copy raw data into this directory.

## Findings

Pending verification and registration of source runs. Record measured findings here, with run IDs and reproducible analysis commands.

## Limitations and checks

Reconcile timing scopes and headline speedups before publication.

## Outputs

Derived tables in [results/](results/README.md); figures in [figures/](figures/README.md).

# E03 — Temporal resource variation

**Initial status:** Partial; cold traces available

## Research question

Does temporal resource variation persist in warm proving?

## Protocol

Isolated cold/warm jobs, representative small/medium/large partitions; sample at 0.1 s initially.

## Measurements

Time-weighted mean RSS, peak RSS, peak/mean, near-peak duration, phase coverage and retained memory.

## Existing tooling / references

- [analysis/section_3/analyze_characterization.py](../../analysis/section_3/analyze_characterization.py)
- [analysis/section_3/packing_motivation.py](../../analysis/section_3/packing_motivation.py)

## Run registration

Add rows to [the master index](../experiment_index.csv) with experiment_id `E03`. Do not copy raw data into this directory.

## Findings

Pending verification and registration of source runs. Record measured findings here, with run IDs and reproducible analysis commands.

## Limitations and checks

Internal Halo2 phases are observational, not admission boundaries; slack is not necessarily safe capacity.

## Outputs

Derived tables in [results/](results/README.md); figures in [figures/](figures/README.md).

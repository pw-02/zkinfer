# E07 — Non-additivity

**Initial status:** Planned; tooling needed

## Research question

Are combined partition costs poorly predicted by sums of isolated operator costs?

## Protocol

Sample 50–100 overlapping candidate windows across architectures; retain exact ordered operator IDs, model and input provenance.

## Measurements

Combined time / sum of isolated times; combined peak RSS / max isolated RSS; boundary size and feature extraction cost.

## Existing tooling / references

- No dedicated overlapping-window generator verified.

## Run registration

Add rows to [the master index](../experiment_index.csv) with experiment_id `E07`. Do not copy raw data into this directory.

## Findings

Pending verification and registration of source runs. Record measured findings here, with run IDs and reproducible analysis commands.

## Limitations and checks

Use matched cache conditions; preserve failed/OOM cases; avoid changing quantization silently.

## Outputs

Derived tables in [results/](results/README.md); figures in [figures/](figures/README.md).

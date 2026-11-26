# E13 — Storage-aware proving-key materialization

**Initial status:** Implementation committed; native build and experimental validation pending.

## Research question

When is reconstructing omitted proving-key polynomial representations preferable to retrieving precomputed data from storage? How much persistent storage and aggregate I/O can be saved without changing the proving protocol?

## Representations

| Representation | Persisted components | Reconstruction |
| --- | --- | --- |
| Full | VK, fixed/permutation Lagrange, coefficient and coset polynomials, boundary polynomials | None |
| Reduced | Full minus fixed/permutation cosets | Extended-domain cosets |
| Compact | VK and fixed/permutation Lagrange polynomials | Coefficients, cosets and boundary polynomials |

All materialize the ordinary in-memory PK. This is not an out-of-core or peak-memory optimization. VK serialization is unchanged.

## Protocol

1. Rebuild EZKL against the instrumented Halo2 revision; check small-model proof verification for all three representations.
2. Start with monolithic `nano_gpt_2_layers_32_embd`; reuse the same calibrated settings and compiled circuit, same inputs, SRS, hardware and fixed thread budget. Add other feasible workloads/partitions after the pilot.
3. Run each representation with `cache.circuits: true`, `cache.keys: false`, `profiling.mode: research`, one active job, resource history disabled. Confirm circuit hits and actual PK representation. Record at least three successful repeats per condition and retain failures. Randomize order where practical.
4. Separately measure warm key reuse using prepopulated, separate cache roots for each representation; verify actual hits. Distinguish OS page-cache state from zkInfer artifact-cache state. Do not assume a freshly generated PK is a cold filesystem read.
5. Compare effective loading conditions (local/shared storage or controlled bandwidth, if available). Document storage path, concurrent readers, hardware and page-cache policy; avoid disruptive global cache drops on shared machines.
6. Optionally compare concurrent readers with the same total CPU budget and sufficient memory. Report throughput and latency as well as aggregate bytes. GPU experiments and automatic selection are out of scope for the initial evaluation.

Example experiment configuration (merge into existing YAML):

```yaml
workload: nano_gpt_2_layers_32_embd
partitioning:
  enabled: false
proving:
  key_representation: compact # repeat for full and reduced
cache:
  circuits: true
  keys: false
scheduling:
  max_concurrent_jobs: 1
  memory_admission: disabled
resource_model:
  use_history: false
  update_history: false
profiling:
  mode: research
```

Run with `python -m experiments.run_local_exp`. Keep all other settings identical. Set a fixed feasible `resources.prover_threads` for the comparison.

## Measurements

Use per-job `halo2_key_metrics.csv` and `reports/job_report.csv`:

- Actual and requested representation, key file size, logical bytes deserialized.
- PK serialization, deserialization, reconstruction and complete materialization times.
- Key generation, EZKL proving call, end-to-end request latency and peak RSS.
- Verification result, cache hit, hardware, thread count, storage and backend revisions.

Materialization contains deserialization and reconstruction; do not sum nested timings. Serialization includes buffered flush, not durable fsync. Logical bytes are not physical disk reads. Missing metrics are not zero; cached keys have no new serialization measurement.

## Analytical model

For representation r and worker w, estimate materialization as T(r,w) = S(r)/B(w) + C(r,w), where S is serialized bytes, B is effective loading throughput and C is reconstruction time. Choose the available representation minimizing T. For a smaller representation versus full, the pairwise break-even throughput is (S(full) - S(r)) / C(r,w). This is a simplified model: validate against independently measured loads, report residual overhead and avoid using the same observation for both prediction and validation.

## Preliminary evidence (unregistered pilot)

User-supplied component measurements for NanoGPT 2 layers, embedding 32 suggest the following estimates. These are not measured reduced/compact runs; original run ID and provenance still need registration.

| Representation | Estimated GiB | Estimated reduction | Reconstruction proxy (s) |
| --- | ---: | ---: | ---: |
| Full | 151.06 | 0% | 0 |
| Reduced | 35.06 | 76.8% | 154.88 |
| Compact | 14.56 | 90.4% | 187.36 |

Cosets account for 116 GiB. Reconstruction proxies sum initial key-generation transformation timings, not actual loading reconstruction timings. Predicted selection thresholds are approximately 0.63 GiB/s (compact/reduced) and 0.75 GiB/s (reduced/full); validate rather than report as observed results.

## Outputs

- Table of size, serialization, deserialization, reconstruction, materialization, peak RSS and proof/request latency per representation.
- Stacked materialization breakdown and predicted versus measured latency across bandwidths.
- Optional concurrent-reader throughput and aggregate I/O comparison.

Derived tables in [results/](results/README.md); plots in [figures/](figures/README.md).

## Run registration and findings

Register source runs in [the master index](../experiment_index.csv) under `E13`. Record representation, storage condition, page-cache condition and actual cache hits in notes; pin backend commits. Raw data remains in `experiments/runs/` or its original archive. No run IDs have been inferred. Record reproducible analysis commands and verified findings here as experiments complete.

## Implementation references

- Halo2 component profiling: [cfbb257](https://github.com/pw-02/halo2/commit/cfbb25779badb3d050283f4e34c77706cb0a2da9)
- EZKL file-level profiling: [87ec4c4](https://github.com/pw-02/ezkl/commit/87ec4c40079a1a328b7c4ad080fc930cefee16be)
- zkInfer report integration: [435fca5](https://github.com/pw-02/zkinfer/commit/435fca54f03cd7fe67ef482a242021690830c80a)
- [Configuration guide](../../docs/CONFIGURATION.md)

### Optional targeted page-cache control

Use `proving.evict_pk_page_cache: true` for isolated cold-page-cache trials; retain `false` for the normal baseline. Record eviction status, cached bytes before/after and eviction/probe duration from `job_report.csv`. A successful advisory call alone is not evidence of a cold read. Compare verified residency, keep unknown residency separate, and avoid concurrent reads of the same inode during isolated trials. Native PK materialization excludes eviction preparation; outer proving time includes it.

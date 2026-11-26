# EZKL backend notes

zkInfer uses the standard EZKL setup and prove APIs, with custom Halo2 profiling. Circuit and key caching are independently controlled by the experiment configuration.

## Cold versus warm proving

A cold key execution calls `ezkl.setup` to generate and serialize the VK and PK, then calls `ezkl.prove` to load the PK and construct the proof. A warm key execution skips setup and uses compatible cached keys. The witness is generated per request during coordinator preparation.

The key generation time includes key serialization; proof time includes key loading and proof serialization. Cold jobs still require temporary PK storage when persistent key caching is disabled.

Halo2 phase and key-component instrumentation remain available.

## Stage-labelled resource traces

Each prover writes its current stage to `status.txt`. `resource_usage.csv` samples the prover process tree and records the current stage alongside CPU and RSS. Exact transition timestamps are stored separately, so stage duration does not depend on the resource sampling interval.

Typical runtime labels include:

```text
CALIBRATING
COMPILING
GETTING_SRS
GENERATING_WITNESS
KEY_GEN
PROVING
DONE
```

When a prover is waiting for local admission, the trace records `WAITING_FOR_<STAGE>` rather than attributing the wait to the preceding stage.

## Node-local stage admission

`StageAdmissionController` gates major EZKL stages using predicted peak memory for the same setup key. `StageResourceHistory` stores recent node-local observations from successful runs.

With fewer than five prior memory observations, the estimate is the observed maximum. Once enough history exists, the controller uses the configured historical percentile. If no stage history exists, it falls back to the job's whole-job peak estimate.

The configuration is:

```yaml
node:
  stage_admission:
    enabled: true
    memory_safety_reserve_gb: 8
    history_percentile: 0.90
```

`memory_safety_reserve_gb` is a fixed node reserve that remains unavailable to proof workloads independently of prediction history.

CPU is measured for characterization but is not an admission constraint.

Stage-admission metrics include wait time, predicted memory, actual memory, prediction error, estimate source, and historical sample count.

## Recommended comparisons

For cold versus warm setup, compare end-to-end latency, setup/key-generation time, proof time, setup restoration/persistence, proving-key size, and peak memory by stage.

For the node admission mechanism, compare:

```yaml
node:
  stage_admission:
    enabled: false
```

against the same workload with admission enabled, holding hardware, partitioning, setup condition, and coordinator scheduling policy fixed. Useful outcomes include throughput, request latency, makespan, peak memory, OOM/failure rate, admission wait time, and memory-prediction error.

# zkInfer configuration

zkInfer separates experiment-facing configuration from runtime defaults.

- `experiments/config/config.yaml` describes the experiment you want to run.
- `zkinfer/config/config.yaml` contains runtime defaults for the coordinator and node processes.
- `experiments/run_local_exp.py` translates the experiment-facing settings into runtime overrides.

The experiment configuration intentionally exposes only choices that are useful for experiments or system behavior. Networking, heartbeat timing, coordinator state paths, and other plumbing remain runtime settings.

## Partitioning

The experiment configuration uses two fields only:

```yaml
execution:
  partitioning_enabled: true
  partition_size: 1
```

`partitioning_enabled: false` creates one proof job for the whole model.

`partitioning_enabled: true` partitions the model into consecutive groups of `partition_size` meaningful ONNX operators. A `partition_size` of `1` therefore gives one proof job per meaningful operator.

The runtime still translates these settings to the splitter's internal `none` and `fixed` modes, but those strings are not experiment-facing configuration.

## Request preparation

The synchronous gRPC server acknowledges an accepted request after registration
(and persistence when the state store is enabled). Calibration, compilation,
and witness generation run in a separate bounded preparation executor.

```yaml
preparation:
  dispatch: streaming # batch | streaming
  workers: 1
  max_pending: 8
```

These settings are translated to `coordinator.preparation_dispatch`,
`coordinator.preparation_workers`, and `coordinator.preparation_max_pending`.
At most workers + max_pending requests occupy preparation slots. Full queues
return gRPC RESOURCE_EXHAUSTED immediately; no request ID is issued. For burst
experiments configure capacity for the intended offered load, or report
rejections as outcomes. The batch submitter records explicit rejections.

An acknowledgement is not proof of successful preparation. Poll the request
status until completed or failed. Batch releases jobs after full preparation;
streaming releases each ready job but verifies only a sealed complete graph.
Shutdown drains accepted preparation before shutting down the finalizer.
Interrupted unsealed preparation is failed on restart, not resumed.

`preparation_queue_wait_time(s)` measures executor queue delay separately.
First/all-job-ready and preparation-finished times retain their existing
preparation-start-relative meaning. Total request elapsed time includes
queueing, preparation, execution and verification; overlapping stages are
not additive.

## Coordinator scheduling

The coordinator chooses which whole proof job to lease to a polling node. The experiment-facing setting is:

```yaml
coordinator:
  scheduling_policy: resource_aware
```

Supported policies are:

- `fifo` gives request-fair queue order.
- `lpt` chooses the longest predicted job within the selected request.
- `resource_aware` first rejects jobs whose predicted peak cannot fit on the machine, then prioritizes predicted duration within the selected request.

Request fairness is applied before the within-request policy. Stage-level packing is not a coordinator responsibility.

## Node concurrency

```yaml
node:
  max_concurrent_jobs: 1
```

This is an optional hard whole-job cap.

- `1` gives the isolated sequential baseline.
- `N` gives a fixed-concurrency baseline.
- `null` removes the fixed job-count cap so local stage admission can control memory safety.

`max_concurrent_jobs` is not the main zkInfer scheduling policy. It is a safety or baseline override.

## Stage admission

The node decides when an already assigned proof job may enter its next execution stage.

```yaml
node:
  stage_admission:
    enabled: true
    memory_safety_reserve_gb: 8
    history_percentile: 0.90
```

`enabled` turns node-local stage admission on or off.

`mode` selects `stage` (the backward-compatible default) or `whole_job`.
Both modes require `enabled: true`. Whole-job mode reserves the coordinator's
predicted job peak before the prover pipeline starts and retains it through
preparation, proving, and output publication, releasing it on success or error.
Stage callbacks still record timings but cannot resize or release that reservation.
If node-local stage history exists, whole-job mode uses the maximum estimated
stage peak, retaining a larger coordinator-provided job estimate when available.
This lets both admission modes start from the same precollected node profiles.
Artifact materialization in the supervisor precedes this prover reservation.
Without a positive job estimate, the existing conservative available-memory
fallback applies; collect matching cold/warm profiles before comparison runs.

Use these configurations for controlled admission comparisons, with the same
`resource_aware` coordinator policy, artifact state, and memory reserve:

| Configuration | `max_concurrent_jobs` | `stage_admission.enabled` | `stage_admission.mode` |
| --- | --- | --- | --- |
| Fixed concurrency (k) | k | false | stage |
| Whole-job admission | null | true | whole_job |
| Stage admission | null | true | stage |

For example, run the whole-job baseline with:

```bash
python -m experiments.run_local_exp coordinator.scheduling_policy=resource_aware node.max_concurrent_jobs=null node.stage_admission.enabled=true node.stage_admission.mode=whole_job
```

Keep artifact reuse and profiling settings identical across the compared runs.
The launcher forwards an explicit null job cap so the runtime's sequential
default does not silently override the experiment.

`memory_safety_reserve_gb` is RAM that zkInfer intentionally leaves unavailable to proof workloads. It is a fixed safety margin and is independent of historical predictions.

`history_percentile` controls the historical stage-memory estimate after enough observations exist. Sparse history uses the observed maximum. The default policy uses memory for admission. CPU is still measured for characterization but is not a configurable admission constraint.

For isolated Section 3 characterization, use:

```yaml
node:
  max_concurrent_jobs: 1
  stage_admission:
    enabled: false
```

For resource-aware concurrent execution after profiles have been collected, use:

```yaml
node:
  max_concurrent_jobs: null
  stage_admission:
    enabled: true
```

## Resource sampling

```yaml
node:
  resource_sampling_interval_sec: 0.25
```

This controls periodic RSS/CPU samples written to `resource_usage.csv`. Exact stage-transition timestamps are recorded separately and do not depend on the sampling interval.

## Storage

The storage configuration is organized by data lifetime.

```yaml
storage:
  reusable_setup:
    enabled: true
    path: storage/cache

  request_data:
    path: storage/requests

```

`reusable_setup` contains circuit-specific state that can be reused across requests, including the partition model, settings, compiled circuit, proving key, and verification key. `enabled: false` gives the cold setup baseline. `enabled: true` permits reuse when the setup identity matches.

`request_data` contains request-owned inputs, proofs, witnesses, and manifests.

Resource history is controlled independently of circuit/key reuse:

```yaml
storage:
  reusable_setup:
    enabled: false
  resource_profiles:
    enabled: true
    path: storage/profiles  # null keeps profiles under reusable_setup.path
```

This configuration regenerates setup for every job while using earlier resource
measurements for scheduling. Successful cold jobs also update their history.
Profiles use the storage backend/bucket configured for the runtime. The layout is
`<profile root>/<model namespace>/<partition namespace>/cold_resource_profile.json` when setup reuse is disabled
and `<profile root>/<model namespace>/<partition namespace>/resource_profile.json` when it is enabled. The latter
preserves the existing reuse-enabled path. No fallback loads a warm profile into
a forced-cold job. Reuse-enabled history may include initial cache misses; prepare
and snapshot warm-only measurements for controlled warm evaluation.

`resource_profiles.enabled: false` disables history reading; `update_history` independently controls persistence
without disabling circuit/key reuse. Node-local stage history remains separately
managed under `node.runtime_state_dir`; use matching snapshots in experiments.
Changing `resource_profiles.path` does not copy existing history automatically.

Workers materialize artifacts in private job workspaces; no persistent worker artifact cache is used. Filesystem cache namespaces use 16 hexadecimal SHA-256 characters in directory names, with full identities recorded and checked in `.identity.sha256` files. Old full-hash directories are not migrated automatically.

The lower-level runtime configuration also supports `filesystem` and `s3` backends. For S3, the configured paths are interpreted as prefixes beneath `storage.s3_prefix`.

## Output retention

The experiment configuration uses two explicit switches:

```yaml
output:
  keep_proof_artifacts: false
  keep_request_data: false
```

`keep_proof_artifacts` controls whether bulky proof-generation artifacts are copied into `experiments/runs/.../artifacts/jobs/...` for debugging. Traces, logs, manifests, and reports are retained regardless of this setting.

`keep_request_data` controls whether the canonical request-owned files under `storage.request_data.path/<request_id>/` are preserved after the experiment finishes. When it is `false`, the launcher removes only the requests submitted by that experiment. Reusable setup, node state, and experiment records are preserved.

These settings affect different copies of the data. A typical characterization run therefore uses both as `false` to keep reports and traces without retaining duplicate proof files or stale request-owned storage.

## Runtime-only settings

The runtime defaults also contain settings that are not research-policy knobs, including coordinator host/port, gRPC message limits, heartbeat timing, coordinator state storage, verification configuration, and the node runtime-state directory. They remain configurable for deployment but are intentionally not duplicated in the experiment-facing policy surface.

## Useful experiment configurations

Isolated fine-grained characterization:

```yaml
execution:
  partitioning_enabled: true
  partition_size: 1

node:
  max_concurrent_jobs: 1
  resource_sampling_interval_sec: 0.25
  stage_admission:
    enabled: false

storage:
  reusable_setup:
    enabled: false

output:
  keep_proof_artifacts: false
  keep_request_data: false
```

Resource-aware runtime:

```yaml
coordinator:
  scheduling_policy: resource_aware

node:
  max_concurrent_jobs: null
  stage_admission:
    enabled: true
    memory_safety_reserve_gb: 8
    history_percentile: 0.90

storage:
  reusable_setup:
    enabled: true
```

## Terminology

The configuration uses the following terms consistently:

- **partitioning** describes how a model becomes proof jobs.
- **coordinator scheduling** chooses which whole proof job is leased to a node.
- **stage admission** decides when an assigned job may enter its next execution stage.
- **reusable setup** is circuit-specific state reused across requests.
- **request data** is owned by one inference request.
- **local cache** is node-local materialized data.

The experimental `transfer_only` runtime mode has been removed. Transfer measurements remain available as part of normal proof execution without changing proof-job semantics.

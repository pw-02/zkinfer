# zkInfer

zkInfer is a distributed runtime for partitioned zkML proof serving with an EZKL/Halo2 backend. The `sacdad27` branch focuses on resource-aware proof execution, reusable setup state, request-fair placement, node-local memory admission, verification, recovery, and reproducible experiment accounting.

## Architecture

```text
Client(s)
   |
   v
Coordinator
  |- partitions each request into proof jobs
  |- maintains request-fair global scheduling state
  |- chooses feasible jobs for polling nodes
  |- considers reusable-setup locality
  |- persists leases and request state
  `- requeues jobs after node failure
          |
          | gRPC control plane
          v
NodeSupervisor                         one per machine
  |- reports fresh node and cache state
  |- owns the bounded local cache
  |- restores reusable setup
  |- launches one isolated prover process per proof job
  `- applies node-local stage admission
          |
          v
Prover subprocess
  |- executes EZKL/Halo2 using local paths
  |- requests admission before major execution stages
  |- records exact stage transitions and sampled resources
  `- publishes request-owned proof outputs
```

Large models, witnesses, proving keys, and proofs are stored through the data plane rather than transported inside gRPC messages.

## Configuration

The experiment-facing configuration intentionally uses a small vocabulary. See `docs/CONFIGURATION.md` for the complete description.

```yaml
execution:
  partitioning_enabled: true
  partition_size: 1
  simplify_model: true

coordinator:
  scheduling_policy: resource_aware

node:
  max_concurrent_jobs: null
  resource_sampling_interval_sec: 0.25
  stage_admission:
    enabled: true
    memory_safety_reserve_gb: 8
    history_percentile: 0.90

storage:
  reusable_setup:
    enabled: true
    path: storage/setup
  request_data:
    path: storage/requests
  local_cache:
    path: storage/node/cache
    max_gb: 50

output:
  keep_proof_artifacts: false
  keep_request_data: false
```

### Partitioning

`partitioning_enabled: false` creates one proof job for the whole model. With partitioning enabled, `partition_size` is the number of consecutive meaningful ONNX operators grouped into each proof job. `partition_size: 1` gives the fine-grained characterization configuration.

### Coordinator scheduling

The coordinator decides which whole proof job to lease to a node. Request fairness is applied before the within-request policy. Supported policies are `fifo`, `lpt`, `cache_aware`, and `resource_aware`.

The coordinator does not perform stage-level packing. For the resource-aware policy it rejects work whose predicted peak cannot fit the machine at all, then uses setup locality and predicted duration as placement preferences.

### Stage admission

`StageAdmissionController` decides when an already assigned proof job may enter its next major execution stage. Memory is the admission resource.

Each setup key has a node-local `StageResourceHistory`. With fewer than five prior memory observations for a stage, the estimate is the observed maximum. Once at least five observations exist, the estimate uses `history_percentile`. If no stage history exists, zkInfer falls back to the job's whole-job peak estimate.

`memory_safety_reserve_gb` is a fixed amount of node RAM that is never made available to proof workloads. It is independent of historical predictions.

CPU is still sampled for characterization, but CPU demand does not gate stage admission.

The principal stages currently exposed to the controller are:

```text
CALIBRATING
COMPILING
GETTING_SRS
GENERATING_WITNESS
KEY_GEN
PROVING
FUSED_KEYGEN_PROVE
```

The additional EZKL/Halo2 tracing around parameter loading and key-generation internals is characterization instrumentation only unless a later experiment establishes a useful controllable boundary.

### Node concurrency

`max_concurrent_jobs` is an optional hard whole-job cap. It is useful for isolated and fixed-concurrency baselines, but it is not the primary zkInfer resource policy.

```yaml
node:
  max_concurrent_jobs: 1      # sequential baseline
```

```yaml
node:
  max_concurrent_jobs: null   # no count cap
  stage_admission:
    enabled: true
```

When the fixed cap is absent, provisional memory reservations prevent the polling loop from leasing an uncontrolled burst of jobs before newly started prover processes become visible in live RSS.

## Resource and stage traces

Each proof job records periodic process-tree CPU/RSS samples in `resource_usage.csv`. The sampling frequency is controlled by:

```yaml
node:
  resource_sampling_interval_sec: 0.25
```

Stage transitions have their own exact timestamps and therefore do not depend on the resource sampling period. Waiting for admission is represented explicitly as `WAITING_FOR_<STAGE>` so admission delay is not attributed to the preceding execution stage.

Stage-admission metrics use the `stage_admission_` prefix, including values such as:

```text
stage_admission_enabled
stage_admission_wait_time(s)
stage_admission_proving_wait_time(s)
stage_admission_proving_predicted_memory(GB)
stage_admission_proving_actual_memory(GB)
stage_admission_proving_memory_error(GB)
stage_admission_proving_memory_abs_pct_error
stage_admission_proving_estimate_source
stage_admission_proving_estimate_samples
```

CPU measurements remain available through the ordinary stage resource metrics, for example `stage_proving_avg_cpu_machine(%)`, but are not admission reservations.

## Reusable setup and request data

Shared storage is separated by lifetime.

```text
reusable setup
  <setup_key>/
    model.onnx
    settings.json
    network.compiled
    pk.json.gz
    vk.json
    resource_profile.json

request data
  <request_id>/
    inputs/
    proofs/
```

`storage.reusable_setup.enabled: false` gives a cold setup baseline. When it is enabled, compatible settings, compiled circuits, proving keys, and verification keys can be restored and reused.

The setup key includes the partition model and versioned setup-policy identity. A setup-policy change therefore does not silently reuse incompatible proving state.

Node-local materialized copies live in the local cache. Setup locality is a scheduling preference, not a reason to override machine-feasibility or local stage-admission checks.

## Output retention

The experiment layer distinguishes two retention decisions:

```yaml
output:
  keep_proof_artifacts: false
  keep_request_data: false
```

`keep_proof_artifacts` controls whether bulky generated proof files are also copied into the experiment run directory. Traces, logs, manifests, and reports remain available regardless.

`keep_request_data` controls whether canonical request-owned inputs and proof outputs in request storage are preserved after the experiment. Cleanup never removes reusable setup or experiment records.

## Failure and retry behavior

Each leased job carries a lease identifier. Results from stale leases are rejected. Ordinary proof failures use bounded retries. If a node stops heartbeating, its in-progress jobs are returned to the coordinator queue and may be leased elsewhere. Successful terminal results are idempotent, which allows a worker to retry completion acknowledgement without publishing a second accepted result.

Jobs are not migrated while executing.

## Section 3 characterization

A typical isolated characterization configuration is:

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
```

The research plan and experiment registration workflow are documented in [research/README.md](research/README.md).

## Main entry points

```bash
# Local experiment suite
python experiments/run_local_exp.py

# Submit one configured workload to an already running coordinator
python experiments/submit_job.py +workload=mnist_gan

# Characterization analysis
python analysis/section_3/analyze_characterization.py \
  --runs-dir experiments/runs \
  --output-dir analysis/section_3/generated

# Stage-admission analysis
python analysis/section_3/analyze_stage_admission.py <run-or-report-dir>
```

The local tmux launchers are under `experiments/scripts/` for filesystem and S3-backed runs.

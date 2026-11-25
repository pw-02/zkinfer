# Experimental upfront witness preparation

This default execution path retains the original independent ORT path for
performance comparisons. It does not modify EZKL or change the scheduler's
eligibility rules. The existing custom EZKL ccgrid wheel is required.

## Run

In experiments/config/config.yaml:

    partitioning:
      enabled: true
      size: 2
      simplify: true
      witness_preparation: upfront
      max_propagated_scale: 16
      allow_boundary_rescale: true

    output:
      keep_proofs: true
      keep_cache: true

Keep the other existing configuration fields. Run:

    bash experiments/scripts/launch_tmux.sh

Start with mnist_classifier and one concurrent job. The simple launcher enables
verification automatically for upfront mode. Direct runtime users must set
coordinator.witness_preparation=upfront and verification.enabled=true.
Upfront exact verification is now the default. Explicitly select independent to reproduce the unlinked baseline, and disable verification for direct runtime baseline runs. S3 runs must select independent until prepared-artifact transport is implemented.

## Design

- The ONNX splitter still supplies representative inputs and static shapes.
  Exact mode refuses missing partition sources or missing model outputs.
- Prepare partitions in topological dependency order by ONNX tensor name.
- Calibrate/compile with public inputs, public outputs and fixed parameters.
  Restrict candidate scales when incoming scales agree and are below the
  configured cap. Actual settings are always authoritative.
- For mixed/high scales or failed restricted calibration, optionally use
  ordinary independent calibration and deterministic integer rescaling.
  Downscaling rounds to nearest with ties away from zero; upscaling shifts
  left. This defines a specific composed quantised computation, not an
  assertion of bitwise equivalence to monolithic EZKL. Set
  allow_boundary_rescale=false to reject all differing boundary scales.
- Generate exactly one witness per partition per request. Calibration
  itself still performs EZKL forward passes; those are not proof witnesses.
- Cache settings/compiled circuits independently of key generation when
  setup caching is enabled. The preparation cache resides under
  storage/setup/_prepared_v1. The preparation cache uses model bytes,
  scale policy and backend identity; it reuses the first calibrated
  configuration on later requests, as with existing setup reuse. A new
  input distribution may need recalibration. Disable setup caching to
  measure cold runs.
- Bind the worker setup identity to the exact settings/compiled bytes.
  Transport the prepared artifact descriptor with the existing input
  ArtifactRef. The worker checks digests, restores the circuit/settings,
  loads the witness and uses the existing keygen/proving stages.
- Verify each proof against coordinator-prepared settings, extract its
  authenticated instances, check expected request witnesses, and check
  every graph edge using exact integer arithmetic. No tolerance window,
  witness-only linking or PolyCommit swap is used on this path.

## Trust and scope

This is an experimental local/shared-filesystem implementation. S3 mode is
rejected explicitly, not silently downgraded. Separate hosts must use a common
trusted SRS; the existing local SRS workflow is not a distributed trust setup.

By default, key generation stays on trusted proving workers, consistent with
local systems experiments. This is not a malicious-worker model-binding
guarantee: fixed parameters are bound by a correctly generated VK, not by
trusting a worker-supplied VK. For externally trusted VKs, set
verification.trusted_vk_manifest (in the simple YAML or direct runtime) to
a JSON mapping of exact setup keys to independently generated/approved
verification-key SHA-256 digests. A missing or mismatching pin fails
verification. The verification report explicitly records the key trust mode.
The coordinator's prepared graph, settings, expected request inputs and SRS
must also be trusted and protected from untrusted workers.

Public linking discloses all partition inputs and outputs. No privacy claim
is made for these activations. The guarantee concerns the composed prepared
quantised partitions with explicit rebasing; validate model accuracy and
partition semantics separately, especially dynamic shapes and fused or
repeated graph operators. Unsupported EZKL interfaces fail closed. The
existing ONNX splitter freezes some shape inputs; general dynamic-shape
correctness is outside this initial patch.

## Tests and evaluation

    python -m pytest tests/test_exact_witness_preparation.py
    python -m pytest tests/test_launcher_config.py tests/test_runtime_core.py tests/test_shared_local_setup.py

The added unit tests cover field encoding, signed exact rounding, graph
fan-out, multi-input order, invalid boundaries, cache reuse, and the
one-witness orchestration with a mock backend. Run with the real ccgrid
wheel for the essential integration tests:

1. MNIST, partition sizes 2 and 1, then nanoGPT, increasing workers.
2. Keep proofs and check verification_report.json, particularly
   split_chain.valid, checked_edges, rescaled_edges and key_trust.
3. Verify proof rejection for a tampered public instance. Separately test
   two valid individual proofs with inconsistent boundaries; the graph
   verification must reject them.
4. Compare independent and upfront under matched model, input, hardware,
   cache policy, partition size and scheduling configuration. Also isolate
   the cost of public/fixed visibility if interpreting the overhead.
5. Compare warm and cold runs. Use total_elapsed_time(s), not just
   request_runtime(s), since preparation moved before enqueueing.
6. Measure model accuracy, logrows, coordinator RSS and witness/circuit
   transfer. Request report columns include preparation time, calibration,
   compilation, witness time and bytes, cache hits, rescaled inputs and
   worker witness-load time. The coordinator RSS column is a process-wide
   high-water mark, not an isolated per-request measurement.

No real EZKL end-to-end run or performance result is implied by the mock
tests. Approximate linking is deliberately not implemented: it would be a
different, explicitly weaker verification policy.

## Streaming preparation and cache reuse

The experiment YAML now accepts:

```yaml
preparation:
  cache_enabled: true
  dispatch: streaming # batch | streaming
cache:
  setup: true # PK/VK reuse (independent of preparation cache)
```

Streaming publishes a job after its prepared artifacts and validated witness are available, in topological preparation order. The existing scheduler selects among ready jobs throughout preparation. The request is sealed only after all planned jobs are published. Verification cannot accept an unsealed or partially prepared graph. A preparation error fails the request after already published jobs drain; restart during preparation marks the incomplete request failed rather than resuming it. The submit RPC still returns after preparation, while worker polling and proving proceed concurrently on other gRPC threads. At least two gRPC threads are required; many simultaneous submissions may contend for the pool.

The preparation cache reuses settings and compiled circuits keyed by partition model/backend/scale configuration. It is populated on first use, not by a separate offline representative calibration dataset. Warm witnesses are still generated from the current request, and out-of-range inputs must fail rather than silently accept an invalid proof. Use fresh cache roots for cold experiments. With preparation.cache_enabled=false, calibration and compilation run for each prepared partition even if cache.setup is true. This independent cache setting applies to upfront preparation; the legacy independent worker path still follows cache.setup.

Compare batch and streaming with preparation cache off/on, and independently hold key caching and profiles constant. Reports include first_job_ready_time(s), all_jobs_ready_time(s), preparation_finished_time(s), preparation_overlap_time(s) (window between first lease and end of preparation, not measured active proving time), preparation_wall_time(s), and expected_proof_jobs. The existing preparation_time(s) is time to first queue publication, and request_runtime(s) can include overlapping preparation in streaming mode. Do not sum overlapping intervals as if they were disjoint. Run names record dispatch and preparation cache settings.

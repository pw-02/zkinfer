# Configuration

Edit `experiments/config/config.yaml` and run
`bash experiments/scripts/launch_tmux.sh`. The launcher merges it with
`zkinfer/config/config.yaml`, the canonical standalone runtime defaults.
Both use the same vocabulary. The effective configuration is saved in each run.
The old intermediate runtime_experiment.yaml and S3 launcher are retired.

## Research settings

- **partitioning.enabled, size, simplify**: choose monolithic or contiguous
  operator partitions and optional ONNX simplification.
- **partitioning.max_propagated_scale**: maximum incoming fixed-point scale
  eligible for restricted calibration. Zero normally avoids propagating positive
  scales; it does not set the circuit scale to zero.
- **partitioning.allow_boundary_rescale**: allow deterministic, verified conversion
  between differing fixed-point scales. Downscaling can round values; compare
  numerical accuracy separately.
- **preparation.dispatch**: batch releases jobs after all witnesses are ready;
  streaming releases each ready job. Both require exact linked preparation.
- **preparation.workers, max_pending**: concurrent request preparations and
  additional waiting requests; full queues return RESOURCE_EXHAUSTED.
- **scheduling.policy**: fifo, lpt or resource_aware; request fairness first.
- **scheduling.max_concurrent_jobs**: per-node cap; null requires admission.
- **scheduling.memory_admission**: disabled, whole_job or stage.
- **scheduling.memory_reserve_gib, history_percentile**: fixed reserve and
  historical memory estimator; sparse histories use the observed maximum.
- **resources.coordinator_threads, prover_threads**: independent Rayon pool
  sizes; null removes inherited overrides and enables automatic detection.
  Not CPU affinity or CPU admission.
- **cache.circuits, keys**: independent compatible artifact reuse.
- **resource_model.use_history, update_history**: independent history reading
  and updating. History is separated by configured prover thread count.
- **profiling.mode, sampling_interval_sec**: basic/research/debug and RSS/CPU
  sampling interval. Stage transitions are timestamped separately.
- **output.keep_proofs**: retain bulky proof artifacts after success; logs and
  measurements remain regardless.

## Deployment

Start the coordinator and worker independently using their existing Python
module entry points and Hydra overrides with canonical keys, e.g.
`coordinator.host=0.0.0.0` on the coordinator and the coordinator address on
workers. All workers must see the same shared filesystem paths. Use explicit
matching prover thread counts for controlled distributed experiments.
Cache and request roots must not overlap. `resource_model.path: null` stores
history alongside partition artifacts.

Verification is mandatory in the supported configuration. The optional
trusted VK manifest pins setup identities to trusted verification key digests;
without it workers generating verification keys are trusted. Legacy low-level
storage and verification helpers remain for programmatic compatibility, but
independent witness preparation and S3 are no longer exposed by the launchers.

An accepted submission returns its request ID before preparation completes.
Poll until completed or failed. Interrupted preparation is failed on restart.
Preparation queue wait, first/all-job-ready times and overlap are reported
separately. Overlapping activities are not additive.

See comments in the two YAML files for remaining deployment settings.

## Shared artifact lifetime

The flat partition directory holds one canonical circuit and one compatible
key bundle. Published groups are write-once; metadata is committed last under
a filesystem lock. Workers read models, prepared descriptors, settings,
compiled circuits, witnesses and compatible PK/VK directly. New artifacts
are staged privately on shared storage, then renamed into the cache if the
slot remains available. Incompatible or disabled reuse generates request-owned
artifacts without replacing the canonical cache. Fully cold runs keep model
and circuit artifacts request-owned too; resource-history flags remain independent.
Existing key metadata without the SRS digest is not reused; reset stale caches
only when no requests are active. All nodes must share the same SRS directory.
Do not edit, replace or remove published files while requests are active.
Filesystem output publication uses hard links where possible, with a cross-device
copy only when needed. Optional debug retention and verifier snapshots can
still copy artifacts. Direct references do not eliminate native deserialization
or NFS reads.

### Proving-key representation

`proving.key_representation: full | reduced | compact` selects the persistent format of newly generated keys. The default is `full`. The node passes this setting to each prover subprocess as `EZKL_PK_REPRESENTATION`. Updated EZKL/Halo2 bindings automatically detect the stored representation when reading. Existing cached keys are reused irrespective of the requested representation; to compare formats set `cache.keys: false` (or use separate cache roots). This setting does not change the in-memory proving algorithm or verification keys. Results record the requested representation, not necessarily the format of a cache hit.

### Proving-key profiling

In research/debug mode, `halo2_key_metrics.csv` records component serialization, deserialization (logical bytes consumed), reconstruction, actual loaded representation and file-level materialization and serialization totals. `job_report.csv` summarizes these as `pk_*` fields and distinguishes `requested_key_representation` from the actual `pk_representation`. Materialization includes deserialization and reconstruction; do not add these nested timings to the total. Serialization includes buffered flush, not an fsync. Missing measurements are blank, not zero. Cached keys have no new serialization time. Updated native bindings are required.

### Targeted PK page-cache eviction

Set `proving.evict_pk_page_cache: true` to flush the selected PK with `fsync` and request Linux `POSIX_FADV_DONTNEED` before the native proving call. Default false leaves caching unchanged. No global cache drops or root permissions are needed on ordinary Linux filesystems. `fincore` (if installed and supported) measures residency before/after; missing residency is blank. A successful advisory request does not establish a cold read. Failures to flush or advise fail the opted-in job rather than silently measuring a different condition. Shared files/hard links refer to the same inode and concurrent readers can repopulate it; use isolated trials. The eviction/probe durations are separately reported; they are included in the outer `ezkl_proof_time(s)` but not native `pk_materialization_time(s)`. This controls only client-side PK page cache, not SRS, filesystem server, storage device or remote caches.

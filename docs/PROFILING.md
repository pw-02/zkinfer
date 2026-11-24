# Research profiling (ccgrid)

The default mode is `research`. Set `ZKINFER_PROFILING_MODE` in the
environment of **every prover node** before starting the supervisor.

| Mode | Runtime RSS and stages | Halo2 phase events | FFT/MSM files and legacy CPU report | Admission snapshot |
| --- | --- | --- | --- | --- |
| basic | yes | no | no | no |
| research (default) | yes | yes | no | no |
| debug | yes | yes | yes | yes |

Rebuild the custom EZKL wheel against halo2/ccgrid for these backend modes.
The environment variable controls output; it does not yet eliminate all
internal timing/CPU instrumentation overhead.

Keep the original per-job event streams for compatibility with the runtime
and older analysis scripts. The new execution_trace.csv is a chronological
view across those streams; it contains overlapping hierarchical events, not
additive durations. circuit_features.json preserves the returned EZKL
settings and Halo2 circuit statistics. halo2_phase_resources.csv attributes
sampled process RSS to timestamped phases. Missing samples are not zero.
The original job/request reports and transfer reports remain compatible.

After collecting the job directories from all nodes into one local run root,
execute:

    python -m zkinfer.profiling.collect_reports /path/to/run

This writes reports/phase_report.csv with one row per phase occurrence and
a relative job-directory identifier. It also refreshes unified per-job
traces and phase summaries. No shared-storage or coordinator aggregation is
performed by this command.

Sampling remains configured via node.resource_sampling_interval_sec.
For isolated warm profiling, start with 0.05 or 0.1 seconds; measure
instrumentation overhead before choosing an interval for serving experiments.

The runtime still needs the live status file and major transition events.
Do not remove them during execution. No admission logic is changed here.

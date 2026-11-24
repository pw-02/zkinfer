# Burst-latency figure

The included example uses invented values solely to demonstrate the figure layout.
It is not a prediction of zkInfer performance. Do not use these values in the paper.

Install matplotlib and numpy, then run:

```bash
python plot_bursts.py --demo --output illustrative_bursts
python plot_bursts.py --csv measured_results.csv --output burst_latency
```

Measured CSV columns:

`workload,policy,requests,trial,median_latency_min,completed_requests`

Use one row per complete trial. Workloads: MNIST GAN, MobileNetV2, NanoGPT (4L).
Policies: Fixed-k, Whole-job, zkInfer. For each trial, calculate the median of the
individual request latencies (submission to completion) in minutes. Do not put
batch makespan in this field. Points show the mean of trial medians; error bars
show their min/max range. They are not confidence intervals or request percentiles.

Use identical request-count points and paired trial IDs across policies. Use at
least three independent trials for the proposed protocol. With a single trial,
the script produces zero-width error bars; this is not evidence of low variation.
Choose fixed-k in a separate pilot and hold it fixed through the load sweep.
Missing policy points raise an error, and incomplete batches are rejected rather
than silently showing successful-request-only latency. If failures occur, report
them and add explicit failed-run annotations instead of dropping those runs.

Figure: 7.1 inches wide, three panels, serif fonts embedded in PDF, linear x-axis,
zero-based independent y-axis ranges, shared legend, distinct colors/markers/lines.
Independent y ranges preserve within-model comparisons; do not compare slopes
across panels without reading the units. All panels use minutes.

Warm setup means compiled circuits and proving keys are available in local files;
this does not mean request results or witnesses are cached. State a 256 GiB
admission budget in the caption. Use separate files for cold runs.

Main metric: request latency. Keep batch makespan as a separate supporting figure
with identical panel/style conventions. Do not call request count 'active requests'
throughout a burst: the number of active requests decreases as work completes.

# Partition-granularity figure template

All values in illustrative_granularity.csv/PDF/PNG/SVG are INVENTED LAYOUT DATA.
They are not measured results, simulations, forecasts, or reused legacy data.
The different curve shapes are examples only; do not interpret them as findings.

The figure is 7.1 inches wide for a two-column paper, with serif fonts,
vector PDF/SVG output and a 300-dpi PNG preview. It matches the blue zkInfer
style used in the earlier request-burst example. Three workload columns and
two request-count rows keep the single-request and batch results separate.
Each panel uses its own zero-based y scale because workloads differ in duration.
The ordered x positions represent the doubling choices 1, 2, 4 and 8.

## Reproduce the preview

```bash
python -m pip install matplotlib numpy
python plot_granularity.py --demo --output illustrative_granularity
```

## Use real results

Copy results_template.csv to measured_results.csv and fill in every trial row.
Times are in seconds; the plot converts them to minutes. For N=1 use end-to-end
request latency. For N=16 use batch makespan from submission to the final request
completion, NOT mean/median request latency. Source these from traffic summaries
and per-request records with consistent timing boundaries across configurations.

```bash
python plot_granularity.py --csv measured_results.csv --output partition_granularity
```

Columns:
- workload: MNIST GAN, MobileNetV2, or NanoGPT (4L)
- g: maximum operators per partition, 1/2/4/8
- requests: 1 or 16
- trial: trial identifier (one row per completed experiment)
- completion_time_s: request latency for N=1, batch makespan for N=16
- completed_requests: must match requests
- data_kind: measured for actual results; illustrative for any dummy data

The script takes the median across trials and plots observed minimum–maximum
ranges (not confidence intervals). A single observation gets no error bar.
Missing cells, duplicate trials, invalid timings, and incomplete batches are
rejected. Report failed/OOM/timed-out trials separately; do not quietly omit
failures and describe the remainder as universally successful configurations.
Do not mix policies, thread settings, memory budgets or setup conditions in this
CSV. The intended configuration is zkInfer, 256 GiB, warm setup, g=1/2/4/8.

Illustrative CSV input automatically retains the visible dummy-data label.
Only use data_kind=measured for actual measurements. Replace the caption's
three-trial statement if you collect a different number of repetitions.

figure.tex includes both a safe dummy-data insertion and a commented measured-data
version. Copy the selected PDF into figures/evaluation/ in your paper project.
The figure requires graphicx. No changes to running evaluation scripts are needed.

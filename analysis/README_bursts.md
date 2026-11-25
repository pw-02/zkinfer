# Request-burst plots

This retained analysis utility reads archived campaign outputs; it does not launch experiments. Run from the repository root with NumPy and Matplotlib installed:

```bash
python analysis/plot_bursts.py \
  --sources /path/to/archived-campaign-a /path/to/archived-campaign-b \
  --models bert mobilenet_v2 nano_gpt_4_layers_64_embd \
  --output figures/evaluation/burst_serving
```

```bash
python analysis/plot_bursts.py \
  --sources /path/to/archived-campaign-a /path/to/archived-campaign-b \
  --models bert mobilenet_v2 nano_gpt_4_layers_64_embd \
  --output figures/evaluation/burst_serving
```


You can instead pass individual completed cells (shell globs are expanded by Bash):

```bash
python analysis/plot_bursts.py \
  --sources /path/to/archived-campaign-a/runs/burst-*-b128-r1 \
            /path/to/archived-campaign-b/runs/burst-bert-*-b128-r1 \
  --output figures/evaluation/burst_serving_b128
```

```bash
python analysis/plot_bursts.py \
  --sources /path/to/archived-campaign-a/runs/burst-*-b128-r1 \
            /path/to/archived-campaign-b/runs/burst-bert-*-b128-r1 \
  --output figures/evaluation/burst_serving_b128
```

Replace the Bert campaign path with the actual directory. Supply matching
configurations. If an older campaign has multiple budgets, select cells explicitly
or filter with `--reserve-gib` (physical host memory minus the admission budget,
not the admission budget itself). Warm setup is the default; use `--setup cold`
for a separate cold figure. `--models` accepts YAML stems or display names.
Omit it to plot every available model, including MNIST GAN.

The output prefix produces PDF, PNG and an audit CSV containing source paths,
settings and one row per trial. Figures contain a column per available model,
median latency above and burst makespan below. Colors match the requested example.

Campaign roots use only `runs/*/DONE.json`'s chosen attempt. Explicit attempts and
`run/` paths are accepted if their `run_status.json` and traffic summary report
complete success. Profiles, Poisson trials, mixed-model trials and failed/partial
trials are not included. Overlapping input paths are deduplicated. Repeated
model/policy/request-count/seed combinations are rejected to avoid counting the
same experiment twice. Different seeds are treated as separate trials.

Latency is the median of request latencies within a trial, including client
submission and completion observation. Makespan comes directly from the load
generator's `batch_makespan_s`. With multiple trials, points average the
trial-level medians/makespans and error bars show min/max across trials, not a
confidence interval. A single trial provides no trial-variability error bar.
Missing cells are reported and left as gaps; they are never zero-filled.

The script checks matching setup, reserve, partitioning, simplification,
verification, sampling, cache capacity, history percentile, coordinator policy
and thread settings within each model. It records Git revision in the audit CSV;
this does not replace checking hardware and initial profile compatibility when
combining campaigns.

Validation:

```bash
python -m unittest discover -s analysis -p test_plot_bursts.py -v
```

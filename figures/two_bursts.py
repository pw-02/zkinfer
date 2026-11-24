
#!/usr/bin/env python3
"""Plot raw request measurements for Whole-job and zkInfer only.

python plot_bursts_two_systems.py --demo
python plot_bursts_two_systems.py                  # edit RAW_DATA first
python plot_bursts_two_systems.py --json raw.json
python plot_bursts_two_systems.py --csv requests.csv

JSON uses the same structure as RAW_DATA (burst-size keys are strings in JSON).
CSV columns:
workload,policy,requests,trial,request_id,latency_s,makespan_s
One CSV row per request; repeat the trial's makespan_s on all its rows.
Only enter successful COMPLETE burst trials from matching configurations.
Units are SECONDS at input, MINUTES in the figure.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

LABELS = {
    "bert": "TinyBERT", "tiny_bert": "TinyBERT", "mnist_gan": "MNIST GAN",
    "mobilenet_v2": "MobileNetV2", "nano_gpt_4_layers_64_embd": "NanoGPT (4L)",
}
POLICIES = {"fixed-8": "Fixed(k=8)", "whole-job": "Whole-job", "stage": "zkInfer"}
PLOT_POLICIES = ["Whole-job", "zkInfer"]
COLORS = {"Fixed(k=8)": "#B56B15", "Whole-job": "#606060", "zkInfer": "#1768AC"}
MARKERS = {"Fixed(k=8)": "s", "Whole-job": "^", "zkInfer": "o"}
LINES = {"Fixed(k=8)": "--", "Whole-job": "-.", "zkInfer": "-"}

# EDIT HERE: model -> system -> burst size -> list of independent trials.
# For each trial, provide exactly N individual request latencies in seconds.
# makespan_s is elapsed time from first arrival to last completion.
# Burst size represents concurrency only when requests launch concurrently.
# The empty lists are placeholders, NOT zero latency measurements.
RAW_DATA = {
    "TinyBERT": {
        "Whole-job": {1: [], 4: [], 8: [], 16: []},
        "zkInfer": {1: [], 4: [], 8: [], 16: []},
    },
    "MobileNetV2": {
        "Whole-job": {1: [], 4: [], 8: [], 16: []},
        "zkInfer": {1: [], 4: [], 8: [], 16: []},
    },
    "NanoGPT (4L)": {
        "Whole-job": {1: [], 4: [], 8: [], 16: []},
        "zkInfer": {1: [], 4: [], 8: [], 16: []},
    },
}
# Example cell, illustrative only:
# RAW_DATA["TinyBERT"]["Whole-job"][4] = [
#     {"trial": "r1", "latencies_s": [960, 1020, 1080, 1140], "makespan_s": 1200},
#     {"trial": "r2", "latencies_s": [990, 1050, 1110, 1170], "makespan_s": 1230},
# ]


def model_label(value):
    return LABELS.get(value, value)


def policy_label(value):
    aliases = {**POLICIES, "Fixed (k=8)": "Fixed(k=8)"}
    label = aliases.get(value, value)
    if label not in POLICIES.values():
        raise ValueError(f"Unknown policy: {value}")
    return label


def positive_int(value):
    number = int(value)
    if number <= 0 or float(value) != number:
        raise ValueError(f"Expected a positive integer: {value}")
    return number


def flatten(data):
    for model, systems in data.items():
        for policy, bursts in systems.items():
            for n, trials in bursts.items():
                if not isinstance(trials, list):
                    raise ValueError(f"Expected a list of trials: {model}/{policy}/{n}")
                for trial in trials:
                    yield dict(trial, workload=model, policy=policy, requests=n)


def read_csv(path):
    groups = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"workload", "policy", "requests", "trial", "request_id",
                    "latency_s", "makespan_s"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV needs columns: {sorted(required)}")
        for r in reader:
            key = (model_label(r["workload"]), policy_label(r["policy"]),
                   positive_int(r["requests"]), r["trial"].strip())
            makespan = float(r["makespan_s"])
            if key not in groups:
                groups[key] = dict(workload=key[0], policy=key[1], requests=key[2],
                                   trial=key[3], latencies_s=[], makespan_s=makespan,
                                   request_ids=set())
            group = groups[key]
            request_id = r["request_id"].strip()
            if not request_id or request_id in group["request_ids"]:
                raise ValueError(f"Missing/duplicate request_id for {key}: {request_id}")
            if makespan != group["makespan_s"]:
                raise ValueError(f"Conflicting makespan_s values for {key}")
            group["request_ids"].add(request_id)
            group["latencies_s"].append(float(r["latency_s"]))
    return list(groups.values())


def summarize(trials):
    rows, seen = [], set()
    for t in trials:
        model, policy = model_label(t["workload"]), policy_label(t["policy"])
        n, trial = positive_int(t["requests"]), str(t["trial"]).strip()
        key = (model, policy, n, trial)
        if not trial or key in seen:
            raise ValueError(f"Missing/duplicate trial ID: {key}")
        seen.add(key)
        values = np.asarray(t["latencies_s"], dtype=float)
        if values.ndim != 1 or len(values) != n:
            raise ValueError(f"{key}: expected exactly {n} request latencies")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{key}: latencies must be finite and nonnegative")
        makespan = float(t["makespan_s"])
        if not math.isfinite(makespan) or makespan <= 0:
            raise ValueError(f"{key}: makespan_s must be finite and positive")
        if makespan + 1e-9 < float(np.max(values)):
            raise ValueError(f"{key}: batch makespan cannot be shorter than a request latency")
        rows.append(dict(workload=model, policy=policy, requests=n, trial=trial,
                         median_latency_min=float(np.median(values)) / 60,
                         makespan_min=makespan / 60, completed_requests=n))
    return rows


def demo_trials():
    # Synthetic requests arrive simultaneously. Only for this demo, makespan
    # therefore equals the largest request latency. These are NOT measurements.
    bases = {"TinyBERT": [8, 20, 36, 68],
             "MobileNetV2": [90, 230, 410, 780],
             "NanoGPT (4L)": [3, 7, 12, 22]}
    for model, medians in bases.items():
        for policy, scale in zip(POLICIES.values(), [1, .88, .77]):
            for n, median_min in zip([1, 4, 8, 16], medians):
                for trial, factor in enumerate([.97, 1, 1.03], 1):
                    spread = np.linspace(.7, 1.3, n) if n > 1 else np.ones(1)
                    values = (spread * median_min * 60 * scale * factor).tolist()
                    yield dict(workload=model, policy=policy, requests=n,
                               trial=str(trial), latencies_s=values,
                               makespan_s=max(values))


def plot(rows, models, bursts, output, demo=False):
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8,
        'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.fontsize': 8,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    groups = {}
    for row in rows:
        for metric in ('median_latency_min', 'makespan_min'):
            groups.setdefault((row['workload'], row['policy'], row['requests'], metric), []).append(row[metric])
    fig, axes = plt.subplots(2, len(models), figsize=(2.35 * len(models), 3.2),
                             sharex=True, squeeze=False)
    x = np.arange(len(bursts))
    any_repeated = False
    for r, (metric, ylabel) in enumerate([
        ('median_latency_min', 'Median latency (min)'), ('makespan_min', 'Makespan (min)')
    ]):
        for c, model in enumerate(models):
            ax = axes[r, c]
            for policy in PLOT_POLICIES:
                means, lower, upper = [], [], []
                present = False
                for n in bursts:
                    vs = groups.get((model, policy, n, metric), [])
                    if not vs:
                        means.append(np.nan); lower.append(0); upper.append(0)
                        if r == 0:
                            print(f'Missing: {model}, {policy}, N={n}', file=sys.stderr)
                        continue
                    present = True
                    avg = float(np.mean(vs))
                    means.append(avg); lower.append(avg - min(vs)); upper.append(max(vs) - avg)
                    any_repeated |= len(vs) > 1
                if present:
                    ax.errorbar(x, means, yerr=[lower, upper], label=policy,
                                color=COLORS[policy], marker=MARKERS[policy],
                                linestyle=LINES[policy], linewidth=1.15, markersize=3.8,
                                capsize=2, elinewidth=.65, zorder=3)
            if r == 0:
                ax.set_title(model, pad=3)
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks(x, bursts)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis='y', color='#dddddd', linewidth=.45)
    handles = {}
    for ax in axes.flat:
        hs, ls = ax.get_legend_handles_labels()
        handles.update(zip(ls, hs))
    labels = [p for p in PLOT_POLICIES if p in handles]
    fig.legend([handles[p] for p in labels], labels, loc='upper center',
               bbox_to_anchor=(.52, .995), ncol=2, frameon=False, columnspacing=1.5)
    fig.supxlabel('Requests per burst', y=.035, fontsize=8)
    fig.subplots_adjust(left=.09 if len(models) == 1 else .08, right=.995,
                        bottom=.16, top=.84, wspace=.27, hspace=.12)
    if demo:
        fig.text(.5, -.025, "ILLUSTRATIVE DATA ONLY — NOT EXPERIMENTAL RESULTS",
                 ha="center", fontsize=8, color="#9B2525")
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        path = output.parent / (output.name + '.' + extension)
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.02)
        print(f'Wrote {path}')
    plt.close(fig)
    print('Points average trial-level metrics. ' +
          ('Error bars show the observed trial range.' if any_repeated else
           'One trial per point; no trial-variability error bars.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--json", type=Path)
    source.add_argument("--csv", type=Path)
    source.add_argument("--demo", action="store_true", default=True,)
    parser.add_argument("--output", type=Path, default=Path("burst_two_systems"))
    args = parser.parse_args()
    try:
        if args.demo:
            trials = demo_trials()
        elif args.json:
            trials = flatten(json.loads(args.json.read_text()))
        elif args.csv:
            trials = read_csv(args.csv)
        else:
            trials = flatten(RAW_DATA)
        rows = [row for row in summarize(trials) if row["policy"] in PLOT_POLICIES]
        if not rows:
            raise ValueError("No Whole-job or zkInfer trials supplied. Fill RAW_DATA, pass --json/--csv, or use --demo.")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.error(str(exc))
    order = ["TinyBERT", "MNIST GAN", "MobileNetV2", "NanoGPT (4L)"]
    available = {r["workload"] for r in rows}
    models = [m for m in order if m in available] + sorted(available - set(order))
    bursts = sorted({r["requests"] for r in rows})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(args.output) + ".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summarized {len(rows)} complete trials; wrote {args.output}.csv")
    plot(rows, models, bursts, args.output, demo=args.demo)


if __name__ == "__main__":
    main()

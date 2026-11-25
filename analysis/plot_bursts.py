#!/usr/bin/env python3
"""Plot completed burst trials from campaign roots or explicit run directories.

python analysis/plot_bursts.py --sources /path/to/archived-campaign-a /path/to/bert-campaign
python analysis/plot_bursts.py --sources /path/to/runs/burst-bert-warm-stage-n4-b128-r1

Campaign discovery follows runs/*/DONE.json, never unfinished attempts or profiles.
An explicitly supplied attempt/run directory can also be read if it completed.
Outputs include PDF, PNG and one audit CSV row per trial. Error bars show the
min/max range across trials, not confidence intervals. One trial has no error bar.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

LABELS = {
    'bert': 'TinyBERT', 'tiny_bert': 'TinyBERT',
    'mnist_gan': 'MNIST GAN', 'mobilenet_v2': 'MobileNetV2',
    'nano_gpt_4_layers_64_embd': 'NanoGPT (4L)',
}
POLICIES = {'fixed-8': 'Fixed(k=8)', 'whole-job': 'Whole-job', 'stage': 'zkInfer'}
COLORS = {'Fixed(k=8)': '#B56B15', 'Whole-job': '#606060', 'zkInfer': '#1768AC'}
MARKERS = {'Fixed(k=8)': 's', 'Whole-job': '^', 'zkInfer': 'o'}
LINES = {'Fixed(k=8)': '--', 'Whole-job': '-.', 'zkInfer': '-'}


def read_json(path):
    return json.loads(path.read_text())


def discover(source):
    """Only DONE-selected attempts when reading campaign/cell directories."""
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f'Not a directory: {source}')
    if (source / 'plan.json').is_file() and (source / 'traffic').is_dir():
        return [source]
    if (source / 'run/plan.json').is_file():
        return [source / 'run']
    if (source / 'DONE.json').is_file():
        marker = read_json(source / 'DONE.json')
        attempt = marker.get('attempt', '')
        if not attempt or Path(attempt).name != attempt:
            raise ValueError(f'Invalid completed-attempt marker: {source}')
        run = source / attempt / 'run'
        if not (run / 'plan.json').is_file():
            raise ValueError(f'Completed run is missing its plan: {run}')
        return [run]
    runs = source / 'runs' if (source / 'runs').is_dir() else source
    cells = sorted(p for p in runs.iterdir() if p.is_dir() and (p / 'DONE.json').is_file())
    result = []
    for cell in cells:
        # Profile preparation markers can exist under a broader output directory.
        signature = read_json(cell / 'DONE.json').get('signature', {})
        if signature.get('arguments', {}).get('action') == 'prepare':
            continue
        result.extend(discover(cell))
    if not result:
        print(f'No completed measured cells under {source}', file=sys.stderr)
    return result


def load_trial(run):
    plan = read_json(run / 'plan.json')
    args = plan['arguments']
    if args.get('arrival', 'burst') != 'burst':
        return None
    baseline = args.get('baseline', plan.get('baseline'))
    if baseline not in POLICIES:
        print(f'Skipping unsupported baseline {baseline}: {run}', file=sys.stderr)
        return None
    if args.get('allow_unprofiled') or args.get('bootstrap_setup'):
        return None
    workload = args['workload']
    if ',' in workload:
        print(f'Skipping mixed workload: {run}', file=sys.stderr)
        return None
    status = read_json(run / 'run_status.json')
    summary = read_json(run / 'traffic/summary.json')
    n = int(args['requests'])
    if (status.get('status') != 'completed' or summary.get('request_count') != n
            or summary.get('statuses') != {'completed': n}):
        print(f'Skipping incomplete/failed trial: {run}', file=sys.stderr)
        return None
    latency = summary.get('successful_request_latency_median_s')
    makespan = summary.get('batch_makespan_s')
    if any(v is None or not math.isfinite(float(v)) or float(v) <= 0
           for v in (latency, makespan)):
        raise ValueError(f'Missing or invalid latency/makespan: {run}')
    overrides = plan.get('runtime_overrides', {})
    return {
        'workload': LABELS.get(workload, workload),
        'workload_config': workload,
        'policy': POLICIES[baseline],
        'requests': n,
        'seed': args.get('seed'),
        'median_latency_min': float(latency) / 60,
        'makespan_min': float(makespan) / 60,
        'completed_requests': n,
        'setup': args.get('setup', 'cold'),
        'reserve_gib': args.get('reserve_gib', 8),
        'partition_size': args.get('partition_size'),
        'simplify_model': args.get('simplify_model', False),
        'verify': args.get('verify', False),
        'sampling_seconds': args.get('sampling_seconds'),
        'cache_gib': args.get('cache_gib'),
        'history_percentile': args.get('history_percentile', 0.9),
        'coordinator_policy': overrides.get('coordinator.scheduling_policy'),
        'threads': json.dumps(plan.get('thread_environment', {}), sort_keys=True),
        'git_sha': plan.get('git_sha', ''),
        'source': str(run),
    }


def validate_trials(rows):
    seen = set()
    signatures = {}
    fields = ('setup', 'reserve_gib', 'partition_size', 'simplify_model', 'verify',
              'sampling_seconds', 'cache_gib', 'history_percentile',
              'coordinator_policy', 'threads')
    for row in rows:
        key = (row['workload'], row['policy'], row['requests'], row['seed'])
        if key in seen:
            raise ValueError(f'Duplicate trial for {key}. Narrow --sources to the intended runs.')
        seen.add(key)
        signature = tuple(row[k] for k in fields)
        previous = signatures.setdefault(row['workload'], signature)
        if signature != previous:
            differing = [k for k, a, b in zip(fields, previous, signature) if a != b]
            raise ValueError(f"Incompatible configurations for {row['workload']}: {differing}. "
                             'Select matching runs (for example only b128 runs).')


def plot(rows, models, bursts, output):
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
            for policy in POLICIES.values():
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
    labels = [p for p in POLICIES.values() if p in handles]
    fig.legend([handles[p] for p in labels], labels, loc='upper center',
               bbox_to_anchor=(.52, .995), ncol=3, frameon=False, columnspacing=1.5)
    fig.supxlabel('Requests per burst', y=.035, fontsize=8)
    fig.subplots_adjust(left=.09 if len(models) == 1 else .08, right=.995,
                        bottom=.16, top=.84, wspace=.27, hspace=.12)
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sources', type=Path, nargs='+', required=True,
                        help='Campaign roots, runs directories, cells, attempts, or run directories')
    parser.add_argument('--models', nargs='+', help='Optional config names, e.g. bert mobilenet_v2')
    parser.add_argument('--bursts', nargs='+', type=int, default=[1, 4, 8, 16])
    parser.add_argument('--setup', choices=['warm', 'cold'], default='warm')
    parser.add_argument('--reserve-gib', type=float,
                        help='Optional reserve filter (host total minus desired admission budget)')
    parser.add_argument('--output', type=Path, default=Path('figures/evaluation/burst_serving'),
                        help='Output prefix without extension')
    args = parser.parse_args()
    wanted = [LABELS.get(m, m) for m in args.models] if args.models else None
    rows = []
    # Deduplicate overlapping paths, e.g. campaign root plus one of its cells.
    runs = sorted({run.resolve() for source in args.sources for run in discover(source)})
    for run in runs:
        row = load_trial(run)
        if row is None or row['setup'] != args.setup or row['requests'] not in args.bursts:
            continue
        if wanted and row['workload'] not in wanted:
            continue
        if args.reserve_gib is not None and abs(float(row['reserve_gib']) - args.reserve_gib) > .001:
            continue
        rows.append(row)
    if not rows:
        parser.error('No matching completed burst trials found.')
    validate_trials(rows)
    order = ['TinyBERT', 'MNIST GAN', 'MobileNetV2', 'NanoGPT (4L)']
    available = {r['workload'] for r in rows}
    models = [m for m in (wanted or order + sorted(available - set(order))) if m in available]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = args.output.parent / (args.output.name + '.csv')
    with audit.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'Read {len(rows)} completed trials; wrote audit table {audit}')
    plot(rows, models, args.bursts, args.output)


if __name__ == '__main__':
    main()

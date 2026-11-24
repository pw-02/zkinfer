"""Plot the mean of trial-level median request latencies, with min/max bars.

Fill HARD_CODED below, then: python plot_bursts.py
Demo: python plot_bursts.py --demo
CSV:  python plot_bursts.py --csv results.csv --output burst_latency

CSV columns: workload,policy,requests,trial,median_latency_min,completed_requests
All latencies must be in MINUTES.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

MODELS = ["TinyBERT", "MobileNetV2", "NanoGPT (4L)"]
POLICIES = ["Fixed (k=8)", "Whole-job", "zkInfer"]
BURST_SIZES = [1, 4, 8, 16]

STYLES = {
    "Fixed (k=8)": dict(color="#B56B15", marker="^", linestyle=":"),
    "Whole-job": dict(color="#606060", marker="s", linestyle="--"),
    "zkInfer": dict(color="#1768AC", marker="o", linestyle="-"),
}

# EDIT HERE. Structure: model -> system -> requests per burst -> trial medians.
# Every [] holds one median latency per independent, COMPLETE trial, in minutes.
# Example ONLY: 4: [17.2, 16.8, 17.5] means three trials at burst size 4.
# It does NOT mean three individual requests within one trial.
# Do not enter zero for missing data; fill every cell before running.
# For each trial, verify all N requests completed before adding its median.
# Burst size equals concurrency only if your experiment launches N requests
# concurrently. The code cannot infer the actual concurrency from latency.
HARD_CODED = {
    "TinyBERT": {                          # LEFT plot
        "Fixed (k=8)": {1: [4.679146488633341], 4: [], 8: [], 16: []},  # orange triangles
        "Whole-job":   {1: [], 4: [], 8: [], 16: []},  # gray squares
        "zkInfer":     {1: [], 4: [], 8: [], 16: []},  # blue circles
    },
    "MobileNetV2": {                      # MIDDLE plot
        "Fixed (k=8)": {1: [], 4: [], 8: [], 16: []},
        "Whole-job":   {1: [], 4: [], 8: [], 16: []},
        "zkInfer":     {1: [], 4: [], 8: [], 16: []},
    },
    "NanoGPT (4L)": {                     # RIGHT plot
        "Fixed (k=8)": {1: [], 4: [], 8: [], 16: []},
        "Whole-job":   {1: [], 4: [], 8: [], 16: []},
        "zkInfer":     {1: [], 4: [], 8: [], 16: []},
    },
}


def hardcoded_rows(data):
    """Convert explicitly labelled trial medians into the original row format."""
    if set(data) != set(MODELS):
        raise ValueError(f"Expected model keys: {MODELS}")
    rows = []
    for model in MODELS:
        if set(data[model]) != set(POLICIES):
            raise ValueError(f"{model}: expected system keys {POLICIES}")
        for policy in POLICIES:
            cells = data[model][policy]
            if set(cells) != set(BURST_SIZES):
                raise ValueError(f"{model}/{policy}: expected burst sizes {BURST_SIZES}")
            for n in BURST_SIZES:
                values = cells[n]
                if not isinstance(values, (list, tuple)) or not values:
                    raise ValueError(
                        f"Fill HARD_CODED[{model!r}][{policy!r}][{n}] "
                        "with a list of trial median latencies in minutes."
                    )
                for trial, value in enumerate(values, 1):
                    rows.append(dict(
                        workload=model, policy=policy, requests=n, trial=trial,
                        median_latency_min=value, completed_requests=n,
                    ))
    return rows


def demo_rows():
    # Illustrative data, never used by the default hard-coded mode.
    examples = {
        "TinyBERT": {
            "Fixed (k=8)": [8, 20, 36, 68],
            "Whole-job": [8, 17, 30, 56],
            "zkInfer": [8, 15, 25, 44],
        },
        "MobileNetV2": {
            "Fixed (k=8)": [90, 230, 410, 780],
            "Whole-job": [90, 205, 375, 710],
            "zkInfer": [91, 210, 365, 680],
        },
        "NanoGPT (4L)": {
            "Fixed (k=8)": [3, 7, 12, 22],
            "Whole-job": [3, 6, 10, 18],
            "zkInfer": [3, 5.8, 9.5, 17],
        },
    }
    data = {
        m: {p: {n: [v * f for f in (.97, 1, 1.03)]
                for n, v in zip(BURST_SIZES, examples[m][p])}
            for p in POLICIES}
        for m in MODELS
    }
    return hardcoded_rows(data)


def group_rows(rows):
    groups, seen = {}, set()
    for index, r in enumerate(rows, 1):
        try:
            m, p = r["workload"], r["policy"]
            n = int(r["requests"])
            trial = str(r["trial"]).strip()
            completed = int(r["completed_requests"])
            value = float(r["median_latency_min"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid data row {index}: {exc}") from exc
        if m not in MODELS or p not in POLICIES or n not in BURST_SIZES:
            raise ValueError(f"Unknown model/system/burst size: {m}/{p}/N={n}")
        if not trial:
            raise ValueError(f"Missing trial ID in data row {index}")
        key = (m, p, n, trial)
        if key in seen:
            raise ValueError(f"Duplicate trial: {key}")
        seen.add(key)
        if completed != n:
            raise ValueError(f"Incomplete batch: {m}/{p}/N={n}/trial={trial}")
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid latency at {key}: {value}")
        groups.setdefault((m, p, n), []).append(value)
    missing = [(m, p, n) for m in MODELS for p in POLICIES
               for n in BURST_SIZES if (m, p, n) not in groups]
    if missing:
        raise ValueError("Missing data for:\n" + "\n".join(
            f"  {m} / {p} / N={n}" for m, p, n in missing))
    return groups


def plot(groups, output, demo=False):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .65,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3))
    xpos = np.arange(len(BURST_SIZES))
    for ax, model in zip(axes, MODELS):
        for policy in POLICIES:
            values = [groups[(model, policy, n)] for n in BURST_SIZES]
            means = np.array([np.mean(v) for v in values])
            low = means - np.array([min(v) for v in values])
            high = np.array([max(v) for v in values]) - means
            ax.errorbar(
                xpos, means, yerr=[low, high], **STYLES[policy],
                linewidth=1.2, markersize=4, capsize=2, elinewidth=.7,
                markerfacecolor="white", markeredgewidth=1, zorder=3,
            )
        ax.set_title(model)
        ax.set_xlabel("Requests per burst")
        ax.set_xticks(xpos)
        ax.set_xticklabels(BURST_SIZES)
        ax.set_xlim(-.25, len(BURST_SIZES) - .75)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y", color="#dddddd", linewidth=.5, zorder=0)
    axes[0].set_ylabel("Median request latency (min)")
    handles = [Line2D([0], [0], **STYLES[p], label=p, markersize=4,
                      markerfacecolor="white", linewidth=1.2) for p in POLICIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, 1),
               ncol=3, frameon=False, handlelength=2.6, columnspacing=2)
    fig.subplots_adjust(left=.09, right=.99, bottom=.20, top=.76, wspace=.32)
    if demo:
        fig.text(.5, .025, "ILLUSTRATIVE DATA ONLY — NOT EXPERIMENTAL RESULTS",
                 ha="center", fontsize=8, color="#9B2525")
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = output.with_suffix("." + ext)
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print("Wrote", out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--csv", type=Path, help="Read CSV instead of HARD_CODED")
    src.add_argument("--demo", action="store_true", help="Use illustrative data")
    ap.add_argument("--output", type=Path, default=Path("burst_latency_line"))
    args = ap.parse_args()
    try:
        if args.demo:
            rows = demo_rows()
        elif args.csv is not None:
            with args.csv.open(newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
        else:
            rows = hardcoded_rows(HARD_CODED)
        groups = group_rows(rows)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    plot(groups, args.output, demo=args.demo)
    if args.demo:
        with args.output.with_suffix(".csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

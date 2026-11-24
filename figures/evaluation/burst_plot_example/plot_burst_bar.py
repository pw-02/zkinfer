"""Plot request-burst latency and makespan.

Demo:
    python plot_bursts.py --demo

Custom:
    python plot_bursts.py --csv results.csv --output burst_serving
"""

import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

MODELS = ["TinyBERT", "MobileNetV2", "NanoGPT (4L)"]
POLICIES = ["Fixed(k=8)", "Whole-job", "zkInfer"]
BURSTS = [1, 4, 8, 16]

COLORS = {"Fixed(k=8)":"#B56B15", "Whole-job":"#606060", "zkInfer":"#1768AC"}
MARKERS = {"Fixed(k=8)":"s", "Whole-job":"^", "zkInfer":"o"}
LINES = {"Fixed(k=8)":"--", "Whole-job":"-.", "zkInfer":"-"}


def demo_rows():
    ex = {
        "TinyBERT": [
            ([8,20,36,68], [8,29,54,101]),
            ([8,17,30,56], [8,24,44,80]),
            ([8,15,25,44], [8,20,34,59]),
        ],
        "MobileNetV2": [
            ([90,230,410,780], [90,320,590,1100]),
            ([90,205,375,710], [90,280,510,950]),
            ([90,180,300,520], [90,230,390,690]),
        ],
        "NanoGPT (4L)": [
            ([3,7,12,22], [3,10,18,33]),
            ([3,6,10,18], [3,8,14,25]),
            ([3,5.8,9.5,15], [3,7,12,20]),
        ],
    }

    return [
        dict(workload=m, policy=p, requests=n, trial=t,
             median_latency_min=lat*f, makespan_min=mk*f,
             completed_requests=n)
        for m in MODELS
        for p, (lats, mks) in zip(POLICIES, ex[m])
        for n, lat, mk in zip(BURSTS, lats, mks)
        for t, f in enumerate([.97, 1, 1.03], 1)
    ]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--csv", type=Path,
                     default=Path("figures/evaluation/burst_plot_example/illustrative_bursts.csv"))
    src.add_argument("--demo", action="store_true")
    ap.add_argument("--output", type=Path,
                    default=Path("figures/evaluation/burst_plot_example/burst_serving"))
    args = ap.parse_args()

    if args.demo:
        rows = demo_rows()
    else:
        with args.csv.open(newline="") as f:
            rows = list(csv.DictReader(f))

    groups = {}
    for r in rows:
        m, p, n = r["workload"], r["policy"], int(r["requests"])
        if int(r["completed_requests"]) != n:
            continue
        for metric in ["median_latency_min", "makespan_min"]:
            groups.setdefault((m,p,n,metric), []).append(float(r[metric]))

    plt.rcParams.update({
        "font.family":"serif", "font.size":8,
        "axes.titlesize":9, "axes.labelsize":8,
        "legend.fontsize":8, "xtick.labelsize":7.5, "ytick.labelsize":7.5,
        "axes.spines.top":False, "axes.spines.right":False,
        "axes.linewidth":.6, "pdf.fonttype":42, "ps.fonttype":42,
    })

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.2), sharex=True)
    metrics = [
        ("median_latency_min", "Median latency (min)"),
        ("makespan_min", "Makespan (min)")
    ]
    x = np.arange(len(BURSTS))

    for r, (metric, ylabel) in enumerate(metrics):
        for c, m in enumerate(MODELS):
            ax = axes[r,c]

            for p in POLICIES:
                means, low, high = [], [], []
                for n in BURSTS:
                    vs = groups[(m,p,n,metric)]
                    avg = np.mean(vs)
                    means.append(avg)
                    low.append(avg - min(vs))
                    high.append(max(vs) - avg)

                ax.errorbar(
                    x, means, yerr=[low,high],
                    label=p, color=COLORS[p],
                    marker=MARKERS[p], linestyle=LINES[p],
                    linewidth=1.15, markersize=3.8,
                    capsize=2, elinewidth=.65, zorder=3
                )

            if r == 0:
                ax.set_title(m, pad=3)
            if c == 0:
                ax.set_ylabel(ylabel)

            ax.set_xticks(x, BURSTS)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis="y", color="#dddddd", linewidth=.45, zorder=0)

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(.52,.995), ncol=3,
               frameon=False, columnspacing=1.5)

    fig.supxlabel("Requests per burst", y=.035, fontsize=8)
    fig.subplots_adjust(left=.08, right=.995, bottom=.16, top=.84,
                        wspace=.25, hspace=.12)

    if args.demo:
        fig.text(.995,.015, "ILLUSTRATIVE DATA ONLY",
                 ha="right", fontsize=6.5, color="#9B2525")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf","png"):
        out = args.output.with_suffix("." + ext)
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=.02)
        print("Wrote", out)

    if args.demo:
        with args.output.with_suffix(".csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    plt.close(fig)


if __name__ == "__main__":
    main()
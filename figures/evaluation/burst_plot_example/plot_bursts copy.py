"""Plot trial-level median request latency as line plots.

Default:
    python plot_bursts.py

Demo:
    python plot_bursts.py --demo

Custom:
    python plot_bursts.py --csv results.csv --output burst_latency
"""

import argparse, csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator



MODELS = ["TinyBERT", "MobileNetV2", "NanoGPT (4L)"]
POLICIES = ["Fixed (k=8)", "Whole-job", "zkInfer"]

STYLES = {
    "Fixed (k=8)": dict(color="#B56B15", marker="^", linestyle=":"),
    "Whole-job": dict(color="#606060", marker="s", linestyle="--"),
    "zkInfer": dict(color="#1768AC", marker="o", linestyle="-"),
}


def demo_rows():
    ex = {
        "TinyBERT": [
            [8, 20, 36, 68],
            [8, 17, 30, 56],
            [8, 15, 25, 44],
        ],
        "MobileNetV2": [
            [90, 230, 410, 780],
            [90, 205, 375, 710],
            [91, 210, 365, 680],
        ],
        "NanoGPT (4L)": [
            [3, 7, 12, 22],
            [3, 6, 10, 18],
            [3, 5.8, 9.5, 17],
        ],
    }

    return [
        dict(
            workload=m,
            policy=p,
            requests=n,
            trial=t,
            median_latency_min=v * f,
            completed_requests=n,
        )
        for m in MODELS
        for p, vals in zip(POLICIES, ex[m])
        for n, v in zip([1, 4, 8, 16], vals)
        for t, f in enumerate([.97, 1, 1.03], 1)
    ]


def main():
    ap = argparse.ArgumentParser()

    src = ap.add_mutually_exclusive_group()

    src.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "figures/evaluation/burst_plot_example/illustrative_bursts.csv"
        ),
    )

    src.add_argument(
        "--demo",
        action="store_true",
    )

    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "figures/evaluation/burst_plot_example/burst_latency_line"
        ),
    )

    args = ap.parse_args()

    if args.demo:
        rows = demo_rows()
    else:
        with args.csv.open(newline="") as f:
            rows = list(csv.DictReader(f))

    groups = {}
    seen = set()

    for r in rows:
        m = r["workload"]
        p = r["policy"]
        n = int(r["requests"])

        key = (m, p, n, str(r["trial"]))

        if key in seen:
            raise ValueError(f"Duplicate trial: {key}")

        seen.add(key)

        if m not in MODELS or p not in POLICIES:
            raise ValueError(
                f"Unknown workload/policy: {m}, {p}"
            )

        if int(r["completed_requests"]) != n:
            raise ValueError(
                f"Incomplete batch: {m}/{p}/N={n}"
            )

        v = float(r["median_latency_min"])

        if not np.isfinite(v) or v < 0:
            raise ValueError("Invalid latency")

        groups.setdefault((m, p, n), []).append(v)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 9,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": .65,

        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.1, 2.3),
    )

    for ax, m in zip(axes, MODELS):

        # IMPORTANT:
        # use a set so we only get 1, 4, 8, 16 once each
        xs = sorted({
            n
            for mm, _, n in groups
            if mm == m
        })

        # equally spaced positions: 0, 1, 2, 3
        xpos = np.arange(len(xs))

        for p in POLICIES:
            means = []
            low = []
            high = []

            for n in xs:
                vs = groups[(m, p, n)]
                avg = np.mean(vs)

                means.append(avg)
                low.append(avg - min(vs))
                high.append(max(vs) - avg)

            ax.errorbar(
                xpos,
                means,
                yerr=[low, high],
                **STYLES[p],
                linewidth=1.2,
                markersize=4,
                capsize=2,
                elinewidth=.7,
                markerfacecolor="white",
                markeredgewidth=1,
                zorder=3,
            )

        ax.set_title(m)
        ax.set_xlabel("Requests per burst")

        # exactly four ticks
        ax.set_xticks(xpos)

        # exactly: 1 4 8 16
        ax.set_xticklabels(xs)

        ax.set_xlim(-0.25, len(xs) - 0.75)
        ax.set_ylim(bottom=0)

        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5)
        )

        ax.grid(
            axis="y",
            color="#dddddd",
            linewidth=.5,
            zorder=0,
        )

    axes[0].set_ylabel(
        "Median request latency (min)"
    )

    handles = [
        Line2D(
            [0],
            [0],
            **STYLES[p],
            label=p,
            markersize=4,
            markerfacecolor="white",
            linewidth=1.2,
        )
        for p in POLICIES
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(.52, 1),
        ncol=3,
        frameon=False,
        handlelength=2.6,
        columnspacing=2,
    )

    fig.subplots_adjust(
        left=.09,
        right=.99,
        bottom=.20,
        top=.76,
        wspace=.32,
    )

    if args.demo:
        fig.text(
            .5,
            .025,
            "ILLUSTRATIVE DATA ONLY — NOT EXPERIMENTAL RESULTS",
            ha="center",
            fontsize=8,
            color="#9B2525",
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for ext in ("pdf", "png"):
        out = args.output.with_suffix("." + ext)

        fig.savefig(
            out,
            dpi=220,
            bbox_inches="tight",
        )

        print("Wrote", out)

    if args.demo:
        with args.output.with_suffix(".csv").open(
            "w",
            newline="",
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=rows[0].keys(),
            )

            w.writeheader()
            w.writerows(rows)

    plt.close(fig)


if __name__ == "__main__":
    main()
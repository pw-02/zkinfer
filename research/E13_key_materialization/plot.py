"""Generate readable, linear-axis E13 figures from measured trials."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from analyse import GROUP, HERE, MODES

COLORS = {"full": "#4477AA", "reduced": "#228833", "compact": "#CC6677"}
PARTS = (("deserialize_s", "Deserialization", "#4477AA"),
         ("reconstruct_s", "Reconstruction", "#EEAA44"),
         ("residual_s", "Other overhead", "#BBBBBB"))
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.labelsize": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#A0A0A0",
    "axes.labelcolor": "#253040", "text.color": "#253040",
    "xtick.color": "#253040", "ytick.color": "#253040",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "savefig.facecolor": "white",
})


def new_axes(title, ylabel):
    fig, ax = plt.subplots(figsize=(6.6, 4.3), layout="constrained")
    ax.set_title(title, loc="left", pad=14)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDE2E7", linewidth=0.8)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="both", length=0)
    ax.set_ylim(bottom=0)
    return fig, ax


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def labels(ax, modes, group, metric):
    counts = [group.loc[group.representation == m, metric].notna().sum() for m in modes]
    ax.set_xticks(range(len(modes)), [f"{m.capitalize()}\n(n={n})" for m, n in zip(modes, counts)])


def dots(ax, x, samples):
    jitter = np.linspace(-0.11, 0.11, len(samples)) if len(samples) > 1 else [0]
    ax.scatter(x + np.asarray(jitter), samples, s=25, facecolor="white",
               edgecolor="#263645", linewidth=0.9, zorder=4)


def bar(group, modes, out, metric, title, ylabel, name, savings=False):
    if not group[metric].notna().any():
        return
    fig, ax = new_axes(title, ylabel)
    full = group.loc[group.representation == "full", metric].mean()
    maximum = 0
    for x, mode in enumerate(modes):
        s = group.loc[group.representation == mode, metric].dropna().to_numpy()
        if not len(s):
            ax.text(x, 0, "N/A", ha="center", va="bottom")
            continue
        mean = s.mean()
        sd = s.std(ddof=1) if len(s) > 1 else 0
        ax.bar(x, mean, width=.58, color=COLORS[mode], alpha=.88,
               yerr=sd if len(s)>1 else None, capsize=4,
               error_kw={"ecolor": "#263645", "lw": 1.2})
        dots(ax, x, s)
        top = max(mean + sd, s.max())
        maximum = max(maximum, top)
        text = f"{mean:,.1f}"
        if savings and mode != "full" and np.isfinite(full) and full > 0:
            text += f"\n({100*(1-mean/full):.1f}% saved)"
        ax.annotate(text, (x, top), xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    labels(ax, modes, group, metric)
    ax.set_ylim(0, max(maximum * 1.30, 1e-6))
    ax.set_xlabel("Proving-key representation")
    save(fig, out, name)


def materialization(group, modes, out):
    complete = group.dropna(subset=[p[0] for p in PARTS] + ["materialize_s"])
    if complete.empty:
        return
    fig, ax = new_axes("Proving-key materialization", "Time (s)")
    maximum = 0
    for x, mode in enumerate(modes):
        subset = complete[complete.representation == mode]
        if subset.empty:
            ax.text(x, 0, "N/A", ha="center")
            continue
        bottom = 0
        for metric, label, color in PARTS:
            # Only tiny negative timer residuals can pass the analysis checks.
            height = max(0, subset[metric].mean()) if metric == "residual_s" else subset[metric].mean()
            ax.bar(x, height, bottom=bottom, width=.58, color=color,
                   label=label if mode == complete.representation.iloc[0] else None)
            bottom += height
        s = subset.materialize_s.to_numpy()
        dots(ax, x, s)
        mean, sd = s.mean(), s.std(ddof=1) if len(s)>1 else 0
        if len(s)>1:
            ax.errorbar(x, mean, yerr=sd, fmt="none", color="#263645", capsize=4, zorder=3)
        top = max(bottom, mean + sd, s.max())
        maximum = max(maximum, top)
        ax.annotate(f"{mean:,.1f} s", (x, top), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=10)
    labels(ax, modes, complete, "materialize_s")
    ax.set_ylim(0, max(maximum * 1.25, 1e-6))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=3, fontsize=9)
    save(fig, out, "pk_materialization")


def bandwidth_plot(group, modes, out, low, high):
    fig, ax = new_axes("Materialization vs. storage bandwidth", "Predicted time (s)")
    bandwidth = np.linspace(low, high, 500)
    predictions = {}
    for mode in modes:
        subset = group[group.representation == mode].dropna(subset=["size_gib", "reconstruct_s"])
        if subset.empty:
            continue
        size, cost = subset.size_gib.mean(), subset.reconstruct_s.mean()
        predictions[mode] = (size, cost)
        ax.plot(bandwidth, size / bandwidth + cost, color=COLORS[mode],
                lw=2.4, label=mode.capitalize())
    if not predictions:
        plt.close(fig)
        return
    # Mark only pairwise intersections on the lower envelope, i.e. actual
    # changes in the predicted best available representation.
    names = list(predictions)
    crossings = []
    for i, first in enumerate(names):
        for second in names[i+1:]:
            s1,c1 = predictions[first]
            s2,c2 = predictions[second]
            if c1 == c2:
                continue
            b = (s1-s2)/(c2-c1)
            if not low <= b <= high:
                continue
            t = s1/b+c1
            if t <= min(s/b+c for s,c in predictions.values()) + 1e-7:
                crossings.append((b,t))
    for b,t in sorted(set(crossings)):
        ax.axvline(b, color="#9AA3AE", ls="--", lw=.9)
        ax.scatter(b,t, color="#263645", s=24, zorder=5)
    if crossings:
        ax.text(.98,.70, "Predicted switch points:\n" + ", ".join(f"{b:.2f}" for b in sorted({b for b,t in crossings})) + " GiB/s",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox={"facecolor":"white","edgecolor":"none","alpha":.9})
    ax.set(xlim=(low,high), xlabel="Assumed effective loading bandwidth (GiB/s)")
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.legend(loc="upper right")
    save(fig, out, "bandwidth_sensitivity")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=HERE / "results")
    p.add_argument("--output", type=Path, default=HERE / "figures")
    p.add_argument("--min-bandwidth", type=float, default=.25)
    p.add_argument("--max-bandwidth", type=float, default=3.0)
    args = p.parse_args()
    if not 0 < args.min_bandwidth < args.max_bandwidth:
        p.error("Bandwidth range must satisfy 0 < min < max")
    data = pd.read_csv(args.input / "measurements.csv", keep_default_na=False)
    data = data[data.eligible.astype(str).str.lower() == "true"].copy()
    if data.empty:
        raise SystemExit("No eligible measurements; inspect results/validation.csv")
    for metric in ("size_gib", "serialize_s", "deserialize_s", "reconstruct_s",
                   "materialize_s", "residual_s", "peak_rss_gib", "prove_s", "request_s"):
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
    args.output.mkdir(parents=True, exist_ok=True)
    for i, (key, group) in enumerate(data.groupby(list(GROUP), dropna=False, sort=True), 1):
        out = args.output / f"group_{i:03d}"
        out.mkdir(exist_ok=True)
        (out / "conditions.txt").write_text("\n".join(f"{k}: {v}" for k,v in zip(GROUP,key)) + "\n")
        modes = [m for m in MODES if m in set(group.representation)]
        bar(group, modes, out, "size_gib", "Proving-key storage", "Serialized size (GiB)", "pk_size", savings=True)
        bar(group, modes, out, "serialize_s", "Proving-key serialization", "Time (s)", "pk_serialization")
        bar(group, modes, out, "peak_rss_gib", "Peak process memory", "Peak sampled RSS (GiB)", "peak_rss")
        bar(group, modes, out, "prove_s", "EZKL proving call", "Time (s)", "proof_time")
        bar(group, modes, out, "request_s", "End-to-end request latency", "Time (s)", "request_latency")
        materialization(group, modes, out)
        bandwidth_plot(group, modes, out, args.min_bandwidth, args.max_bandwidth)
    print(f"Generated PNG plots under {args.output}")


if __name__ == "__main__":
    main()

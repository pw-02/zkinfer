"""Generate E13 publication plots separately for each comparison group."""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyse import GROUP, HERE, MODES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=HERE / "results")
    p.add_argument("--output", type=Path, default=HERE / "figures")
    args = p.parse_args()
    data = pd.read_csv(args.input / "measurements.csv", keep_default_na=False)
    data = data[data.eligible.astype(str).str.lower() == "true"].copy()
    if data.empty:
        raise SystemExit("No eligible measurements; inspect results/validation.csv")
    for metric in ("size_gib", "serialize_s", "deserialize_s", "reconstruct_s",
                   "materialize_s", "residual_s", "peak_rss_gib"):
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
    args.output.mkdir(parents=True, exist_ok=True)
    for i, (key, group) in enumerate(data.groupby(list(GROUP), dropna=False, sort=True), 1):
        out = args.output / f"group_{i:03d}"
        out.mkdir(exist_ok=True)
        (out / "conditions.txt").write_text("\n".join(f"{k}: {v}" for k,v in zip(GROUP,key)) + "\n")
        modes = [m for m in MODES if m in set(group.representation)]
        x = np.arange(len(modes))
        def values(metric):
            return [group.loc[group.representation == m, metric].dropna().to_numpy() for m in modes]
        def save(fig, name):
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(out / f"{name}.{ext}", dpi=200)
            plt.close(fig)
        def bar(metric, ylabel, name):
            samples = values(metric)
            if not any(len(s) for s in samples):
                return
            fig, ax = plt.subplots(figsize=(4.2, 3))
            for j, s in enumerate(samples):
                if len(s):
                    ax.bar(j, np.mean(s), yerr=np.std(s, ddof=1) if len(s)>1 else None, capsize=3)
                    ax.scatter(np.full(len(s), j), s, color="black", s=12, zorder=3)
            ax.set_xticks(x, [m.capitalize() for m in modes])
            ax.set_ylabel(ylabel)
            save(fig, name)
        bar("size_gib", "PK size (GiB)", "pk_size")
        bar("serialize_s", "Serialization (s)", "pk_serialization")
        bar("peak_rss_gib", "Peak sampled RSS (GiB)", "peak_rss")
        fig, ax = plt.subplots(figsize=(4.2, 3))
        complete = group.dropna(subset=["deserialize_s", "reconstruct_s", "residual_s", "materialize_s"])
        bottom = np.zeros(len(modes))
        for metric, label in (("deserialize_s", "Deserialization"), ("reconstruct_s", "Reconstruction"), ("residual_s", "Residual overhead")):
            heights = np.array([complete.loc[complete.representation == m, metric].mean() for m in modes])
            heights = np.nan_to_num(heights)
            ax.bar(x, heights, bottom=bottom, label=label)
            bottom += heights
        for j, s in enumerate(values("materialize_s")):
            if len(s):
                ax.scatter(np.full(len(s), j), s, color="black", s=12, zorder=3)
        ax.set_xticks(x, [m.capitalize() for m in modes])
        ax.set_ylabel("Materialization (s)")
        ax.legend(fontsize=8)
        save(fig, "pk_materialization")
        fig, ax = plt.subplots(figsize=(4.5, 3))
        bandwidth = np.geomspace(0.05, 10, 200)
        for mode in modes:
            subset = group[group.representation == mode].dropna(subset=["size_gib","reconstruct_s"])
            if len(subset):
                ax.plot(bandwidth, subset.size_gib.mean()/bandwidth + subset.reconstruct_s.mean(), label=mode.capitalize())
        ax.set(xscale="log", yscale="log", xlabel="Assumed effective loading bandwidth (GiB/s)",
               ylabel="Predicted materialization (s)")
        ax.legend()
        save(fig, "bandwidth_sensitivity")
    print(f"Generated plots under {args.output}")


if __name__ == "__main__":
    main()

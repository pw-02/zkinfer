from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ops = [1, 2, 4, 8]
x = np.arange(len(ops))

data = {
    "TinyBERT": {
        "agg": [892.708904, np.nan, np.nan, np.nan],
        "mem": [51.0979, np.nan, np.nan, np.nan],
    },
    "MobileNetV2": {
        "agg": [5362, 6241, 6872, 7458],
        "mem": [45.3, 55.6, 75.9, 148.2],
    },
    "NanoGPT (4L)": {
        "agg": [775, 882, 930, 1425],
        "mem": [7.8, 13.2, 14.6, 25.7],
    },
}

colors = {
    "TinyBERT": "#606060",
    "MobileNetV2": "#B56B15",
    "NanoGPT (4L)": "#1768AC",
}

hatches = {
    "TinyBERT": "///",
    "MobileNetV2": "...",
    "NanoGPT (4L)": "",
}

out = Path("figures/evaluation/partition_granularity")
out.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3,
    "axes.linewidth": .6,
    "hatch.linewidth": .4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.7))

width = .23
models = list(data)

metrics = [
    ("agg", "Aggregate time (s)"),
    ("mem", "Peak memory (GiB)"),
]

for ax, (metric, ylabel) in zip(axes, metrics):
    for i, model in enumerate(models):
        ax.bar(
            x + (i - 1) * width,
            data[model][metric],
            width=width,
            label=model,
            color=colors[model],
            edgecolor="0.2",
            linewidth=.4,
            hatch=hatches[model],
            zorder=3,
        )

    ax.set_xticks(x, ops)
    ax.set_xlabel("Operators per job")
    ax.set_ylabel(ylabel, labelpad=2)
    ax.grid(axis="y", color="0.9", linewidth=.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, width=.5, pad=2)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(.5, 1.01),
    ncol=3,
    frameon=False,
    columnspacing=.9,
    handlelength=1.4,
    handletextpad=.4,
)

fig.subplots_adjust(
    left=.15,
    right=.99,
    bottom=.24,
    top=.73,
    wspace=.48,
)

for ext in ("pdf", "png"):
    path = out.with_suffix("." + ext)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.025)
    print("Saved", path)

plt.show()
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Raw data
# MobileNetV2 values are illustrative placeholders for now.
# Replace them when ready.
# ------------------------------------------------------------

data = {
    "TinyBERT": {
        "Cold": {"error": 0.4478, "under": 3 / 24 * 100},
        "Warm": {"error": 0.3552, "under": 7 / 24 * 100},
    },
    "MobileNetV2": {  # dummy values
        "Cold": {"error": 0.2100, "under": 5 / 24 * 100},
        "Warm": {"error": 0.1500, "under": 4 / 24 * 100},
    },
    "NanoGPT": {
        "Cold": {"error": 0.2931, "under": 7 / 24 * 100},
        "Warm": {"error": 0.0667, "under": 4 / 24 * 100},
    },
}

models = list(data.keys())
conditions = ["Cold", "Warm"]

colors = {
    "TinyBERT": "#606060",
    "MobileNetV2": "#B56B15",
    "NanoGPT": "#1768AC",
}
hatches = {
    "TinyBERT": "///",
    "MobileNetV2": "...",
    "NanoGPT": "",
}

out = Path("figures/evaluation/profile_accuracy_conditions")
out.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.6,
    "hatch.linewidth": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.55))
x = np.arange(len(conditions))
width = 0.22

metrics = [
    ("error", "Median abs. error (%)"),
    ("under", "Underestimation (%)"),
]

for ax, (metric, ylabel) in zip(axes, metrics):
    for i, model in enumerate(models):
        vals = [data[model][c][metric] for c in conditions]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=model,
            color=colors[model],
            edgecolor="0.2",
            linewidth=0.4,
            hatch=hatches[model],
            zorder=3,
        )

    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_xticks(x, conditions)
    ax.grid(axis="y", color="0.88", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2, width=0.5, pad=1.5)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    columnspacing=1.0,
    handlelength=1.4,
    handletextpad=0.45,
)

fig.subplots_adjust(left=0.16, right=0.99, bottom=0.22, top=0.73, wspace=0.42)

for ext in ("pdf", "png"):
    path = out.with_suffix("." + ext)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("Saved", path)

plt.show()
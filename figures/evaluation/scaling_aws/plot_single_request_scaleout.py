from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ------------------------------------------------------------
# Measured AWS results
# ------------------------------------------------------------

nodes = np.array([1, 2, 4, 6, 8, 10])

times_s = {
    "NanoGPT (4L)": [
        1108.401158,
        448.284717,
        226.143169,
        144.759470,
        110.780916,
        89.403294,
    ],
    "MobileNetV2": [
        9639.0,
        3405.093471,
        1713.609643,
        1154.744915,
        914.102607,
        761.712696,
    ],
    "TinyBERT": [
        7900.0,
        3120.0,
        1525.0,
        1100.0,
        870.0,
        750.0,
    ],
}

colors = ["#44799B", "#A47836", "#548365"]
hatches = ["", "///", ".."]

output_dir = Path("figures/evaluation")
output_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Style
# ------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Nimbus Roman",
        "DejaVu Serif",
    ],
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "hatch.linewidth": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ------------------------------------------------------------
# Validate data
# ------------------------------------------------------------

measurements = {}

for model, values in times_s.items():
    values = np.asarray(values, dtype=float)

    if (
        len(values) != len(nodes)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise ValueError(f"Invalid measurements for {model}")

    measurements[model] = values


# ------------------------------------------------------------
# Compact two-panel figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(3.35, 1.9),
)

x = np.arange(len(nodes))
width = 0.24

for i, ((model, values), color, hatch) in enumerate(
    zip(measurements.items(), colors, hatches)
):
    positions = x + (i - 1) * width

    common = {
        "width": width,
        "color": color,
        "edgecolor": "0.2",
        "linewidth": 0.4,
        "hatch": hatch,
        "label": model,
        "zorder": 3,
    }

    # Completion time in minutes.
    axes[0].bar(
        positions,
        values / 60.0,
        **common,
    )

    # Speedup relative to one prover node.
    axes[1].bar(
        positions,
        values[0] / values,
        **common,
    )


# Ideal linear speedup.
ideal_line, = axes[1].plot(
    x,
    nodes,
    color="0.25",
    linestyle=":",
    linewidth=1.0,
    label="Ideal",
    zorder=4,
)


# ------------------------------------------------------------
# Axes
# ------------------------------------------------------------

axes[0].set_ylabel("Completion time (min)", labelpad=2)
axes[1].set_ylabel("Speedup", labelpad=2)

axes[0].set_xlabel("(a) Prover nodes", labelpad=3)
axes[1].set_xlabel("(b) Prover nodes", labelpad=3)

maximum_time = max(v.max() / 60 for v in measurements.values())
maximum_speedup = max(
    (v[0] / v).max()
    for v in measurements.values()
)

axes[0].set_ylim(0, maximum_time * 1.08)
axes[1].set_ylim(
    0,
    max(float(nodes[-1]), maximum_speedup) * 1.08,
)

for ax in axes:
    ax.set_xticks(x, [str(n) for n in nodes])
    ax.set_xlim(-0.55, len(nodes) - 0.45)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.grid(
        axis="y",
        color="0.9",
        linewidth=0.4,
    )

    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    ax.tick_params(
        length=2,
        width=0.5,
        pad=2,
    )


# ------------------------------------------------------------
# Shared legend
# ------------------------------------------------------------

handles, labels = axes[0].get_legend_handles_labels()

legend_handles = [
    handles[0],
    handles[2],
    handles[1],
    ideal_line,
]

legend_labels = [
    labels[0],
    labels[2],
    labels[1],
    "Ideal linear",
]

fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=4,
    frameon=False,
    handlelength=1.5,
    columnspacing=1.2,
    handletextpad=0.5,
    labelspacing=0.4,
)


fig.subplots_adjust(
    # left=0.14,
    # right=0.99,
    bottom=0.25,
    top=0.83,
    wspace=0.38,
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

for extension in ("pdf", "png"):
    path = output_dir / f"single_request_scaleout.{extension}"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    print(f"Saved {path}")

plt.show()
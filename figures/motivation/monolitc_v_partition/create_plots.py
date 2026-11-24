import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.patches import Patch
from pathlib import Path

# Model, monolithic time, partitioned time, monolithic peak memory, partitioned peak memory.
data = [
    ('TinyBERT', 4539.025, 2697.275264, 279.6, 54.59),
    ('MobileNetV2', 25652.40, 21480.096852, 592.7, 50.27),
    ('NanoGPT\n(4L)', 8527.58, 3372.340305, 564.97, 11.52),
    ('NanoGPT\n(10L)', None, 8066.13, None, 12.01),
]
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'font.size': 7.5, 'axes.labelsize': 7.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.linewidth': .65,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
x = np.arange(len(data))
w = .32
colors = ['#AFC8DA', '#E8C99F']
edge = '#383838'
fig, axes = plt.subplots(2, 1, figsize=(3.35, 2.45), sharex=True)
fig.subplots_adjust(left=.19, right=.985, bottom=.15, top=.875, hspace=.22)

for ax, mi, pi, ylabel, log in [
    (axes[0], 1, 2, 'Request completion\ntime (s)', False),
    (axes[1], 3, 4, 'Peak memory (GiB)', True),
]:
    if log:
        ax.set_yscale('log')
        ax.set_ylim(1, 1800)
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_locator(NullLocator())
    else:
        ax.set_ylim(0, 34000)
        ax.set_yticks([0, 10000, 20000, 30000])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1000:g}k' if v else '0'))
    for idx, (col, off) in enumerate([(mi, -w/2), (pi, w/2)]):
        for i, row in enumerate(data):
            v = row[col]
            xi = x[i] + off
            if v is None:
                ax.text(xi, .06, 'OOM', transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=6.5, color='#963B3B', weight='bold')
                continue
            bottom = 1 if log else 0
            ax.bar(xi, v-bottom, w, bottom=bottom, color=colors[idx],
                   edgecolor=edge, linewidth=.55, zorder=3)
            label = (f'{v:.1f}' if v < 100 else f'{v:.0f}') if log else f'{v/1000:.1f}k'
            ax.annotate(label, (xi, v), xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=6.4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_xlim(-.52, 3.52)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linewidth=.4, color='#D7D7D7', linestyle='-')
    ax.tick_params(direction='out', length=2.5, width=.65, pad=2)
axes[0].tick_params(axis='x', bottom=False, labelbottom=False)
axes[1].set_xticks(x, [d[0] for d in data])
fig.align_ylabels(axes)
fig.legend([Patch(facecolor=c, edgecolor=edge, linewidth=.55) for c in colors],
           ['Monolithic', 'Partitioned (serial)'], loc='upper center',
           bbox_to_anchor=(.58, .995), ncol=2, frameon=False,
           columnspacing=1.2, handlelength=1.4, handletextpad=.5)

# Keep the physical canvas size fixed for predictable LaTeX scaling.
output_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
fig.savefig(output_dir / 'partitioning_refined.pdf', bbox_inches=None)
fig.savefig(output_dir / 'partitioning_refined.png', dpi=400, bbox_inches=None)
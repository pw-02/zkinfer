from pathlib import Path
import csv, math
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT

RUNS = {
    "TinyBERT": (Path("experiments/runs/2026-10-15_12-48-52_bert_admission-false-cold"), 121),
    "MobileNetV2": (Path("experiments/runs/2026-10-12_15-27-48_mobilenet_v2_admission-false"), 100),
    "NanoGPT (4L)": (Path("experiments/runs/2026-10-13_02-54-42_nano_gpt_4_layers_64_embd_admission-false"), 181),
}

TICKS = {
    "TinyBERT": ([0, 400, 800, 1200], [0, 15, 30, 45]),
    "MobileNetV2": ([0, 600, 1200, 1800], [0, 15, 30, 45]),
    "NanoGPT (4L)": ([0, 45, 90, 135], [0, 2, 4, 6]),
}

METRICS = [
    ("duration_s", "Duration (s)"),
    ("peak_memory_gib", "Peak memory (GiB)"),
]

TRACE = "#2E3A46"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": .65,
    "axes.edgecolor": ".28",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": .6,
    "ytick.major.width": .6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": .02,
})


def load_jobs(path, expected):
    reports = list(path.rglob("job_report.csv"))
    if len(reports) != 1:
        raise ValueError(f"Expected one report under {path}; found {len(reports)}")

    with reports[0].open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != expected:
        raise ValueError(f"Expected {expected} jobs; found {len(rows)}")
    if len({r["job_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate job IDs")

    jobs = []
    for r in rows:
        if r["job_status"] != "COMPLETED":
            raise ValueError(f"Job {r['job_id']} did not complete")

        d = float(r["job_runtime(s)"])
        m = float(r["max_process_memory(GB)"])

        if not all(math.isfinite(v) and v > 0 for v in (d, m)):
            raise ValueError(f"Invalid measurement for job {r['job_id']}")

        jobs.append({
            "job_id": r["job_id"],
            "duration_s": d,
            "peak_memory_gib": m,
        })

    return jobs


def export_csv(data):
    path = OUT / "job_heterogeneity.csv"

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "job_id",
                "duration_s",
                "peak_memory_gib",
            ],
        )
        writer.writeheader()

        for model, jobs in data.items():
            for job in jobs:
                writer.writerow({
                    "model": model,
                    "job_id": job["job_id"],
                    "duration_s": job["duration_s"],
                    "peak_memory_gib": job["peak_memory_gib"],
                })

    print(f"Saved {path}")


def ecdf(ax, values, ticks):
    x = np.sort(np.asarray(values))
    y = 100 * np.arange(1, len(x) + 1) / len(x)
    upper = max(ticks[-1] * 1.03, x[-1] * 1.04)

    ax.step(
        np.r_[0, x], np.r_[0, y],
        where="post",
        color=TRACE,
        lw=1.3,
        zorder=3,
    )

    ax.set(xlim=(0, upper), ylim=(0, 103))
    ax.set_xticks(ticks)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=".92", lw=.45)
    ax.tick_params(direction="out", length=2.5, width=.6, pad=2)
    ax.set_axisbelow(True)


def main():
    data = {m: load_jobs(p, n) for m, (p, n) in RUNS.items()}

    # Export the exact job-level data used by the plots.
    export_csv(data)

    fig = plt.figure(figsize=(7.15, 1.65))
    grid = fig.add_gridspec(
        1, 8,
        width_ratios=[1, 1, .22, 1, 1, .22, 1, 1],
        left=.07, right=.995, bottom=.245, top=.75, wspace=.22,
    )

    for g, (model, jobs) in enumerate(data.items()):
        axes = []

        for p, (metric, xlabel) in enumerate(METRICS):
            ax = fig.add_subplot(grid[0, g * 3 + p])
            axes.append(ax)

            ecdf(ax, [j[metric] for j in jobs], TICKS[model][p])
            ax.set_xlabel(xlabel, labelpad=3)

            if g == 0 and p == 0:
                ax.set_ylabel("Cumulative jobs (%)", labelpad=3)
            else:
                ax.tick_params(left=False, labelleft=False)
                ax.spines["left"].set_visible(False)

        center = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2

        fig.text(
            center, .89, model,
            ha="center", va="center",
            fontsize=8,
        )
        fig.text(
            center, .81, f"{len(jobs)} jobs",
            ha="center", va="center",
            fontsize=6.3, color=".45",
        )

    for ext in ("pdf", "png"):
        path = OUT / f"job_heterogeneity.{ext}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
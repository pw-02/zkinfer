from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
OUT = ROOT

RUNS = {
    "TinyBERT": Path("experiments/runs/2026-10-15_12-48-52_bert_admission-false-cold-0.25"),
    "MobileNetV2": Path("experiments/runs/2026-10-12_15-27-48_mobilenet_v2_admission-false"),
    "NanoGPT (4L)": Path("experiments/runs/2026-10-13_02-54-42_nano_gpt_4_layers_64_embd_admission-false"),
}

STAGES = [
    ("Preparation", "#EEF4F8"),
    ("Key generation", "#F6F1E8"),
    ("Proving", "#EEF5EF"),
]

TRACE = "#2E3A46"
PEAK = "0.45"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": .65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": .02,
})


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def first_time(rows, stage):
    return min(float(r["elapsed_s"]) for r in rows if r["stage"] == stage)


def load_job(job_dir):
    tr = read_csv(job_dir / "stage_transitions.csv")
    rs = read_csv(job_dir / "resource_usage.csv")

    b = [
        0.0,
        first_time(tr, "KEY_GEN"),
        first_time(tr, "PROVING"),
        first_time(tr, "PERSISTING_KEYS"),
    ]

    samples = [
        (float(r["elapsed_s"]), float(r["memory_gb"]))
        for r in rs
        if r["type"].strip().lower() == "process_tree"
        and r["memory_gb"]
        and 0 <= float(r["elapsed_s"]) <= b[-1]
    ]

    s = np.asarray(sorted(samples), dtype=float)

    return {
        "job_id": job_dir.name,
        "raw_samples": s,
        "samples": s.copy(),
        "boundaries": b,
        "peak": float(s[:, 1].max()),
    }


def select_job(run_dir):
    report = list(run_dir.rglob("job_report.csv"))[0]
    rows = read_csv(report)
    jobs_dir = report.parent.parent / "artifacts" / "jobs"

    jobs = [
        load_job(jobs_dir / r["job_id"])
        for r in rows
        if r["job_status"] == "COMPLETED"
    ]

    return max(jobs, key=lambda j: j["peak"])

def downsample_1s(samples):
    """Aggregate high-frequency samples into 1-second means."""
    bins = np.floor(samples[:, 0]).astype(int)

    return np.asarray([
        (
            samples[bins == sec, 0].mean(),
            samples[bins == sec, 1].mean(),
        )
        for sec in np.unique(bins)
    ], dtype=float)


def stage_at(t, b):
    if t < b[1]:
        return "Preparation"
    if t < b[2]:
        return "Key generation"
    return "Proving"

def smooth_trace(samples, window=5):
    x = samples[:, 0]
    y = samples[:, 1]

    kernel = np.ones(window) / window
    y_smooth = np.convolve(y, kernel, mode="same")

    return np.column_stack((x, y_smooth))

def save_samples(model, job, suffix, samples):
    name = model.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = OUT / f"{name}_{suffix}.csv"

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "job_id", "elapsed_s", "memory_gib", "stage"])

        for t, m in samples:
            w.writerow([
                model,
                job["job_id"],
                f"{t:.6f}",
                f"{m:.6f}",
                stage_at(t, job["boundaries"]),
            ])

    print(f"Saved {path}")


def plot_trace(ax, model, job):
    s, b, peak = job["samples"], job["boundaries"], job["peak"]

    for i, (_, color) in enumerate(STAGES):
        ax.axvspan(b[i], b[i + 1], color=color, lw=0)

    ax.plot(s[:, 0], s[:, 1], color=TRACE, lw=1.3)
    ax.axhline(peak, color=PEAK, lw=.75, ls=":")

    ax.text(
        .98, peak, f"{peak:.1f} GiB",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom",
        fontsize=6.2, color=".33",
    )

    ax.set_title(model, pad=5)
    ax.set_xlim(0, b[-1] * 1.02)
    ax.set_ylim(0, peak * 1.15)
    ax.set_xlabel("Elapsed time (s)", labelpad=3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.grid(axis="y", color=".92", lw=.45)
    ax.tick_params(direction="out", length=2.5, width=.6, pad=2)
    ax.set_axisbelow(True)


def main():
    jobs = {m: select_job(p) for m, p in RUNS.items()}

    # TinyBERT: 0.25 s -> 1 s, preserving the maximum in each second.
    jobs["TinyBERT"]["samples"] = downsample_1s(
        jobs["TinyBERT"]["raw_samples"]
    )
    jobs["TinyBERT"]["samples"] = smooth_trace(
    jobs["TinyBERT"]["samples"],
    window=45,
)

    OUT.mkdir(parents=True, exist_ok=True)

    for model, job in jobs.items():
        save_samples(model, job, "raw", job["raw_samples"])
        save_samples(model, job, "plot_samples", job["samples"])

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.05))

    for ax, (model, job) in zip(axes, jobs.items()):
        plot_trace(ax, model, job)

    axes[0].set_ylabel("Process memory (GiB)", labelpad=3)

    handles = [
        Patch(facecolor=c, edgecolor=".62", linewidth=.45, label=n)
        for n, c in STAGES
    ] + [
        Line2D([0], [0], color=TRACE, lw=1.3, label="Process memory"),
        Line2D([0], [0], color=PEAK, lw=.75, ls=":", label="Peak"),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(.5, .995),
        ncol=5,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.0,
    )

    fig.subplots_adjust(
        left=.07,
        right=.995,
        bottom=.21,
        top=.74,
        wspace=.28,
    )

    for ext in ("pdf", "png"):
        path = OUT / f"stage_memory.{ext}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
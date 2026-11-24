from pathlib import Path
import argparse
import csv
import math
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


MODEL_LABELS = {
    "mnist_gan": "MNIST GAN",
    "mobilenet_v2": "MobileNetV2",
    "nano_gpt_4_layers_64_embd": "NanoGPT (4L)",
}

MODEL_ORDER = list(MODEL_LABELS)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) and result >= 0 else None


def circuit_key(row):
    # Keep distinct circuits, setup conditions, and settings separate.
    return (
        row["model"],
        int(row["partition"]),
        row["condition"],
        row["setup_key"],
        row["settings_fingerprint"],
    )


def load_points(analysis_dirs, stage):
    repeatability = {}
    errors = defaultdict(list)

    for directory in analysis_dirs:
        repeatability_path = directory / "repeatability.csv"
        predictions_path = directory / "predictions.csv"

        for path in (repeatability_path, predictions_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        for row in read_csv(repeatability_path):
            if row["stage"] != stage:
                continue

            if row["model"] not in MODEL_LABELS:
                continue

            if row["condition"] not in {"cold", "warm"}:
                continue

            key = circuit_key(row)

            if key in repeatability:
                raise ValueError(
                    f"Duplicate circuit/condition in supplied results: {key}. "
                    "Select one campaign per model."
                )

            repeatability[key] = {
                "memory_cv": number(row["peak_memory_gib_cv"]),
                "measured_runs": int(
                    row["memory_measured_executions"]
                ),
            }

        for row in read_csv(predictions_path):
            if (
                row["stage"] != stage
                or row["metric"] != "peak_memory_gib"
            ):
                continue

            if row["model"] not in MODEL_LABELS:
                continue

            error = number(row["absolute_pct_error"])

            if error is not None:
                errors[circuit_key(row)].append(error)

    points = []

    for key, values in repeatability.items():
        model, partition, condition, setup_key, fingerprint = key

        points.append({
            "model": model,
            "partition": partition,
            "condition": condition,
            "setup_key": setup_key,
            "settings_fingerprint": fingerprint,
            "memory_cv_pct": (
                100 * values["memory_cv"]
                if values["memory_cv"] is not None
                else None
            ),
            "median_prediction_error_pct": (
                statistics.median(errors[key])
                if errors[key]
                else None
            ),
            "memory_measured_runs": values["measured_runs"],
            "predictions": len(errors[key]),
        })

    if not points:
        raise ValueError(f"No results found for stage {stage}")

    return points


def draw_panel(ax, points, models, metric, ylabel, title):
    plotted = 0

    for index, model in enumerate(models):
        # Assign each circuit the same horizontal position in both panels
        # and setup conditions. Horizontal offsets are visual only.
        identities = sorted({
            (point["partition"], point["setup_key"])
            for point in points
            if point["model"] == model
        })

        offsets = (
            [
                -0.05 + 0.10 * i / (len(identities) - 1)
                for i in range(len(identities))
            ]
            if len(identities) > 1
            else [0.0]
        )

        circuit_offsets = dict(zip(identities, offsets))

        for condition, shift, marker in (
            ("cold", -0.15, "o"),
            ("warm", 0.15, "s"),
        ):
            selected = [
                point for point in points
                if point["model"] == model
                and point["condition"] == condition
                and point[metric] is not None
            ]

            x = [
                index + shift + circuit_offsets[
                    (point["partition"], point["setup_key"])
                ]
                for point in selected
            ]

            y = [point[metric] for point in selected]

            ax.scatter(
                x,
                y,
                marker=marker,
                s=30,
                facecolors="black" if condition == "cold" else "white",
                edgecolors="black",
                linewidths=1,
                zorder=3,
            )

            plotted += len(selected)

    if not plotted:
        raise ValueError(f"No valid measurements for {metric}")

    ax.set_xticks(
        range(len(models)),
        [MODEL_LABELS[model] for model in models],
    )

    ax.set_xlim(-0.45, len(models) - 0.55)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.set_title(title, fontsize=9, pad=8)

    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser(
        description="Plot proving-memory repeatability and prediction accuracy."
    )

    parser.add_argument(
        "--analysis",
        type=Path,
        nargs="+",
        required=True,
        help="One or more results/analysis directories.",
    )

    parser.add_argument(
        "--stage",
        default="PROVING",
        help="Exact stage name in the analysis CSVs.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures"),
    )

    args = parser.parse_args()
    points = load_points(args.analysis, args.stage)

    models = [
        model for model in MODEL_ORDER
        if any(point["model"] == model for point in points)
    ]

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.7),
    )

    draw_panel(
        axes[0],
        points,
        models,
        "memory_cv_pct",
        "Memory variation across runs (CV, %)",
        "(a) Repeatability",
    )

    draw_panel(
        axes[1],
        points,
        models,
        "median_prediction_error_pct",
        "Median absolute prediction error (%)",
        "(b) Prediction accuracy",
    )

    legend = [
        Line2D(
            [], [],
            marker="o",
            linestyle="none",
            color="black",
            markersize=5,
            label="Cold",
        ),
        Line2D(
            [], [],
            marker="s",
            linestyle="none",
            color="black",
            markerfacecolor="white",
            markersize=5,
            label="Warm",
        ),
    ]

    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.99),
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    fig.text(
        0.5,
        0.02,
        "Each point represents one selected circuit under one setup condition",
        ha="center",
        fontsize=8,
        color="0.35",
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.22,
        top=0.77,
        wspace=0.38,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    for extension in ("pdf", "png"):
        path = args.output / f"resource_repeatability.{extension}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")

    plt.close(fig)

    # Retain the values and measurement counts behind every plotted point.
    summary_path = args.output / "resource_repeatability_points.csv"

    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)

    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()

# python plot_repeatability.py \
#   --analysis \
#     /path/to/mnist_campaign/results/analysis \
#     /path/to/mobilenet_campaign/results/analysis \
#     /path/to/nanogpt_campaign/results/analysis \
#   --output figures
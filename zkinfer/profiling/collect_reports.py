"""Collect research-friendly reports from a local request/run directory."""
import argparse
import csv
from pathlib import Path

from zkinfer.profiling.execution_trace import write_execution_trace
from zkinfer.profiling.halo2_phases import FIELDS, write_phase_report


def collect(root):
    root = Path(root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / "phase_report.csv"
    count = 0
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["job_directory", *FIELDS])
        writer.writeheader()
        for resource in sorted(root.rglob("resource_usage.csv")):
            directory = resource.parent
            if not (directory / "halo2_stage_transitions.csv").is_file():
                continue
            write_execution_trace(directory)
            write_phase_report(directory)
            with (directory / "halo2_phase_resources.csv").open(newline="", encoding="utf-8") as source:
                for row in csv.DictReader(source):
                    writer.writerow({"job_directory": str(directory.relative_to(root)), **row})
                    count += 1
    return output, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    output, count = collect(args.root)
    print(f"{output}: {count} phase occurrences")


if __name__ == "__main__":
    main()

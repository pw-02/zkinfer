"""Collect research-friendly reports from a local request/run directory."""
import argparse
import csv
from pathlib import Path

from zkinfer.profiling.execution_trace import write_execution_trace
from zkinfer.profiling.halo2_phases import FIELDS, phase_rows


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
            trace = directory / "execution_trace.csv"
            if not trace.is_file():
                write_execution_trace(directory)
            if not trace.is_file():
                continue
            for row in phase_rows(resource, trace):
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

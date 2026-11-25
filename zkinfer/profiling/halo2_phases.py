"""Offline attribution of sampled prover RSS to exact Halo2 phase events.

The phase events and the RSS logger use the same host's UTC wall clock.
Do not use these characterization statistics as admission reservations.
"""
import argparse
import bisect
import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

FIELDS = (
    "stage", "occurrence", "start_unix_us", "end_unix_us", "duration_s",
    "status", "rss_samples", "rss_first_gib", "rss_last_gib",
    "rss_peak_gib", "rss_mean_gib", "cpu_machine_mean_pct",
)


def _unix_us(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("RSS timestamps must include a timezone")
    delta = dt - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def _number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) and n >= 0 else None
    except (TypeError, ValueError):
        return None


def phase_rows(resource_path, transition_path):
    """Return one row per phase occurrence, including incomplete phases.

    Attribution uses half-open intervals [start, end), never sampled labels.
    Missing observations remain None. First/last are sampled values, not
    exact boundary measurements. CPU percentages cover sampling intervals.
    """
    samples = []
    if Path(resource_path).is_file():
        with open(resource_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("type") not in ("process", "process_tree"):
                    continue
                samples.append((_unix_us(row["timestamp"]),
                                _number(row.get("memory_gb")),
                                _number(row.get("cpu_machine_percent"))))
    samples.sort(key=lambda item: item[0])
    times = [s[0] for s in samples]
    pending = {}
    counts = defaultdict(int)
    rows = []

    def finish(stage, occurrence, start, end=None, duration=None, status="incomplete"):
        values = [] if end is not None and end < start else samples[
            bisect.bisect_left(times, start):bisect.bisect_left(times, end) if end is not None else len(samples)]
        rss = [v[1] for v in values if v[1] is not None]
        cpu = [v[2] for v in values if v[2] is not None]
        rows.append(dict(zip(FIELDS, (
            stage, occurrence, start, end, duration, status, len(rss),
            rss[0] if rss else None, rss[-1] if rss else None,
            max(rss) if rss else None, sum(rss)/len(rss) if rss else None,
            sum(cpu)/len(cpu) if cpu else None,
        ))))

    with open(transition_path, newline="", encoding="utf-8") as f:
        for event in csv.DictReader(f):
            if event.get("source", "halo2") != "halo2":
                continue
            stage = event["stage"]
            if not stage.startswith("PROVING::"):
                continue
            stamp = int(event["timestamp_unix_us"])
            if event["event"] == "start":
                if stage in pending:
                    occurrence, start = pending.pop(stage)
                    finish(stage, occurrence, start)
                counts[stage] += 1
                pending[stage] = (counts[stage], stamp)
            elif event["event"] == "end":
                if stage not in pending:
                    counts[stage] += 1
                    finish(stage, counts[stage], stamp, stamp, None, "missing_start")
                    continue
                occurrence, start = pending.pop(stage)
                duration = _number(event.get("duration_s"))
                status = "complete" if stamp >= start and duration is not None else "invalid"
                finish(stage, occurrence, start, stamp, duration, status)
    for stage, (occurrence, start) in pending.items():
        finish(stage, occurrence, start)
    return sorted(rows, key=lambda row: row["start_unix_us"])


def write_phase_report(directory, *, retain=True):
    directory = Path(directory)
    transitions = directory / "halo2_stage_transitions.csv"
    if not transitions.is_file():
        transitions = directory / "execution_trace.csv"
    if not transitions.is_file():
        return {"halo2_phase_trace_available": False}
    rows = phase_rows(directory / "resource_usage.csv", transitions)
    if retain:
        with (directory / "halo2_phase_resources.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    if not rows:
        return {"halo2_phase_trace_available": False}
    complete = [r for r in rows if r["status"] == "complete"]
    return {
        "halo2_phase_trace_available": True,
        "halo2_phase_count": len(rows),
        "halo2_phase_complete_count": len(complete),
        "halo2_phase_sampled_count": sum(r["rss_samples"] > 0 for r in complete),
        "halo2_phase_total_duration(s)": sum(r["duration_s"] for r in complete),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="One prover job artifact directory")
    args = parser.parse_args()
    print(write_phase_report(args.directory))


if __name__ == "__main__":
    main()

"""Consolidate per-job event streams without changing runtime accounting."""
import csv
from pathlib import Path
from datetime import datetime, timezone

FIELDS = ["timestamp_unix_us", "source", "stage", "event", "duration_s", "elapsed_s"]


def unix_us(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Expected timezone-aware timestamp")
    delta = dt - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def write_execution_trace(directory):
    directory = Path(directory)
    events = []
    target = directory / "execution_trace.csv"
    if not any((directory / name).is_file() for name in ("stage_transitions.csv", "halo2_stage_transitions.csv")):
        if target.is_file():
            with target.open(newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        return events
    for filename, source in [
        ("stage_transitions.csv", "runtime"),
        ("halo2_stage_transitions.csv", "halo2"),
    ]:
        path = directory / filename
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stamp = row.get("timestamp_unix_us")
                events.append({
                    "timestamp_unix_us": int(stamp) if stamp else unix_us(row["timestamp"]),
                    "source": source,
                    "stage": row["stage"],
                    "event": row.get("event") or "enter",
                    "duration_s": row.get("duration_s", ""),
                    "elapsed_s": row.get("elapsed_s", ""),
                })
    events.sort(key=lambda r: r["timestamp_unix_us"])
    if events:
        temporary = target.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(events)
        temporary.replace(target)
    return events

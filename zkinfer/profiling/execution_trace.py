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
    # Runtime already imports fused EZKL events; retain their provenance in a
    # separate stream rather than pretending all records are disjoint intervals.
    for filename, source in [
        ("stage_transitions.csv", "runtime"),
        ("fused_stage_transitions.csv", "ezkl"),
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
    # Imported fused transitions duplicate runtime enter records. Prefer the
    # original EZKL record when the timestamps and stage names coincide.
    fused = {(r["timestamp_unix_us"], r["stage"]) for r in events if r["source"] == "ezkl"}
    events = [r for r in events if not (
        r["source"] == "runtime" and r["event"] == "enter" and
        (r["timestamp_unix_us"], r["stage"]) in fused)]
    events.sort(key=lambda r: r["timestamp_unix_us"])
    if events:
        with (directory / "execution_trace.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(events)
    return events

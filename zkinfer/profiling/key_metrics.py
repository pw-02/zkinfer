"""Summarize native proving-key metrics without conflating nested timings."""
import csv
import math
from pathlib import Path


def summarize_key_metrics(path):
    """Missing metrics stay absent; zero reconstruction is valid only after a full load."""
    path = Path(path)
    if not path.is_file():
        return {}
    operations = {}
    representations = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                duration = float(row["duration_s"])
                if not math.isfinite(duration) or duration < 0:
                    continue
                component, operation = row["component"], row["operation"]
                byte_count = int(row["bytes"]) if row.get("bytes") else None
                if byte_count is not None and byte_count < 0:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            if operation == "materialize" and component in {"full", "reduced", "compact"}:
                representations.append((component, duration))
                continue
            operations.setdefault(operation, []).append((component, duration, byte_count))

    def total(operation, component=None):
        rows = [r for r in operations.get(operation, [])
                if component is None or r[0] == component]
        return sum(r[1] for r in rows) if rows else None

    result = {}
    serial = total("serialize_total", "pk_total")
    if serial is not None:
        result["pk_serialization_time(s)"] = serial
    # Historical component sums are useful, but not equivalent to file-level totals.
    components = total("serialize")
    if components is not None:
        result["pk_component_serialization_time(s)"] = components
    decoded = total("deserialize")
    if decoded is not None:
        result["pk_deserialization_time(s)"] = decoded
        counts = [r[2] for r in operations["deserialize"]]
        if all(n is not None for n in counts):
            result["pk_deserialized_bytes"] = sum(counts)
    reconstructed = total("reconstruct")
    if reconstructed is not None:
        result["pk_reconstruction_time(s)"] = reconstructed
    elif representations and all(r[0] == "full" for r in representations):
        result["pk_reconstruction_time(s)"] = 0.0
    materialized = total("materialize", "pk_total")
    if materialized is not None:
        result["pk_materialization_time(s)"] = materialized
        counts = [r[2] for r in operations["materialize"] if r[0] == "pk_total"]
        if all(n is not None for n in counts):
            result["pk_materialized_file_bytes"] = sum(counts)
    if representations:
        names = {r[0] for r in representations}
        result["pk_representation"] = next(iter(names)) if len(names) == 1 else "mixed"
        result["pk_halo2_materialization_time(s)"] = sum(r[1] for r in representations)
        result["pk_materialization_count"] = len(representations)
    return result

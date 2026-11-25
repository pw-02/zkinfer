"""Generate research summaries beside each request's existing reports."""
import argparse
import csv
import json
import logging
from pathlib import Path

from zkinfer.profiling.execution_trace import write_execution_trace
from zkinfer.profiling.halo2_phases import FIELDS, phase_rows


def _flatten(value, prefix):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten(item, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix, json.dumps(value) if isinstance(value, list) else value


def collect_request(request):
    request = Path(request)
    reports = request / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / "phase_report.csv"
    count = 0
    circuits = []
    jobs = request / "artifacts" / "jobs"
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["request_id", "job_id", "lease_id", "job_status", *FIELDS])
        writer.writeheader()
        statuses = {}
        report = reports / "job_report.csv"
        if report.is_file():
            with report.open(newline="", encoding="utf-8") as f:
                statuses = {row["job_id"]: row for row in csv.DictReader(f)}
        canonical = [(p, p.name, statuses.get(p.name, {}).get("job_status", "UNKNOWN"),
                      statuses.get(p.name, {}).get("lease_id", ""))
                     for p in sorted(jobs.iterdir()) if p.is_dir()] if jobs.is_dir() else []
        failed_root = request / "artifacts" / "failed_attempts"
        failed = [(p, p.parent.name, "FAILED", p.name)
                  for p in sorted(failed_root.glob("*/*")) if p.is_dir()]
        seen = set()
        for directory, job_id, status, lease_id in canonical + failed:
            identity = {"request_id": request.name, "job_id": job_id,
                        "lease_id": lease_id, "job_status": status}
            seen.add(job_id)
            circuit = dict(identity)
            try:
                features = directory / "circuit_features.json"
                if features.is_file():
                    payload = json.loads(features.read_text(encoding="utf-8"))
                    circuit.update(dict(_flatten(payload, "")))
                    circuit["lease_id"] = lease_id or payload.get("lease_id", "")
                else:
                    circuit["features_available"] = False
                circuits.append(circuit)
            except Exception:
                logging.exception("Cannot read circuit features: %s", directory)
                circuits.append({**identity, "features_available": False})
            try:
                resource = directory / "resource_usage.csv"
                if not resource.is_file():
                    continue
                trace = directory / "execution_trace.csv"
                if not trace.is_file():
                    write_execution_trace(directory)
                if not trace.is_file():
                    continue
                for row in phase_rows(resource, trace):
                    writer.writerow({**identity, **row})
                    count += 1
            except Exception:
                logging.exception("Cannot read phase trace: %s", directory)
        for job_id, row in statuses.items():
            if job_id not in seen:
                circuits.append({"request_id": request.name, "job_id": job_id,
                                 "lease_id": row.get("lease_id"), "job_status": row.get("job_status"),
                                 "features_available": False})
    temporary.replace(output)
    circuit_output = reports / "circuit_report.csv"
    fields = ["request_id", "job_id"] + sorted({k for row in circuits for k in row} - {"request_id", "job_id"})
    temporary = circuit_output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(circuits)
    temporary.replace(circuit_output)
    return output, count


def collect(root):
    root = Path(root)
    if (root / "artifacts").is_dir() or (root / "reports" / "job_report.csv").is_file():
        return [collect_request(root)]
    requests = root / "requests"
    if not requests.is_dir():
        raise ValueError(f"Expected a run or request directory: {root}")
    return [collect_request(request) for request in sorted(requests.iterdir())
            if (request / "artifacts").is_dir() or (request / "reports" / "job_report.csv").is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for output, count in collect(args.root):
        print(f"{output}: {count} phase occurrences")


if __name__ == "__main__":
    main()

"""Build E13 datasets from registered requests; never modify source runs."""
import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.register_runs import read_csv, read_json, write_csv, truth
from zkinfer.profiling.key_metrics import summarize_key_metrics

HERE = Path(__file__).resolve().parent
MODES = ("full", "reduced", "compact")
GIB = 2**30
METRICS = ("size_gib", "serialize_s", "deserialize_s", "reconstruct_s",
           "materialize_s", "residual_s", "keygen_s", "prove_s",
           "request_s", "peak_rss_gib", "logical_read_gib", "load_gib_s",
           "cached_fraction", "eviction_s", "probe_s")
GROUP = ("model", "partition_size", "storage_condition", "page_cache_condition",
         "observed_page_cache", "cache_condition", "circuit_cache_condition",
         "hardware", "rayon_threads", "settings_digest", "setup_key",
         "srs_digest", "zkinfer_commit", "ezkl_commit", "halo2_commit")
FIELDS = ("experiment_id", "run_id", "request_id", "job_id", "lease_id", "source_path",
          "repetition", "representation", "requested_representation", *GROUP,
          "verification", "job_status", "request_status", "materialization_count",
          "eligible", *METRICS)


def number(x):
    try:
        n = float(x)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def divide(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest() if obj else ""


def extract(entry, checks, components):
    run = ROOT / entry["source_path"]
    request = run / "requests" / entry["request_id"]
    reports = request / "reports"
    reqs = read_csv(reports / "request_report.csv")
    req = reqs[-1] if reqs else {}
    verification = read_json(reports / "verification_report.json")
    jobs = read_csv(reports / "job_report.csv")
    identity = {k: entry.get(k, "") for k in ("experiment_id", "run_id", "request_id", "source_path", "repetition")}
    def check(job, level, message):
        checks.append({**identity, "job_id": job, "level": level, "message": message})
    if not jobs:
        check("", "error", "Missing job report; request retained in index")
        return
    if len(jobs) != 1:
        check("", "error", "E13 pilot requires one canonical job per request")
    for failed in read_csv(reports / "failed_attempt_report.csv"):
        check(failed.get("job_id", ""), "warning", "Failed attempt retained in source reports")
    seen = set()
    for job in jobs:
        jid = job.get("job_id", "")
        if not jid or jid in seen:
            check(jid, "error", "Missing or duplicate job identity")
            continue
        seen.add(jid)
        directory = request / "artifacts/jobs" / jid
        native = directory / "halo2_key_metrics.csv"
        feature = read_json(directory / "circuit_features.json")
        for i, component in enumerate(read_csv(native), 1):
            components.append({**identity, "job_id": jid, "lease_id": job.get("lease_id"),
                               "representation": job.get("pk_representation"), "row": i,
                               **{k: component.get(k) for k in ("component", "operation", "bytes", "duration_s")}})
        measured = summarize_key_metrics(native)
        def metric(k):
            return number(measured.get(k, job.get(k)))
        mode = measured.get("pk_representation") or job.get("pk_representation", "")
        requested = job.get("requested_key_representation") or entry.get("key_representation")
        size = metric("pk_materialized_file_bytes")
        size = size / GIB if size is not None else metric("pk_file_size(GB)")
        read = metric("pk_deserialized_bytes")
        read = read / GIB if read is not None else None
        des = metric("pk_deserialization_time(s)")
        rec = metric("pk_reconstruction_time(s)")
        total = metric("pk_materialization_time(s)")
        count = metric("pk_materialization_count")
        residual = total - des - rec if None not in (total, des, rec) else None
        cached = metric("pk_page_cache_cached_bytes_after") if truth(job.get("pk_page_cache_eviction_requested")) else metric("pk_page_cache_cached_bytes_before")
        fraction = divide(cached, metric("pk_page_cache_file_bytes"))
        observed = ("unknown" if fraction is None else "cold" if fraction == 0 else
                    "resident" if fraction >= 0.99 else "partial")
        row = {**identity, **{k: entry.get(k, "") for k in GROUP},
               "job_id": jid, "lease_id": job.get("lease_id"), "representation": mode,
               "requested_representation": requested, "observed_page_cache": observed,
               "rayon_threads": job.get("prover_rayon_pool_size") or feature.get("rayon_pool_size") or entry.get("rayon_threads"),
               "settings_digest": digest(feature.get("settings")), "setup_key": job.get("setup_key"),
               "srs_digest": job.get("prover_srs_sha256"), "verification": "valid" if verification.get("enabled") is True and verification.get("valid") is True else "invalid" if verification.get("enabled") is True and verification.get("valid") is False else "unconfirmed",
               "job_status": job.get("job_status"), "request_status": req.get("request_status"),
               "materialization_count": count, "size_gib": size, "serialize_s": metric("pk_serialization_time(s)"),
               "deserialize_s": des, "reconstruct_s": rec, "materialize_s": total, "residual_s": residual,
               "keygen_s": metric("ezkl_key_gen_time(s)"), "prove_s": metric("ezkl_proof_time(s)"),
               "request_s": number(req.get("total_elapsed_time(s)")), "peak_rss_gib": metric("max_process_memory(GB)"),
               "logical_read_gib": read, "load_gib_s": divide(read, des), "cached_fraction": fraction,
               "eviction_s": metric("pk_page_cache_eviction_time(s)"), "probe_s": metric("pk_page_cache_probe_time(s)")}
        errors = []
        if len(jobs) != 1 or str(row["job_status"]).upper() != "COMPLETED" or str(row["request_status"]).upper() != "COMPLETED":
            errors.append("Not a completed monolithic request")
        if row["verification"] != "valid":
            errors.append("Proof verification not confirmed")
        if mode not in MODES or mode != requested:
            errors.append("Requested and observed representation differ or missing")
        if count != 1 or None in (size, des, rec, total) or size <= 0:
            errors.append("Incomplete or ambiguous PK materialization metrics")
        if residual is not None and residual < -0.01:
            errors.append("Component durations exceed total materialization; inspect timer nesting")
        if not native.is_file():
            errors.append("Missing native metrics CSV")
        if entry.get("partition_size") != "mono":
            errors.append("Not a monolithic configuration")
        if entry.get("cache_condition") not in ("cold", "warm"):
            errors.append("Actual key cache condition unknown")
        for k in GROUP:
            if not row.get(k) or row[k] == "unknown":
                check(jid, "warning", f"Unconfirmed comparison field: {k}")
        if row.get("circuit_cache_condition") != "hit":
            check(jid, "warning", "Circuit cache hit not confirmed")
        for error in errors:
            check(jid, "error", error)
        row["eligible"] = not errors
        yield row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=ROOT / "research/experiment_index.csv")
    p.add_argument("--output", type=Path, default=HERE / "results")
    args = p.parse_args()
    checks, components, rows = [], [], []
    for entry in read_csv(args.index):
        if entry.get("experiment_id") != "E13":
            continue
        try:
            rows.extend(extract(entry, checks, components))
        except (OSError, ValueError, KeyError) as exc:
            checks.append({k: entry.get(k) for k in ("experiment_id", "run_id", "request_id", "source_path", "repetition")} | {"job_id": "", "level": "error", "message": str(exc)})
    groups = defaultdict(list)
    for row in rows:
        if row["eligible"]:
            groups[tuple(str(row.get(k) or "") for k in GROUP)].append(row)
    summary, thresholds = [], []
    for group, records in sorted(groups.items()):
        context = dict(zip(GROUP, group))
        by_mode = {m: [r for r in records if r["representation"] == m] for m in MODES}
        means = {}
        for mode, rr in by_mode.items():
            if len(rr) < 3:
                checks.append({"level": "warning", "message": f"{context}: {mode} has {len(rr)} eligible repetitions; target is 3"})
            means[mode] = {}
            for metric in METRICS:
                values = [r[metric] for r in rr if r[metric] is not None]
                if values:
                    means[mode][metric] = statistics.mean(values)
                summary.append({**context, "representation": mode, "metric": metric, "n": len(values),
                                "mean": statistics.mean(values) if values else None,
                                "sd": statistics.stdev(values) if len(values) > 1 else None,
                                "median": statistics.median(values) if values else None,
                                "min": min(values) if values else None, "max": max(values) if values else None})
        for larger, smaller in (("full", "reduced"), ("full", "compact"), ("reduced", "compact")):
            a, b = means[larger], means[smaller]
            if all(k in d for d in (a,b) for k in ("size_gib", "reconstruct_s")):
                ds = a["size_gib"] - b["size_gib"]
                dc = b["reconstruct_s"] - a["reconstruct_s"]
                thresholds.append({**context, "larger": larger, "smaller": smaller,
                                   "saved_gib": ds, "extra_reconstruct_s": dc,
                                   "break_even_gib_s": ds / dc if ds > 0 and dc > 0 else None})
    out = args.output
    write_csv(out / "measurements.csv", rows, FIELDS)
    write_csv(out / "component_measurements.csv", components, ["experiment_id","run_id","request_id","source_path","repetition","job_id","lease_id","representation","row","component","operation","bytes","duration_s"])
    write_csv(out / "validation.csv", checks, ["experiment_id","run_id","request_id","source_path","repetition","job_id","level","message"])
    write_csv(out / "summary.csv", summary, [*GROUP,"representation","metric","n","mean","sd","median","min","max"])
    write_csv(out / "break_even.csv", thresholds, [*GROUP,"larger","smaller","saved_gib","extra_reconstruct_s","break_even_gib_s"])
    print(f"{len(rows)} jobs; {sum(r['eligible'] for r in rows)} eligible; {len(groups)} comparison groups. See {out / 'validation.csv'}")


if __name__ == "__main__":
    main()

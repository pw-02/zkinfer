"""Register explicitly selected experimental runs without modifying raw data."""
import argparse
import csv
import json
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "research/experiment_index.csv"
EXTRA = ["key_representation", "storage_condition", "page_cache_condition",
         "hardware", "circuit_cache_condition"]


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def consensus(values):
    values = {str(v) for v in values if v is not None and str(v) != ""}
    return next(iter(values)) if len(values) == 1 else "mixed" if values else ""


def truth(value):
    return {"true": True, "false": False, "1": True, "0": False}.get(str(value).lower())


def register(run, args):
    cfg = OmegaConf.load(run / "config.yaml")
    get = lambda key: OmegaConf.select(cfg, key)
    requests = sorted(p for p in (run / "requests").glob("*") if p.is_dir())
    if not requests:
        raise ValueError(f"No request directories in {run}")
    metadata = [read_json(p) for p in (run / ".runtime/processes").glob("*.json")]
    revision = consensus(m.get("code", {}).get("commit") for m in metadata)
    if any(m.get("code", {}).get("dirty") for m in metadata):
        revision += "+dirty"
    source = str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run)
    for request in requests:
        reports = request / "reports"
        jobs = read_csv(reports / "job_report.csv")
        rr = read_csv(reports / "request_report.csv")
        rr = rr[-1] if rr else {}
        model = get("workload") or rr.get("request_name", "")
        shapes = get(f"workload_definitions.{model}.input_shapes")
        hits = [truth(j.get("ezkl_key_gen_used_cache")) for j in jobs]
        cache = ("warm" if hits and all(h is True for h in hits) else
                 "cold" if hits and all(h is False for h in hits) else
                 "mixed" if True in hits and False in hits else "")
        try:
            circuit_hits = int(float(rr.get("upfront_setup_cache_hits", "")))
            count = int(float(rr.get("num_proof_jobs", "")))
            circuit = ("hit" if count > 0 and circuit_hits == count else
                       "miss" if count > 0 and circuit_hits == 0 else "mixed")
        except (TypeError, ValueError):
            circuit = ""
        yield dict(experiment_id=args.experiment, run_id=run.name,
                   request_id=request.name, source_path=source, model=model,
                   input_shape=json.dumps(OmegaConf.to_container(shapes, resolve=True)) if shapes is not None else "",
                   partition_size=get("partitioning.size") if get("partitioning.enabled") else "mono",
                   cache_condition=cache, scheduler=get("scheduling.policy"),
                   admission=get("scheduling.memory_admission"),
                   concurrency=get("scheduling.max_concurrent_jobs"),
                   nodes=len({j["node_id"] for j in jobs if j.get("node_id")}) or "",
                   rayon_threads=consensus(j.get("prover_rayon_pool_size") for j in jobs) or get("resources.prover_threads"),
                   status=rr.get("request_status", "unknown"), zkinfer_commit=revision,
                   ezkl_commit=args.ezkl_commit, halo2_commit=args.halo2_commit,
                   key_representation=get("proving.key_representation") or consensus(j.get("requested_key_representation") for j in jobs),
                   storage_condition=args.storage_condition, page_cache_condition=args.page_cache_condition,
                   hardware=args.hardware, circuit_cache_condition=circuit, notes=args.notes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--index", type=Path, default=INDEX)
    for field in ("storage-condition", "page-cache-condition", "hardware", "ezkl-commit", "halo2-commit", "notes"):
        p.add_argument("--" + field, default="")
    args = p.parse_args()
    with args.index.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        existing = list(reader)
    fields += [x for x in EXTRA if x not in fields]
    key = lambda r: (r["experiment_id"], r["run_id"], r["request_id"])
    rows = {key(r): r for r in existing}
    for run in args.runs:
        for record in register(run.resolve(), args):
            previous = rows.get(key(record), {})
            if previous.get("source_path") and previous["source_path"] != record["source_path"]:
                raise ValueError(f"Conflicting source paths for {key(record)}")
            rows[key(record)] = {**previous, **{k:v for k,v in record.items() if v is not None and v != ""}}
    # Preserve manually assigned repetition numbers; fill unused numbers within conditions.
    groups = {}
    for r in rows.values():
        g = tuple(r.get(k, "") for k in ("experiment_id", "model", "partition_size", "key_representation",
                  "cache_condition", "storage_condition", "page_cache_condition", "hardware", "rayon_threads"))
        groups.setdefault(g, []).append(r)
    for group in groups.values():
        used = {int(r["repetition"]) for r in group if str(r.get("repetition", "")).isdigit()}
        for r in sorted(group, key=key):
            if not r.get("repetition"):
                i = next(i for i in range(1, len(group)+len(used)+1) if i not in used)
                r["repetition"] = i
                used.add(i)
    write_csv(args.index, sorted(rows.values(), key=key), fields)
    print(f"Registered {len(rows)} request/experiment entries in {args.index}")


if __name__ == "__main__":
    main()

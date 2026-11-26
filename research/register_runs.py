"""Register selected requests in the research index without reading metrics."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "research/experiment_index.csv"


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--index", type=Path, default=INDEX)
    p.add_argument("--notes", default="")
    args = p.parse_args()
    with args.index.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        existing = list(reader)
    key = lambda r: (r["experiment_id"], r["run_id"], r["request_id"])
    rows = {key(r): r for r in existing}
    for run in args.runs:
        run = run.resolve()
        requests = sorted(p for p in (run / "requests").glob("*") if p.is_dir())
        if not requests:
            p.error(f"No request directories in {run}")
        source = str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run)
        for request in requests:
            record = dict(experiment_id=args.experiment, run_id=run.name,
                          request_id=request.name, source_path=source)
            previous = rows.get(key(record), {})
            if previous.get("source_path") and previous["source_path"] != source:
                p.error(f"Conflicting source paths for {key(record)}")
            rows[key(record)] = {**previous, **record,
                                 "notes": args.notes or previous.get("notes", "")}
    write_csv(args.index, sorted(rows.values(), key=key), fields)
    print(f"Registered {len(rows)} request/experiment entries in {args.index}")


if __name__ == "__main__":
    main()

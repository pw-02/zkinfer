#!/usr/bin/env python3
"""Submit many zkInfer requests and measure batch makespan and throughput.

Every input becomes an ordinary zkInfer request. Scheduling policy is configured
once at the coordinator; individual requests do not select independent policies.
Keeping the same logical model name preserves setup-cache identity across the
batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import grpc
from pathlib import Path
from typing import Dict, List

from zkinfer.client.api import ZKInferenceClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="localhost:50051")
    parser.add_argument("--name", required=True, help="Logical model/workload name")
    parser.add_argument("--model", required=True, help="ONNX model path visible to coordinator")
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Input JSON path. Repeat to submit multiple distinct inputs.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the supplied input list this many times.",
    )
    parser.add_argument("--split-mode", default="fixed")
    parser.add_argument("--ops-per-chunk", type=int, default=1)
    parser.add_argument("--simplify-model", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/batch_results"))
    return parser.parse_args()


def _print_batch_progress(statuses: Dict[str, object]) -> None:
    total_requests = len(statuses)
    completed_requests = sum(
        1 for status in statuses.values() if status.status == "completed"
    )
    failed_requests = sum(
        1 for status in statuses.values() if status.status in {"failed", "unknown"}
    )
    total_jobs = sum(int(status.total_jobs) for status in statuses.values())
    completed_jobs = sum(int(status.completed_jobs) for status in statuses.values())
    running_jobs = sum(int(status.in_progress_jobs) for status in statuses.values())
    queued_jobs = sum(int(status.queued_jobs) for status in statuses.values())

    print(
        "Batch progress | "
        f"requests={completed_requests}/{total_requests} completed, {failed_requests} failed | "
        f"jobs={completed_jobs}/{total_jobs} completed, {running_jobs} running, {queued_jobs} queued",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    inputs: List[str] = list(args.inputs) * args.repeat
    client = ZKInferenceClient(target=args.target)

    batch_started = time.perf_counter()
    submitted: List[Dict[str, object]] = []
    for index, input_path in enumerate(inputs, start=1):
        request_started = time.perf_counter()
        try:
            request_id = client.submit_inference_request(
                name=args.name,
                onnx_model_path=args.model,
                input_data_path=input_path,
                split_mode=args.split_mode,
                ops_per_chunk=args.ops_per_chunk,
                simplify_model=args.simplify_model,
            )
        except grpc.RpcError as exc:
            # Preserve the offered burst and report explicit admission rejection.
            # Never retry an ambiguous submission automatically.
            if exc.code() != grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise
            submitted.append({
                "batch_index": index,
                "request_id": None,
                "input_path": input_path,
                "submit_offset_s": time.perf_counter() - batch_started,
                "submit_rpc_time_s": time.perf_counter() - request_started,
                "status": "rejected",
                "message": exc.details(),
            })
            continue
        submitted.append(
            {
                "batch_index": index,
                "request_id": request_id,
                "input_path": input_path,
                "submit_offset_s": time.perf_counter() - batch_started,
                "submit_rpc_time_s": time.perf_counter() - request_started,
                "status": "submitted",
            }
        )
        print(f"Submitted request {index}/{len(inputs)}: {request_id}", flush=True)

    remaining = {str(item["request_id"]): item for item in submitted if item["request_id"]}
    previous_progress = None
    while remaining:
        statuses = {
            str(item["request_id"]): client.get_request_status(str(item["request_id"]))
            for item in submitted if item["request_id"]
        }
        progress = tuple(
            sorted(
                (
                    request_id,
                    status.status,
                    int(status.total_jobs),
                    int(status.completed_jobs),
                    int(status.in_progress_jobs),
                    int(status.queued_jobs),
                    int(status.failed_jobs),
                )
                for request_id, status in statuses.items()
            )
        )
        if progress != previous_progress:
            _print_batch_progress(statuses)
            previous_progress = progress

        for request_id in list(remaining):
            status = statuses[request_id]
            if status.status not in {"completed", "failed", "unknown"}:
                continue
            item = remaining.pop(request_id)
            item["status"] = status.status
            item["message"] = status.message
            item["total_jobs"] = int(status.total_jobs)
            item["completed_jobs"] = int(status.completed_jobs)
            item["failed_jobs"] = int(status.failed_jobs)
            item["completion_offset_s"] = time.perf_counter() - batch_started
        if remaining:
            time.sleep(max(0.1, args.poll_interval))

    makespan_s = time.perf_counter() - batch_started
    completed_requests = sum(1 for item in submitted if item["status"] == "completed")
    failed_requests = len(submitted) - completed_requests
    completed_jobs = sum(int(item.get("completed_jobs") or 0) for item in submitted)
    total_jobs = sum(int(item.get("total_jobs") or 0) for item in submitted)

    summary = {
        "name": args.name,
        "num_requests": len(submitted),
        "completed_requests": completed_requests,
        "failed_requests": failed_requests,
        "rejected_requests": sum(item["status"] == "rejected" for item in submitted),
        "total_proof_jobs": total_jobs,
        "completed_proof_jobs": completed_jobs,
        "makespan_s": makespan_s,
        "completed_requests_per_s": completed_requests / makespan_s if makespan_s > 0 else 0.0,
        "completed_proof_jobs_per_s": completed_jobs / makespan_s if makespan_s > 0 else 0.0,
        "split_mode": args.split_mode,
        "ops_per_chunk": args.ops_per_chunk,
        "requests": submitted,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "batch_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "batch_requests.csv"
    fields = sorted({key for item in submitted for key in item})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(submitted)

    print(json.dumps({key: value for key, value in summary.items() if key != "requests"}, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")

    if failed_requests:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

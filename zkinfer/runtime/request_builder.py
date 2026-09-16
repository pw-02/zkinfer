import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from zkinfer.config.runtime import FileTransferConfig, ProvingCacheConfig
from zkinfer.graph.onnx_splitter import split_onnx_model_with_inputs
from zkinfer.runtime.models import ProofJob
from zkinfer.storage.io import (
    exists,
    load_json,
    save_json,
    save_model_proto,
)


class RequestBuilder:
    def __init__(
        self,
        logger: logging.Logger,
        input_upload_workers: int = 16,
    ):
        self.logger = logger
        self.input_upload_workers = max(1, input_upload_workers)

    def build_jobs(
        self,
        request,
        file_transfer: FileTransferConfig,
        proving_cache: ProvingCacheConfig,
        max_retries: int,
        transfer_only: bool = False,
    ) -> List[ProofJob]:
        materialized_models = self._load_or_split_model(request)

        jobs: List[ProofJob] = []
        pending_input_uploads = []

        for item in materialized_models:
            job_dir = os.path.join(
                file_transfer.root_dir,
                request.request_id,
                item.parent_hash,
                item.sub_hash,
            )

            model_file_path = os.path.join(
                job_dir,
                "model.onnx",
            )

            input_file_path = os.path.join(
                job_dir,
                "input.json",
            )

            cache_path = self._build_cache_path(
                request=request,
                proving_cache=proving_cache,
                model_name=item.name,
                parent_model_hash=item.parent_hash,
                model_hash=item.sub_hash,
            )

            if transfer_only:
                # The transfer-only benchmark measures request inputs.
                # Model transfer and proving profiles are therefore skipped.
                model_write_time = 0.0
                profiling_file_path = None
                profiling_data = {}
                predicted_duration = 0.0

            else:
                profiling_file_path = os.path.join(
                    cache_path,
                    "profiling.json",
                )

                model_write_time = self._save_model_if_needed(
                    model_proto=item.model,
                    model_file_path=model_file_path,
                    file_transfer=file_transfer,
                )

                profiling_data = self._load_profiling_data(
                    model_name=item.name,
                    profiling_file_path=profiling_file_path,
                    file_transfer=file_transfer,
                )

                predicted_duration = float(
                    profiling_data.get(
                        "job_runtime(s)",
                        0.0,
                    )
                    or 0.0
                )

            job = ProofJob(
                inference_request_name=request.name,
                model_name=item.name,
                inference_request_id=request.request_id,
                model_path=model_file_path,
                input_path=input_file_path,
                profiling_file_path=profiling_file_path,
                model_write_time=(
                    model_write_time
                    if file_transfer.backend == "s3"
                    else 0.0
                ),
                input_write_time=0.0,
                profiling_data=profiling_data,
                predicted_duration=predicted_duration,
                max_retries=max_retries,
                parent_model_hash=item.parent_hash,
                model_hash=item.sub_hash,
                cache_path=cache_path,
            )

            jobs.append(job)

            # Delay the input upload until every job has been constructed.
            # The pending inputs are uploaded concurrently below.
            pending_input_uploads.append(
                (
                    job,
                    item.input_data,
                    input_file_path,
                )
            )

        self._save_inputs(
            pending_input_uploads=pending_input_uploads,
            file_transfer=file_transfer,
            preparation_metrics=request.preparation_metrics,
        )

        return jobs

    def _save_inputs(
        self,
        pending_input_uploads,
        file_transfer: FileTransferConfig,
        preparation_metrics: Dict[str, Any],
    ) -> None:
        if not pending_input_uploads:
            preparation_metrics[
                "input_upload_wall_time(s)"
            ] = 0.0

            preparation_metrics[
                "input_upload_service_time(s)"
            ] = 0.0

            preparation_metrics[
                "input_upload_workers"
            ] = 0

            return

        # Concurrent uploads are useful for S3. Keep filesystem writes
        # sequential to avoid unnecessary local I/O contention.
        if file_transfer.backend == "s3":
            upload_workers = min(
                self.input_upload_workers,
                len(pending_input_uploads),
            )
        else:
            upload_workers = 1

        upload_start = time.perf_counter()

        if upload_workers == 1:
            for (
                job,
                input_data,
                input_file_path,
            ) in pending_input_uploads:
                duration = self._save_input(
                    input_data=input_data,
                    input_file_path=input_file_path,
                    file_transfer=file_transfer,
                )

                if file_transfer.backend == "s3":
                    job.input_write_time = duration

        else:
            with ThreadPoolExecutor(
                max_workers=upload_workers,
                thread_name_prefix="input-upload",
            ) as executor:
                future_to_job = {
                    executor.submit(
                        self._save_input,
                        input_data,
                        input_file_path,
                        file_transfer,
                    ): job
                    for (
                        job,
                        input_data,
                        input_file_path,
                    ) in pending_input_uploads
                }

                for future in as_completed(future_to_job):
                    job = future_to_job[future]

                    # Calling result() also propagates upload failures.
                    job.input_write_time = future.result()

        input_upload_wall_time = (
            time.perf_counter() - upload_start
        )

        input_upload_service_time = sum(
            job.input_write_time
            for job, _, _ in pending_input_uploads
        )

        preparation_metrics[
            "input_upload_wall_time(s)"
        ] = input_upload_wall_time

        preparation_metrics[
            "input_upload_service_time(s)"
        ] = input_upload_service_time

        preparation_metrics[
            "input_upload_workers"
        ] = upload_workers

        self.logger.info(
            "Saved %d job inputs using %d upload workers in %.3fs "
            "(aggregate service time %.3fs)",
            len(pending_input_uploads),
            upload_workers,
            input_upload_wall_time,
            input_upload_service_time,
        )

    def _load_or_split_model(self, request):
        split_mode = (
            request.split_mode or "none"
        ).lower()

        self.logger.info(
            "Preparing request %s using split_mode=%s "
            "ops_per_chunk=%s simplify_model=%s",
            request.request_id,
            split_mode,
            request.ops_per_chunk,
            getattr(
                request,
                "simplify_model",
                False,
            ),
        )

        return split_onnx_model_with_inputs(
            model_path=request.onnx_model_path,
            input_data_path=request.input_data_path,
            split_mode=split_mode,
            split_group_size=request.ops_per_chunk,
            simplify_model=getattr(
                request,
                "simplify_model",
                False,
            ),
            simplified_model_path=getattr(
                request,
                "simplified_model_path",
                None,
            ),
            input_shapes=getattr(
                request,
                "input_shapes",
                None,
            ),
            model_name=request.name,
            metrics=request.preparation_metrics,
        )

    def _build_cache_path(
        self,
        request,
        proving_cache: ProvingCacheConfig,
        model_name: str,
        parent_model_hash: str,
        model_hash: str,
    ) -> str:
        parent_cache_dir = os.path.join(
            proving_cache.root_dir,
            f"{request.name}_{parent_model_hash}",
        )

        split_mode = (
            request.split_mode or "none"
        ).lower()

        if split_mode == "none":
            return os.path.join(
                parent_cache_dir,
                "full",
                parent_model_hash,
            )

        split_key = (
            f"{split_mode}_{request.ops_per_chunk}"
        )

        return os.path.join(
            parent_cache_dir,
            "splits",
            split_key,
            f"{model_name}_{model_hash}",
        )

    def _save_model_if_needed(
        self,
        model_proto,
        model_file_path: str,
        file_transfer: FileTransferConfig,
    ) -> float:
        if exists(
            model_file_path,
            storage_type=file_transfer.backend,
            s3_bucket=file_transfer.s3_bucket,
        ):
            return 0.0

        start = time.perf_counter()

        save_model_proto(
            model_proto,
            model_file_path,
            storage_type=file_transfer.backend,
            s3_bucket=file_transfer.s3_bucket,
        )

        return time.perf_counter() - start

    def _load_profiling_data(
        self,
        model_name: str,
        profiling_file_path: str,
        file_transfer: FileTransferConfig,
    ) -> Dict[str, Any]:
        if not exists(
            profiling_file_path,
            storage_type=file_transfer.backend,
            s3_bucket=file_transfer.s3_bucket,
        ):
            self.logger.warning(
                "Profiling data not found for %s at %s. "
                "Using predicted_duration=0.0",
                model_name,
                profiling_file_path,
            )

            return {}

        return load_json(
            profiling_file_path,
            storage_type=file_transfer.backend,
            s3_bucket=file_transfer.s3_bucket,
        )

    def _save_input(
        self,
        input_data,
        input_file_path: str,
        file_transfer: FileTransferConfig,
    ) -> float:
        start = time.perf_counter()

        save_json(
            input_data,
            input_file_path,
            storage_type=file_transfer.backend,
            s3_bucket=file_transfer.s3_bucket,
        )

        return time.perf_counter() - start
import logging
import os
import time
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
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def build_jobs(
        self,
        request,
        file_transfer: FileTransferConfig,
        proving_cache: ProvingCacheConfig,
        max_retries: int,
    ) -> List[ProofJob]:
        materialized_models = self._load_or_split_model(request)
        jobs: List[ProofJob] = []
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

            # Persist the profile across requests, beside this submodel's
            # reusable proving artifacts.
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
                profiling_data.get("job_runtime(s)", 0.0) or 0.0
            )

            if predicted_duration > 0:
                self.logger.info(
                    "Loaded profile for %s: predicted_duration=%.3fs from %s",
                    item.name,
                    predicted_duration,
                    profiling_file_path,
                )

            input_write_time = self._save_input(
                input_data=item.input_data,
                input_file_path=input_file_path,
                file_transfer=file_transfer,
            )

            jobs.append(
                ProofJob(
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
                    input_write_time=(
                        input_write_time
                        if file_transfer.backend == "s3"
                        else 0.0
                    ),
                    profiling_data=profiling_data,
                    predicted_duration=predicted_duration,
                    max_retries=max_retries,
                    parent_model_hash=item.parent_hash,
                    model_hash=item.sub_hash,
                    cache_path=cache_path,
                )
            )


        return jobs

    def _load_or_split_model(self, request):
        split_mode = (request.split_mode or "none").lower()

        self.logger.info(
            "Preparing request %s using split_mode=%s ops_per_chunk=%s simplify_model=%s",
            request.request_id,
            split_mode,
            request.ops_per_chunk,
            getattr(request, "simplify_model", False),
        )

        return split_onnx_model_with_inputs(
            model_path=request.onnx_model_path,
            input_data_path=request.input_data_path,
            split_mode=split_mode,
            split_group_size=request.ops_per_chunk,
            simplify_model=getattr(request, "simplify_model", False),
            simplified_model_path=getattr(request, "simplified_model_path", None),
            input_shapes=getattr(request, "input_shapes", None),
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

        split_mode = (request.split_mode or "none").lower()

        if split_mode == "none":
            return os.path.join(
                parent_cache_dir,
                "full",
                parent_model_hash,
            )

        split_key = f"{split_mode}_{request.ops_per_chunk}"

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
                "Profiling data not found for %s at %s. Using predicted_duration=0.0",
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

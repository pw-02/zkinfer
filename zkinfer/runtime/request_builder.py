import logging
import os
import time
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List

from zkinfer.config.runtime import SetupStoreConfig
from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.models import ProofJob
from zkinfer.runtime.resource_profiles import get_profile_estimates, max_stage_memory_gb
from zkinfer.storage.shared_store import SharedStorage
from zkinfer.storage.io import exists, load_json


# Captures setup-affecting choices currently hard-coded by EZKLProofStages.
# Bump setup_identity_version whenever those semantics or the custom EZKL setup
# implementation change in a way that can alter settings, compiled circuits,
# PKs, or VKs.
EZKL_SETUP_IDENTITY = {
    "setup_identity_version": 2,
    "backend": "ezkl-upstream-v23",
    "settings_generation": "gen_settings_then_calibrate_with_fallback",
    "calibration_target": "resources",
    "proof_type": "single",
    "disable_selector_compression": False,
}


class RequestBuilder:
    """Materialize one inference request into independently executable proof jobs."""

    def __init__(
        self, logger: logging.Logger, witness_preparation: str = "independent",
        max_propagated_scale: int = 16, allow_boundary_rescale: bool = True,
    ):
        self.logger = logger
        self.witness_preparation = witness_preparation
        self.max_propagated_scale = max_propagated_scale
        self.allow_boundary_rescale = allow_boundary_rescale

    def build_jobs(
        self,
        request,
        shared_storage: SharedStorage,
        setup_store: SetupStoreConfig,
        max_retries: int,
        transfer_only: bool = False,
    ) -> List[ProofJob]:
        # transfer_only is retained only for compatibility with historical tests
        # and stored runtime state. New configuration cannot enable it.
        materialized_partitions = self._load_or_split_model(request)
        jobs: List[ProofJob] = []
        prepared = {}
        request.preparation_metrics["witness_preparation_mode"] = self.witness_preparation
        if self.witness_preparation == "upfront":
            if transfer_only or shared_storage.backend != "filesystem" or setup_store.backend != "filesystem":
                raise ValueError("Upfront preparation requires filesystem proving storage")
            from zkinfer.backends.ezkl.preparation import prepare_partitions
            import resource
            started = time.perf_counter()
            prepared = prepare_partitions(
                materialized_partitions,
                Path(shared_storage.requests_root) / request.request_id / "prepared",
                cache_root=(Path(shared_storage.setup_root) / "_prepared_v1" if setup_store.enabled else None),
                max_propagated_scale=self.max_propagated_scale,
                allow_rescale=self.allow_boundary_rescale,
            )
            metrics = request.preparation_metrics
            metrics["upfront_preparation_time(s)"] = time.perf_counter() - started
            # Linux process-wide RSS high-water mark; not a per-request peak
            # when requests overlap or previous requests used more memory.
            metrics["coordinator_process_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            for field in ("setup_time_s", "calibration_time_s", "compilation_time_s", "witness_time_s",
                          "witness_bytes", "compiled_bytes", "rescaled_inputs", "setup_cache_hit"):
                metrics[f"upfront_{field}"] = sum(part.metrics[field] for part in prepared.values())
            metrics["upfront_partition_count"] = len(prepared)
            if request.reports_dir:
                from zkinfer.backends.ezkl.preparation import write_json
                write_json(Path(request.reports_dir) / "witness_preparation.json",
                           {name: part.metrics for name, part in prepared.items()})

        total_model_store_time = 0.0
        total_input_store_time = 0.0

        for index, partition in enumerate(materialized_partitions):
            model_write_time = 0.0
            prepared_part = prepared.get(partition.name)
            identity = dict(EZKL_SETUP_IDENTITY)
            if prepared_part is not None:
                identity.update(prepared_part.artifact_identity())
            if transfer_only:
                model_ref = ArtifactRef.empty()
                setup_key = partition.sub_hash
                setup_ref = ArtifactRef.empty()
            else:
                model_bytes = partition.model.SerializeToString()
                setup_key = shared_storage.setup_key_for_model(
                    model_bytes,
                    identity,
                )
                started = time.perf_counter()
                model_ref = shared_storage.put_model(model_bytes, setup_key=setup_key)
                setup_ref = shared_storage.setup_ref(setup_key)
                model_write_time = time.perf_counter() - started
                total_model_store_time += model_write_time

            # Scheduling history is reusable even when circuit/PK/VK reuse is off.
            # Keep forced-cold observations separate from reuse-enabled history.
            resource_profile_path = None
            if setup_store.resource_profiles_enabled and not transfer_only:
                profile_root = setup_store.resource_profiles_root_dir or setup_store.root_dir
                filename = "resource_profile.json" if setup_store.enabled else "cold_resource_profile.json"
                resource_profile_path = os.path.join(profile_root, setup_key, filename)
            profiling_data = (
                self._load_resource_profile(partition.name, resource_profile_path, setup_store)
                if resource_profile_path else {}
            )
            profile_estimates = get_profile_estimates(profiling_data)
            predicted_duration_s = float(profile_estimates.get("job_runtime_s") or 0.0)
            predicted_peak_memory_gb = max_stage_memory_gb(profiling_data)
            partition_id = f"{index:05d}_{setup_key[:16]}"
            output_prefix_ref = shared_storage.proof_prefix(request.request_id, partition_id)

            started = time.perf_counter()
            input_data = partition.input_data
            if prepared_part is not None:
                from zkinfer.backends.ezkl.preparation import PREPARED_INPUT_KIND, file_digest

                def artifact(path, digest=None):
                    return ArtifactRef(
                        "filesystem", str(path.resolve()),
                        sha256=digest or file_digest(path), size_bytes=path.stat().st_size,
                    ).to_dict()

                input_data = {
                    "kind": PREPARED_INPUT_KIND,
                    "request_id": request.request_id,
                    "partition_name": partition.name,
                    "setup_key": setup_key,
                    "settings": artifact(prepared_part.settings_path, identity["settings_sha256"]),
                    "compiled": artifact(prepared_part.compiled_path, identity["compiled_sha256"]),
                    "witness": artifact(prepared_part.witness_path),
                    "input_specs": [asdict(spec) for spec in prepared_part.input_specs],
                    "output_specs": [asdict(spec) for spec in prepared_part.output_specs],
                }
            input_ref = shared_storage.put_input(
                request.request_id,
                partition_id + (".prepared" if prepared_part is not None else ""),
                input_data,
            )
            input_write_time = time.perf_counter() - started
            total_input_store_time += input_write_time

            job = ProofJob(
                inference_request_name=request.name,
                model_name=partition.name,
                inference_request_id=request.request_id,
                model_ref=model_ref,
                input_ref=input_ref,
                setup_cache_ref=setup_ref,
                output_prefix_ref=output_prefix_ref,
                profiling_file_path=resource_profile_path,
                model_write_time=model_write_time,
                input_write_time=input_write_time,
                profiling_data=profiling_data,
                predicted_duration_s=predicted_duration_s,
                predicted_peak_memory_gb=predicted_peak_memory_gb,
                artifact_key=setup_key,
                max_retries=max_retries,
                parent_model_hash=partition.parent_hash,
                model_hash=partition.sub_hash,
            )
            jobs.append(job)

        request.preparation_metrics["artifact_model_store_time(s)"] = total_model_store_time
        request.preparation_metrics["artifact_input_store_time(s)"] = total_input_store_time
        request.preparation_metrics["input_upload_wall_time(s)"] = total_input_store_time
        request.preparation_metrics["input_upload_service_time(s)"] = total_input_store_time
        request.preparation_metrics["input_upload_workers"] = 1

        self.logger.info(
            "Materialized %d jobs into setup/request storage in %.3fs model + %.3fs input",
            len(jobs),
            total_model_store_time,
            total_input_store_time,
        )
        return jobs

    def _load_or_split_model(self, request):
        # Model execution is needed only for request preparation, not for
        # coordinator recovery, result handling, or control-plane tests.
        from zkinfer.graph.onnx_splitter import split_onnx_model_with_inputs

        split_mode = (request.split_mode or "none").lower()
        self.logger.info(
            "Preparing request %s using split_mode=%s operators_per_partition=%s simplify_model=%s",
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
            **({"strict_boundaries": True} if self.witness_preparation == "upfront" else {}),
        )

    def _load_resource_profile(
        self,
        model_name: str,
        resource_profile_path: str,
        setup_store: SetupStoreConfig,
    ) -> Dict[str, Any]:
        if not exists(
            resource_profile_path,
            storage_type=setup_store.backend,
            s3_bucket=setup_store.s3_bucket,
        ):
            self.logger.info(
                "Resource profile not found for %s at %s; using conservative defaults",
                model_name,
                resource_profile_path,
            )
            return {}
        return load_json(
            resource_profile_path,
            storage_type=setup_store.backend,
            s3_bucket=setup_store.s3_bucket,
        )

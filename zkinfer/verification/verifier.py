from __future__ import annotations

import json
import tempfile
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List

from zkinfer.runtime.models import InferenceRequest
from zkinfer.storage import s3
from zkinfer.verification.request_manifest import (
    JobArtifactBundle,
    job_bundle_from_manifest_ref,
    manifest_refs_for_jobs,
    write_request_artifact_manifest,
)
from zkinfer.verification.split_chain import verify_split_chain


@dataclass(frozen=True)
class JobVerificationReport:
    job_id: str
    valid: bool
    verification_time_s: float
    proof_path: str
    settings_path: str
    vk_path: str
    srs_path: str
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "valid": self.valid,
            "verification_time_s": self.verification_time_s,
            "proof_path": self.proof_path,
            "settings_path": self.settings_path,
            "vk_path": self.vk_path,
            "srs_path": self.srs_path,
            "error": self.error,
        }


@dataclass(frozen=True)
class RequestVerificationReport:
    request_id: str
    enabled: bool
    valid: bool
    total_verification_time_s: float
    job_reports: List[Dict[str, Any]] = field(default_factory=list)
    split_chain: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "enabled": self.enabled,
            "valid": self.valid,
            "total_verification_time_s": self.total_verification_time_s,
            "job_reports": self.job_reports,
            "split_chain": self.split_chain,
            "error": self.error,
        }


class RuntimeVerifier:
    """Coordinator-side proof and split-boundary verifier."""

    def __init__(
        self,
        srs_dir: str,
        enabled: bool = True,
        require_split_chain: bool = False,
        write_request_manifest: bool = True,
        trusted_vk_manifest: str | None = None,
    ):
        self.srs_dir = Path(srs_dir).expanduser().resolve()
        self.enabled = bool(enabled)
        self.require_split_chain = bool(require_split_chain)
        self.write_request_manifest = bool(write_request_manifest)
        self.trusted_vk_manifest = Path(trusted_vk_manifest) if trusted_vk_manifest else None

    def verify_request(self, request: InferenceRequest) -> RequestVerificationReport:
        if not self.enabled:
            return RequestVerificationReport(
                request_id=request.request_id,
                enabled=False,
                valid=True,
                total_verification_time_s=0.0,
                error="verification disabled",
            )

        if request.preparation_metrics.get("witness_preparation_mode") == "upfront":
            return self._verify_prepared_request(request)

        started = time.perf_counter()
        refs = manifest_refs_for_jobs(request.proof_jobs)
        if not refs:
            return RequestVerificationReport(
                request_id=request.request_id,
                enabled=True,
                valid=False,
                total_verification_time_s=0.0,
                error="no job artifact manifests found",
            )

        bundles = [job_bundle_from_manifest_ref(ref) for ref in refs]
        job_reports = [self.verify_job_bundle(bundle).to_dict() for bundle in bundles]
        split_report = verify_split_chain(
            bundles,
            require_chain_witnesses=self.require_split_chain,
            srs_dir=self.srs_dir,
        ).to_dict()

        valid = all(item["valid"] for item in job_reports) and bool(
            split_report.get("valid", False)
        )
        report = RequestVerificationReport(
            request_id=request.request_id,
            enabled=True,
            valid=valid,
            total_verification_time_s=time.perf_counter() - started,
            job_reports=job_reports,
            split_chain=split_report,
        )
        if self.write_request_manifest and request.reports_dir:
            write_request_artifact_manifest(
                request=request,
                out_dir=request.reports_dir,
                verification=report.to_dict(),
                aggregate_metrics={},
            )
        return report

    def _verify_prepared_request(self, request: InferenceRequest) -> RequestVerificationReport:
        """Verify all proofs and their authenticated public boundary instances.

        The coordinator's immutable prepared descriptors, graph interface,
        settings and witness are the expected request. VK trust is either
        externally pinned or explicitly reported as trusted-worker mode.
        """
        from zkinfer.backends.ezkl.preparation import (
            PREPARED_INPUT_KIND, TensorSpec, check_public_graph, file_digest,
            public_tensors, read_json,
        )
        from zkinfer.runtime.data_plane import ArtifactRef

        started = time.perf_counter()
        reports = []
        split_report = {
            "valid": False,
            "method": "exact_public_with_deterministic_rescaling",
            "key_trust": "pinned" if self.trusted_vk_manifest else "trusted_workers",
        }
        error = ""
        try:
            pins = read_json(self.trusted_vk_manifest) if self.trusted_vk_manifest else None
            graph = {}
            seen_jobs = set()

            def checked_ref(ref):
                if ref.backend != "filesystem" or not ref.sha256:
                    raise ValueError("Missing trusted filesystem artifact digest")
                path = Path(ref.uri)
                if file_digest(path) != ref.sha256:
                    raise ValueError(f"Trusted artifact hash mismatch: {path}")
                return path

            def specs(values):
                return [
                    TensorSpec(v["name"], tuple(v["shape"]) if v["shape"] is not None else None)
                    for v in values
                ]

            if not request.proof_jobs:
                raise ValueError("No prepared jobs")
            for job in request.proof_jobs:
                if job.job_id in seen_jobs or job.model_name in graph:
                    raise ValueError("Duplicate job or partition")
                seen_jobs.add(job.job_id)
                if not job.artifact_manifest_ref:
                    raise ValueError(f"Missing proof artifact manifest for {job.job_id}")
                bundle = job_bundle_from_manifest_ref(job.artifact_manifest_ref)
                if (
                    bundle.request_id != request.request_id
                    or bundle.job_id != job.job_id
                    or bundle.setup_key != job.artifact_key
                    or bundle.backend != "filesystem"
                ):
                    raise ValueError("Proof artifact manifest does not match scheduled job")
                descriptor = read_json(checked_ref(job.input_ref))
                if (
                    descriptor.get("kind") != PREPARED_INPUT_KIND
                    or descriptor.get("request_id") != request.request_id
                    or descriptor.get("partition_name") != job.model_name
                    or descriptor.get("setup_key") != job.artifact_key
                ):
                    raise ValueError("Trusted prepared descriptor does not match job")
                settings_source = checked_ref(ArtifactRef.from_dict(descriptor["settings"]))
                expected = read_json(checked_ref(ArtifactRef.from_dict(descriptor["witness"])))
                with tempfile.TemporaryDirectory(prefix="zkinfer_exact_verify_") as temporary:
                    directory = Path(temporary)
                    proof_path = directory / "proof.pf"
                    settings_path = directory / "settings.json"
                    vk_path = directory / "vk.json"
                    shutil.copyfile(bundle.proof_path, proof_path)
                    shutil.copyfile(settings_source, settings_path)
                    shutil.copyfile(bundle.vk_path, vk_path)
                    if pins is not None:
                        expected_vk = pins.get(job.artifact_key)
                        if not expected_vk or file_digest(vk_path) != expected_vk:
                            raise ValueError(f"Untrusted verification key for {job.artifact_key}")
                    checked_bundle = replace(
                        bundle, proof_path=str(proof_path), settings_path=str(settings_path),
                        vk_path=str(vk_path),
                    )
                    report = self.verify_job_bundle(checked_bundle).to_dict()
                    reports.append(report)
                    if not report["valid"]:
                        raise ValueError(f"Invalid proof for {job.job_id}")
                    inputs, outputs = public_tensors(
                        read_json(proof_path), read_json(settings_path),
                        specs(descriptor["input_specs"]), specs(descriptor["output_specs"]),
                    )
                    # These values came from the proof that was just verified,
                    # not from the worker's unauthenticated witness file.
                    if [t.values for t in inputs] != expected["inputs"]:
                        raise ValueError(f"Public inputs differ from prepared witness: {job.job_id}")
                    if [t.values for t in outputs] != expected["outputs"]:
                        raise ValueError(f"Public outputs differ from prepared witness: {job.job_id}")
                    graph[job.model_name] = (inputs, outputs)
            split_report.update(check_public_graph(graph))
            split_report["valid"] = True
        except Exception as exc:
            error = str(exc)
            split_report["error"] = error
        report = RequestVerificationReport(
            request_id=request.request_id, enabled=True,
            valid=split_report["valid"] and len(reports) == len(request.proof_jobs),
            total_verification_time_s=time.perf_counter() - started,
            job_reports=reports, split_chain=split_report, error=error,
        )
        if self.write_request_manifest and request.reports_dir:
            write_request_artifact_manifest(
                request=request, out_dir=request.reports_dir,
                verification=report.to_dict(), aggregate_metrics={},
            )
        return report

    def verify_job_bundle(self, bundle: JobArtifactBundle) -> JobVerificationReport:
        with tempfile.TemporaryDirectory(prefix="zkinfer_verify_") as tmp:
            tmp_dir = Path(tmp)
            proof = self._materialize(
                bundle.backend, bundle.s3_bucket, bundle.proof_path, tmp_dir / "proof.pf"
            )
            settings = self._materialize(
                bundle.backend,
                bundle.s3_bucket,
                bundle.settings_path,
                tmp_dir / "settings.json",
            )
            vk = self._materialize(
                bundle.backend, bundle.s3_bucket, bundle.vk_path, tmp_dir / "vk.json"
            )
            srs_path = self._resolve_srs(settings)

            for path in (proof, settings, vk, srs_path):
                if not path.is_file():
                    return JobVerificationReport(
                        job_id=bundle.job_id,
                        valid=False,
                        verification_time_s=0.0,
                        proof_path=str(proof),
                        settings_path=str(settings),
                        vk_path=str(vk),
                        srs_path=str(srs_path),
                        error=f"missing verification input: {path}",
                    )

            started = time.perf_counter()
            try:
                import ezkl

                valid = bool(
                    ezkl.verify(
                        proof_path=str(proof),
                        settings_path=str(settings),
                        vk_path=str(vk),
                        srs_path=str(srs_path),
                    )
                )
                error = "" if valid else "ezkl.verify returned false"
            except Exception as exc:
                valid = False
                error = str(exc)

            return JobVerificationReport(
                job_id=bundle.job_id,
                valid=valid,
                verification_time_s=time.perf_counter() - started,
                proof_path=str(proof),
                settings_path=str(settings),
                vk_path=str(vk),
                srs_path=str(srs_path),
                error=error,
            )

    @staticmethod
    def _materialize(backend: str, bucket: str, uri: str, local_path: Path) -> Path:
        if backend == "filesystem":
            return Path(uri)
        if backend == "s3":
            if not bucket:
                raise ValueError("S3 artifact requires bucket")
            s3.download_file(bucket, uri, str(local_path))
            return local_path
        raise ValueError(f"Unsupported verification artifact backend: {backend}")

    def _resolve_srs(self, settings_path: Path) -> Path:
        with open(settings_path, "r", encoding="utf-8") as file:
            settings = json.load(file)
        run_args = settings.get("run_args", {})
        logrows = int(run_args["logrows"])
        commitment = str(run_args.get("commitment", "KZG")).lower()
        filename = f"ipa{logrows}.srs" if commitment == "ipa" else f"kzg{logrows}.srs"
        return self.srs_dir if self.srs_dir.is_file() else self.srs_dir / filename

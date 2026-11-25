"""Shared storage organised by setup lifetime and request lifetime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.storage import s3


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Expected a single storage path component: {value!r}")
    return value


class SharedStorage:
    """Reusable setup bundles and request-owned inputs/proof outputs.

    ArtifactRef remains the control-plane transport type; it does not determine
    the lifetime or placement of the bytes it references.
    """

    def __init__(self, backend: str, setup_root: str, requests_root: str, s3_bucket: str = ""):
        if backend not in {"filesystem", "s3"}:
            raise ValueError(f"Unsupported storage backend: {backend}")
        if backend == "s3" and not s3_bucket:
            raise ValueError("S3 storage requires a bucket")
        self.backend = backend
        self.setup_root = setup_root.rstrip("/")
        self.requests_root = requests_root.rstrip("/")
        self.s3_bucket = s3_bucket or ""
        if backend == "filesystem":
            for root in (self.setup_root, self.requests_root):
                Path(root).mkdir(parents=True, exist_ok=True)

    def prefix_ref(self, uri: str) -> ArtifactRef:
        return ArtifactRef(self.backend, uri, s3_bucket=self.s3_bucket)

    def partition_path(self, model_name: str, parent_hash: str, model_bytes: bytes, operator_label: str, *, bind: bool = True) -> str:
        """Readable filesystem namespace bound to full SHA-256 identities."""
        def slug(value):
            return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_- ")[:64] or "model"
        if not re.fullmatch(r"[0-9a-f]{64}", parent_hash):
            raise ValueError("Expected full SHA-256 parent identity")
        partition_hash = hashlib.sha256(model_bytes).hexdigest()
        if self.backend != "filesystem":
            return os.path.join(f"{slug(model_name)}_{parent_hash}",
                                f"{slug(operator_label)}_{partition_hash}")

        from zkinfer.storage.partition_cache import partition_lock

        parent = f"{slug(model_name)}_{parent_hash[:16]}"
        child = f"{slug(operator_label)}_{partition_hash[:16]}"
        # Bind the namespace before reading or publishing mutable artifacts.
        if not bind:
            return os.path.join(parent, child)
        for relative, digest in ((parent, parent_hash),
                                 (os.path.join(parent, child), partition_hash)):
            root = Path(self.setup_root) / relative
            with partition_lock(root):
                self._put_bytes(
                    (digest + "\n").encode("ascii"),
                    str(root / ".identity.sha256"),
                    validate_existing=True,
                )
        return os.path.join(parent, child)

    def setup_ref(self, setup_key: str, partition_path: str | None = None) -> ArtifactRef:
        root = self.setup_root
        if partition_path is not None:
            root = os.path.join(root, *[_component(part) for part in partition_path.split("/")])
        if partition_path is not None and self.backend == "filesystem":
            return self.prefix_ref(root)
        return self.prefix_ref(os.path.join(root, _component(setup_key)))

    def proof_prefix(self, request_id: str, partition_id: str) -> ArtifactRef:
        return self.prefix_ref(os.path.join(
            self.requests_root, _component(request_id), "proofs", _component(partition_id),
        ))

    @staticmethod
    def setup_key_for_model(
        model_bytes: bytes,
        setup_identity: Mapping[str, Any],
    ) -> str:
        """Return a stable setup identity for model bytes plus setup policy.

        The model hash alone is insufficient because settings, compiled circuits,
        and proving keys also depend on backend/setup semantics. The caller
        supplies a versioned, JSON-serialisable setup identity describing those
        semantics. Changing that identity creates a fresh setup bundle without
        requiring a manual setup-prefix change.
        """
        model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        payload = {
            "model_sha256": model_sha256,
            "setup_identity": dict(setup_identity),
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def put_model(self, model_bytes: bytes, setup_key: str | None = None,
                  partition_path: str | None = None) -> ArtifactRef:
        # setup_key identifies reusable proving state; ArtifactRef.sha256 verifies bytes.
        setup_key = setup_key or hashlib.sha256(model_bytes).hexdigest()
        root = (os.path.join(self.setup_root, *[_component(part) for part in partition_path.split("/")])
                if partition_path is not None else self.setup_ref(_component(setup_key)).uri)
        uri = os.path.join(root, "model.onnx")
        return self._put_bytes(model_bytes, uri, validate_existing=True)

    def put_input(self, request_id: str, partition_id: str, obj: Any) -> ArtifactRef:
        uri = os.path.join(
            self.requests_root, _component(request_id), "inputs",
            _component(partition_id) + ".json",
        )
        data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self._put_bytes(data, uri, validate_existing=True)

    def _put_bytes(self, data: bytes, uri: str, validate_existing: bool = False) -> ArtifactRef:
        digest = hashlib.sha256(data).hexdigest()
        if self.backend == "filesystem":
            if Path(uri).exists():
                if validate_existing and sha256_file(uri) != digest:
                    raise ValueError(f"Immutable storage object has different content: {uri}")
            else:
                Path(uri).parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=Path(uri).parent)
                try:
                    with os.fdopen(fd, "wb") as file:
                        file.write(data)
                    try:
                        os.link(tmp, uri)
                    except FileExistsError:
                        if validate_existing and sha256_file(uri) != digest:
                            raise ValueError(f"Immutable storage object has different content: {uri}")
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)
        elif s3.exists(self.s3_bucket, uri):
            if validate_existing and hashlib.sha256(s3.get_bytes(self.s3_bucket, uri)).hexdigest() != digest:
                raise ValueError(f"Immutable storage object has different content: {uri}")
        else:
            s3.put_bytes(data, self.s3_bucket, uri)
        return ArtifactRef(
            self.backend, uri, s3_bucket=self.s3_bucket, sha256=digest, size_bytes=len(data),
        )

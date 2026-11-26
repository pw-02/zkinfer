"""Resolve immutable filesystem artifacts without copying them."""
from pathlib import Path
import logging
import time

from zkinfer.storage import s3
from zkinfer.storage.partition_cache import (
    checked_digest, matches, metadata, partition_lock, publish_group,
)


class ArtifactTransfer:
    SETUP_FILES = ("settings.json", "network.compiled", "pk.json", "vk.json")
    KEY_FILES = ("pk.json", "vk.json")

    def __init__(self, workspace):
        self.workspace = Path(workspace)

    def materialize(self, ref, filename):
        if ref.is_empty:
            raise ValueError("Cannot materialize empty artifact")
        started = time.perf_counter()
        if ref.backend == "filesystem":
            path = Path(ref.uri)
            copied = 0
        elif ref.backend == "s3":
            self.workspace.mkdir(parents=True, exist_ok=True)
            path = self.workspace / filename
            s3.download_file(ref.s3_bucket, ref.uri, str(path))
            copied = path.stat().st_size
        else:
            raise ValueError(f"Unsupported backend: {ref.backend}")
        size = path.stat().st_size
        if ref.sha256 and checked_digest(path) != ref.sha256:
            raise ValueError(f"Artifact hash mismatch: {ref.uri}")
        return str(path), {
            "uri": ref.uri, "backend": ref.backend, "bytes_downloaded": copied,
            "bytes_referenced": size if ref.backend == "filesystem" else 0,
            "materialize_time(s)": time.perf_counter() - started,
            "local_path": str(path),
        }

    def restore_setup(self, ref, setup_key, directory, prepared, srs_digest=None):
        started = time.perf_counter()
        files = self.KEY_FILES if prepared else self.SETUP_FILES
        metrics = {"setup_key": setup_key, "shared_setup_uri": ref.uri,
                   "files": {}, "paths": {}, "bytes_downloaded": 0, "bytes_referenced": 0}
        if not ref.is_empty and ref.backend == "filesystem":
            with partition_lock(ref.uri) as root:
                info = metadata(root).get("keys", {})
                hashes = info.get("sha256", {})
                if (info.get("setup_key") == setup_key
                        and info.get("srs_sha256") == srs_digest
                        and set(hashes) == set(files) and matches(root, hashes)):
                    for name in files:
                        path = root / name
                        metrics["paths"][name] = str(path)
                        size = path.stat().st_size
                        metrics["bytes_referenced"] += size
                        metrics["files"][name] = {"bytes": size, "source": "shared_filesystem"}
        metrics["restore_time(s)"] = time.perf_counter() - started
        return metrics

    def publish_setup(self, ref, setup_key, directory, prepared, srs_digest=None, circuit_hashes=None):
        started = time.perf_counter()
        count = 0
        if not ref.is_empty and ref.backend == "filesystem":
            with partition_lock(ref.uri) as root:
                info = metadata(root)
                canonical = info.get("circuit", {}).get("sha256", {})
                if "keys" in info:
                    logging.getLogger(__name__).info("Key cache publication skipped: committed keys exist at %s", root)
                elif canonical and canonical != circuit_hashes:
                    logging.getLogger(__name__).warning("Key cache publication skipped: circuit hashes differ at %s", root)
                else:
                    files = self.KEY_FILES if prepared else self.SETUP_FILES
                    sources = {name: Path(directory) / name for name in files}
                    hashes = {name: checked_digest(path) for name, path in sources.items()}
                    count = sum(path.stat().st_size for path in sources.values())
                    publish_group(root, info, "keys",
                                  {"setup_key": setup_key, "sha256": hashes,
                                   "srs_sha256": srs_digest, "circuit_sha256": circuit_hashes}, sources)
        return {"bytes_uploaded": 0, "bytes_published": count,
                "publish_time(s)": time.perf_counter() - started}

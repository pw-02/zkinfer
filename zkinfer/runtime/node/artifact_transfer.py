"""Materialise artifacts into a single job workspace; no local cache or eviction."""
from pathlib import Path
import shutil
import time

from zkinfer.storage import s3
from zkinfer.storage.shared_store import sha256_file
from zkinfer.storage.partition_cache import (
    atomic_copy, matches, metadata, partition_lock, save_metadata,
)


class ArtifactTransfer:
    SETUP_FILES = ("settings.json", "network.compiled", "pk.json", "vk.json")
    KEY_FILES = ("pk.json", "vk.json")

    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def materialize(self, ref, filename):
        if ref.is_empty:
            raise ValueError("Cannot materialize empty artifact")
        started = time.perf_counter()
        destination = self.workspace / filename
        if ref.backend == "filesystem":
            shutil.copyfile(ref.uri, destination)
        elif ref.backend == "s3":
            s3.download_file(ref.s3_bucket, ref.uri, str(destination))
        else:
            raise ValueError(f"Unsupported backend: {ref.backend}")
        if ref.sha256 and sha256_file(destination) != ref.sha256:
            raise ValueError(f"Artifact hash mismatch: {ref.uri}")
        return str(destination), {
            "uri": ref.uri, "backend": ref.backend,
            "bytes_downloaded": destination.stat().st_size,
            "materialize_time(s)": time.perf_counter() - started,
            "local_path": str(destination),
        }

    def restore_setup(self, ref, setup_key, directory, prepared):
        started = time.perf_counter()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files = self.KEY_FILES if prepared else self.SETUP_FILES
        metrics = {"setup_key": setup_key, "shared_setup_uri": ref.uri,
                   "files": {}, "bytes_downloaded": 0}
        if not ref.is_empty and ref.backend == "filesystem":
            with partition_lock(ref.uri) as root:
                info = metadata(root)
                hashes = info.get("keys", {}).get("sha256", {})
                if (info.get("keys", {}).get("setup_key") == setup_key
                        and set(hashes) == set(files) and matches(root, hashes)):
                    for name in files:
                        shutil.copyfile(root / name, directory / name)
                        size = (directory / name).stat().st_size
                        metrics["bytes_downloaded"] += size
                        metrics["files"][name] = {"bytes": size, "source": "shared_filesystem"}
        elif not ref.is_empty and ref.backend == "s3":
            # Legacy independent-mode S3 bundles use immutable setup prefixes.
            # Require a complete manifest before downloading a bundle.
            prefix = ref.uri.rstrip("/")
            marker = prefix + "/metadata.json"
            if s3.exists(ref.s3_bucket, marker):
                info = s3.get_json(ref.s3_bucket, marker).get("keys", {})
                hashes = info.get("sha256", {})
                if info.get("setup_key") == setup_key and set(hashes) == set(files):
                    for name in files:
                        path = directory / name
                        s3.download_file(ref.s3_bucket, prefix + "/" + name, str(path))
                        if sha256_file(path) != hashes[name]:
                            raise ValueError(f"Shared setup hash mismatch: {name}")
                        metrics["bytes_downloaded"] += path.stat().st_size
        metrics["restore_time(s)"] = time.perf_counter() - started
        return metrics

    def publish_setup(self, ref, setup_key, directory, prepared):
        started = time.perf_counter()
        directory = Path(directory)
        files = self.KEY_FILES if prepared else self.SETUP_FILES
        hashes = {name: sha256_file(directory / name) for name in files}
        count = 0
        if not ref.is_empty and ref.backend == "filesystem":
            with partition_lock(ref.uri) as root:
                info = metadata(root)
                existing = info.get("keys", {})
                if not (existing.get("setup_key") == setup_key
                        and set(existing.get("sha256", {})) == set(files)
                        and matches(root, existing["sha256"])):
                    for name in files:
                        atomic_copy(directory / name, root / name)
                        count += (directory / name).stat().st_size
                    info["keys"] = {"setup_key": setup_key, "sha256": hashes}
                    save_metadata(root, info)
        elif not ref.is_empty and ref.backend == "s3":
            # Publish to an attempt-independent content-addressed bundle, with
            # a manifest last. Exact prepared requests remain filesystem-only.
            prefix = ref.uri.rstrip("/")
            for name in files:
                s3.upload_file(str(directory / name), ref.s3_bucket, prefix + "/" + name)
                count += (directory / name).stat().st_size
            s3.put_json({"keys": {"setup_key": setup_key, "sha256": hashes}},
                        ref.s3_bucket, prefix + "/metadata.json")
        return {"bytes_uploaded": count, "publish_time(s)": time.perf_counter() - started}

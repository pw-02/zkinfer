from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.storage import s3
from zkinfer.storage.shared_store import sha256_file


class NodeCache:
    """Bounded node-local cache shared by all prover subprocesses.

    Shared storage is canonical. This cache only keeps reusable local copies.
    Concurrent materialization and setup restoration are serialized per object or
    setup key so two local jobs do not download/generate the same artifact at the
    same time. Setup directories can be pinned while active jobs use them so
    eviction never removes in-use proving state.
    """

    SETUP_FILES = ("settings.json", "network.compiled", "pk.json", "vk.json")

    def __init__(self, root_dir: str, max_bytes: int, default_s3_bucket: str = ""):
        self.root_dir = Path(root_dir)
        self.objects_dir = self.root_dir / "objects"
        self.setup_dir_root = self.root_dir / "setup"
        self.max_bytes = max(0, int(max_bytes or 0))
        self.default_s3_bucket = default_s3_bucket or ""
        self.last_eviction: Dict[str, Any] = {"eviction_count": 0, "evicted_bytes": 0}

        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.setup_dir_root.mkdir(parents=True, exist_ok=True)

        self._state_lock = threading.RLock()
        self._object_locks: Dict[str, threading.Lock] = {}
        self._setup_locks: Dict[str, threading.RLock] = {}
        self._pin_counts: Dict[str, int] = {}

    def total_bytes(self) -> int:
        with self._state_lock:
            return self._path_size(self.root_dir)

    def cached_setup_keys(self) -> list[str]:
        with self._state_lock:
            keys = []
            for setup_dir in self.setup_dir_root.iterdir() if self.setup_dir_root.exists() else []:
                if not setup_dir.is_dir():
                    continue
                if all((setup_dir / name).exists() for name in self.SETUP_FILES):
                    keys.append(setup_dir.name)
            return sorted(set(keys))

    @staticmethod
    def _safe_setup_key(setup_key: str) -> str:
        return setup_key.replace("/", "_").replace("..", "_") or "unknown"

    def setup_dir(self, setup_key: str) -> str:
        path = self.setup_dir_root / self._safe_setup_key(setup_key)
        path.mkdir(parents=True, exist_ok=True)
        self._touch(path)
        return str(path)

    def setup_ready(self, setup_key: str) -> bool:
        path = Path(self.setup_dir(setup_key))
        return all((path / name).exists() for name in self.SETUP_FILES)

    def _object_lock(self, key: str) -> threading.Lock:
        with self._state_lock:
            return self._object_locks.setdefault(key, threading.Lock())

    def _setup_lock(self, setup_key: str) -> threading.RLock:
        safe_key = self._safe_setup_key(setup_key)
        with self._state_lock:
            return self._setup_locks.setdefault(safe_key, threading.RLock())

    def pin_setup(self, setup_key: str) -> str:
        with self._state_lock:
            path = Path(self.setup_dir(setup_key)).resolve()
            key = str(path)
            self._pin_counts[key] = self._pin_counts.get(key, 0) + 1
        return str(path)

    def unpin_setup(self, setup_key: str) -> None:
        path = self.setup_dir_root / self._safe_setup_key(setup_key)
        self._unpin_path(path)

    def _unpin_path(self, path: Path) -> None:
        key = str(path.resolve())
        with self._state_lock:
            count = self._pin_counts.get(key, 0)
            if count <= 1:
                self._pin_counts.pop(key, None)
            else:
                self._pin_counts[key] = count - 1

    @contextmanager
    def pinned_setup(self, setup_key: str) -> Iterator[str]:
        """Pin before checking/restoring setup state, atomically with eviction."""
        path = self.pin_setup(setup_key)
        try:
            yield path
        finally:
            self.unpin_setup(setup_key)
            self.evict_if_needed()

    def _object_path(self, ref: ArtifactRef, filename_hint: str) -> Path:
        digest = ref.sha256 or self._stable_ref_digest(ref)
        filename = os.path.basename(ref.uri) or filename_hint
        return self.objects_dir / digest[:2] / digest / filename

    @contextmanager
    def materialized_artifact(
        self, ref: ArtifactRef, filename_hint: str = "artifact"
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Keep an immutable object available until its consumer releases it.

        Pin the intended path before materialization so there is no interval
        between returning a path and acquiring protection from eviction.
        """
        path = self._object_path(ref, filename_hint)
        with self._state_lock:
            key = str(path.resolve())
            self._pin_counts[key] = self._pin_counts.get(key, 0) + 1
        try:
            yield self.materialize_with_metrics(ref, filename_hint)
        finally:
            self._unpin_path(path)
            self.evict_if_needed()

    @contextmanager
    def setup_guard(self, setup_key: str, pin: bool = True) -> Iterator[str]:
        """Serialize setup mutation for one key and optionally pin it."""
        lock = self._setup_lock(setup_key)
        with lock:
            if pin:
                path = self.pin_setup(setup_key)
            else:
                path = self.setup_dir(setup_key)
            try:
                yield path
            finally:
                if pin:
                    self.unpin_setup(setup_key)

    def materialize(self, ref: ArtifactRef, filename_hint: str = "artifact") -> str:
        path, _ = self.materialize_with_metrics(ref, filename_hint=filename_hint)
        return path

    def materialize_with_metrics(
        self, ref: ArtifactRef, filename_hint: str = "artifact"
    ) -> Tuple[str, Dict[str, Any]]:
        if ref.is_empty:
            raise ValueError("Cannot materialize empty ArtifactRef")

        digest = ref.sha256 or self._stable_ref_digest(ref)
        local_path = self._object_path(ref, filename_hint)
        with self._object_lock(digest):
            return self._materialize_at_path(ref, local_path)

    def materialize_model_with_metrics(
        self, ref: ArtifactRef, setup_key: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Store the ONNX beside its derived setup, never in generic objects/."""
        with self._setup_lock(setup_key):
            path = Path(self.setup_dir(setup_key)) / "model.onnx"
            return self._materialize_at_path(ref, path)

    def _materialize_at_path(self, ref: ArtifactRef, local_path: Path) -> Tuple[str, Dict[str, Any]]:
        if ref.is_empty:
            raise ValueError("Cannot materialize empty ArtifactRef")
        started = time.perf_counter()
        metrics: Dict[str, Any] = {
            "uri": ref.uri,
            "backend": ref.backend,
            "cache_hit": False,
            "bytes_from_local_cache": 0,
            "bytes_downloaded": 0,
            "materialize_time(s)": 0.0,
            "local_path": str(local_path),
        }

        if local_path.exists():
            if local_path.name == "model.onnx" and ref.sha256 and sha256_file(local_path) != ref.sha256:
                raise ValueError(f"Cached partition model hash mismatch: {local_path}")
            self._touch(local_path)
            metrics["cache_hit"] = True
            metrics["bytes_from_local_cache"] = local_path.stat().st_size
            metrics["materialize_time(s)"] = time.perf_counter() - started
            return str(local_path), metrics

        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(str(local_path) + f".{os.getpid()}.tmp")
        tmp_path.unlink(missing_ok=True)

        if ref.backend == "filesystem":
            source = Path(ref.uri)
            if not source.exists():
                raise FileNotFoundError(ref.uri)
            if source.resolve() != tmp_path.resolve():
                shutil.copy2(source, tmp_path)
        elif ref.backend == "s3":
            bucket = ref.s3_bucket or self.default_s3_bucket
            if not bucket:
                raise ValueError("S3 artifact ref requires a bucket")
            s3.download_file(bucket, ref.uri, str(tmp_path))
        else:
            raise ValueError(f"Unsupported artifact backend: {ref.backend}")

        if ref.sha256:
            actual = sha256_file(str(tmp_path))
            if actual != ref.sha256:
                tmp_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Hash mismatch for {ref.uri}: expected {ref.sha256}, got {actual}"
                )

        os.replace(tmp_path, local_path)
        self._touch(local_path)
        size_bytes = local_path.stat().st_size
        eviction = self.evict_if_needed(protected_paths=[local_path])
        metrics.update(
            bytes_downloaded=size_bytes,
            eviction_count=eviction["eviction_count"],
            evicted_bytes=eviction["evicted_bytes"],
            **{"materialize_time(s)": time.perf_counter() - started},
        )
        return str(local_path), metrics

    def restore_setup(self, shared_setup_ref: ArtifactRef, setup_key: str) -> str:
        path, _ = self.restore_setup_with_metrics(shared_setup_ref, setup_key)
        return path

    def restore_setup_with_metrics(
        self, shared_setup_ref: ArtifactRef, setup_key: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Restore reusable setup artifacts into the node cache."""
        started = time.perf_counter()
        metrics: Dict[str, Any] = {
            "setup_key": setup_key,
            "shared_setup_uri": shared_setup_ref.uri,
            "files": {},
            "bytes_from_local_cache": 0,
            "bytes_downloaded": 0,
            "pk_decompress_time(s)": 0.0,
            "restore_time(s)": 0.0,
        }

        with self._setup_lock(setup_key):
            # Complete filesystem bundles are immutable canonical setup, not
            # node-owned cache entries. Never copy, touch, or evict their files.
            if shared_setup_ref.backend == "filesystem" and not shared_setup_ref.is_empty:
                shared_dir = Path(shared_setup_ref.uri).resolve()
                if all(
                    (shared_dir / name).is_file() and (shared_dir / name).stat().st_size > 0
                    for name in self.SETUP_FILES
                ):
                    metrics["setup_dir"] = str(shared_dir)
                    metrics["shared_read_only"] = True
                    metrics["bytes_reused_shared"] = 0
                    for name in self.SETUP_FILES:
                        size = (shared_dir / name).stat().st_size
                        metrics["bytes_reused_shared"] += size
                        metrics["files"][name] = {
                            "cache_hit": False, "downloaded": False,
                            "bytes": size, "source": "shared_filesystem",
                        }
                    metrics["restore_time(s)"] = time.perf_counter() - started
                    return str(shared_dir), metrics

            local_setup_dir = Path(self.setup_dir(setup_key))
            metrics["setup_dir"] = str(local_setup_dir)

            if shared_setup_ref.is_empty:
                metrics["restore_time(s)"] = time.perf_counter() - started
                return str(local_setup_dir), metrics

            for filename in self.SETUP_FILES:
                local_path = local_setup_dir / filename
                file_metrics = {"cache_hit": False, "downloaded": False, "bytes": 0, "source": ""}

                if local_path.exists():
                    size = local_path.stat().st_size
                    file_metrics.update(cache_hit=True, bytes=size, source="node_cache")
                    metrics["bytes_from_local_cache"] += size
                    metrics["files"][filename] = file_metrics
                    continue

                if filename == "pk.json":
                    gz_path = Path(str(local_path) + f".{os.getpid()}.gz")
                    if self._copy_or_download_optional(shared_setup_ref, "pk.json.gz", str(gz_path)):
                        compressed_size = gz_path.stat().st_size
                        decompress_started = time.perf_counter()
                        with gzip.open(gz_path, "rb") as src, open(local_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        metrics["pk_decompress_time(s)"] += time.perf_counter() - decompress_started
                        gz_path.unlink(missing_ok=True)
                        file_metrics.update(
                            downloaded=True,
                            bytes=compressed_size,
                            source="shared_cache_compressed",
                            raw_bytes=local_path.stat().st_size,
                        )
                        metrics["bytes_downloaded"] += compressed_size
                        metrics["files"][filename] = file_metrics
                        continue

                downloaded = self._copy_or_download_optional(shared_setup_ref, filename, str(local_path))
                if downloaded:
                    size = local_path.stat().st_size
                    file_metrics.update(downloaded=True, bytes=size, source="shared_cache")
                    metrics["bytes_downloaded"] += size
                metrics["files"][filename] = file_metrics

            self._touch(local_setup_dir)
            eviction = self.evict_if_needed(protected_paths=[local_setup_dir])
            metrics["eviction_count"] = eviction["eviction_count"]
            metrics["evicted_bytes"] = eviction["evicted_bytes"]
            metrics["restore_time(s)"] = time.perf_counter() - started
            return str(local_setup_dir), metrics

    def publish_setup_with_metrics(
        self, local_setup_dir: str, shared_setup_ref: ArtifactRef
    ) -> Dict[str, Any]:
        """Publish reusable setup artifacts back to shared storage."""
        started = time.perf_counter()
        metrics: Dict[str, Any] = {
            "shared_setup_uri": shared_setup_ref.uri,
            "files": {},
            "bytes_uploaded": 0,
            "pk_compress_time(s)": 0.0,
            "publish_time(s)": 0.0,
        }
        if shared_setup_ref.is_empty:
            metrics["publish_time(s)"] = time.perf_counter() - started
            return metrics

        local_dir = Path(local_setup_dir)
        if not local_dir.exists():
            metrics["publish_time(s)"] = time.perf_counter() - started
            return metrics

        if shared_setup_ref.backend == "filesystem":
            import fcntl
            import tempfile
            destination = Path(shared_setup_ref.uri)
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / ".publish.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not all((local_dir / name).is_file() for name in self.SETUP_FILES):
                    raise ValueError("Cannot publish incomplete proving setup")
                for name in self.SETUP_FILES:
                    target = destination / name
                    if target.exists() and name in {"settings.json", "network.compiled"}:
                        if sha256_file(target) != sha256_file(local_dir / name):
                            raise ValueError(f"Shared setup mismatch: {target}")
                # A complete bundle is immutable. Never replace a key already in use.
                if not all((destination / name).is_file() for name in self.SETUP_FILES):
                    for name in self.SETUP_FILES:
                        source = local_dir / name
                        fd, temp = tempfile.mkstemp(prefix=".publish-", dir=destination)
                        os.close(fd)
                        try:
                            shutil.copyfile(source, temp)
                            os.replace(temp, destination / name)
                        finally:
                            Path(temp).unlink(missing_ok=True)
                        metrics["bytes_uploaded"] += source.stat().st_size
                metrics["publish_time(s)"] = time.perf_counter() - started
                return metrics

        for filename in self.SETUP_FILES:
            source = local_dir / filename
            if not source.exists():
                metrics["files"][filename] = {"uploaded": False, "bytes": 0, "reason": "missing"}
                continue

            if filename == "pk.json":
                compressed = local_dir / f"pk.json.{os.getpid()}.gz"
                compress_started = time.perf_counter()
                with open(source, "rb") as raw, gzip.open(compressed, "wb", compresslevel=6) as zipped:
                    shutil.copyfileobj(raw, zipped)
                metrics["pk_compress_time(s)"] += time.perf_counter() - compress_started
                compressed_size = compressed.stat().st_size
                try:
                    self._copy_or_upload(shared_setup_ref, "pk.json.gz", str(compressed))
                    metrics["files"][filename] = {
                        "uploaded": True,
                        "bytes": compressed_size,
                        "raw_bytes": source.stat().st_size,
                        "artifact": "pk.json.gz",
                    }
                    metrics["bytes_uploaded"] += compressed_size
                finally:
                    compressed.unlink(missing_ok=True)
            else:
                size = source.stat().st_size
                self._copy_or_upload(shared_setup_ref, filename, str(source))
                metrics["files"][filename] = {"uploaded": True, "bytes": size, "artifact": filename}
                metrics["bytes_uploaded"] += size

        eviction = self.evict_if_needed(protected_paths=[local_dir])
        metrics["eviction_count"] = eviction["eviction_count"]
        metrics["evicted_bytes"] = eviction["evicted_bytes"]
        metrics["publish_time(s)"] = time.perf_counter() - started
        return metrics

    def evict_if_needed(self, protected_paths: Optional[Iterable[Path]] = None) -> Dict[str, int]:
        with self._state_lock:
            if self.max_bytes <= 0:
                self.last_eviction = {"eviction_count": 0, "evicted_bytes": 0}
                return self.last_eviction

            protected = {str(Path(p).resolve()) for p in (protected_paths or [])}
            protected.update(path for path, count in self._pin_counts.items() if count > 0)
            eviction_count = 0
            evicted_bytes = 0

            while self._path_size(self.root_dir) > self.max_bytes:
                candidates = []
                for child in list(self.objects_dir.glob("*/*")) + list(self.setup_dir_root.glob("*")):
                    if not child.exists():
                        continue
                    resolved = str(child.resolve())
                    if any(
                        resolved == item
                        or resolved.startswith(item + os.sep)
                        or item.startswith(resolved + os.sep)
                        for item in protected
                    ):
                        continue
                    try:
                        stat = child.stat()
                    except FileNotFoundError:
                        continue
                    candidates.append((stat.st_mtime, child))

                if not candidates:
                    break

                _, victim = min(candidates, key=lambda item: item[0])
                victim_bytes = self._path_size(victim)
                if victim.is_dir():
                    shutil.rmtree(victim, ignore_errors=True)
                else:
                    victim.unlink(missing_ok=True)
                eviction_count += 1
                evicted_bytes += victim_bytes

            self.last_eviction = {"eviction_count": eviction_count, "evicted_bytes": evicted_bytes}
            return self.last_eviction

    def _copy_or_download_optional(self, prefix_ref: ArtifactRef, filename: str, local_path: str) -> bool:
        if prefix_ref.backend == "filesystem":
            source = Path(prefix_ref.uri) / filename
            if not source.exists():
                return False
            destination = Path(local_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            return True

        if prefix_ref.backend == "s3":
            bucket = prefix_ref.s3_bucket or self.default_s3_bucket
            key = f"{prefix_ref.uri.rstrip('/')}/{filename}"
            if not bucket or not s3.exists(bucket, key):
                return False
            s3.download_file(bucket, key, local_path)
            return True

        return False

    def _copy_or_upload(self, prefix_ref: ArtifactRef, filename: str, local_path: str) -> None:
        if prefix_ref.backend == "filesystem":
            destination = Path(prefix_ref.uri) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = Path(local_path)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            return

        if prefix_ref.backend == "s3":
            bucket = prefix_ref.s3_bucket or self.default_s3_bucket
            if not bucket:
                raise ValueError("S3 setup publication requires bucket")
            key = f"{prefix_ref.uri.rstrip('/')}/{filename}"
            s3.upload_file(local_path, bucket, key)
            return

        raise ValueError(f"Unsupported setup backend: {prefix_ref.backend}")

    @staticmethod
    def _path_size(path: Path) -> int:
        if path.is_file():
            try:
                return path.stat().st_size
            except FileNotFoundError:
                return 0
        total = 0
        for item in path.rglob("*") if path.exists() else []:
            if item.is_file():
                try:
                    total += item.stat().st_size
                except FileNotFoundError:
                    continue
        return total

    @staticmethod
    def _stable_ref_digest(ref: ArtifactRef) -> str:
        raw = f"{ref.backend}|{ref.s3_bucket}|{ref.uri}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _touch(path: Path) -> None:
        now = time.time()
        try:
            os.utime(path, (now, now))
        except FileNotFoundError:
            pass



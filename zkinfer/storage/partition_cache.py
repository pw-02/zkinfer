"""Write-once partition artifacts; metadata commits complete groups under a lock."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
from functools import lru_cache

from zkinfer.storage.shared_store import sha256_file


@contextmanager
def partition_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield root


def metadata(root):
    path = Path(root) / "metadata.json"
    if not path.exists():
        return {}
    with path.open() as file:
        return json.load(file)


def atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".publish-", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, destination)
    finally:
        Path(temp).unlink(missing_ok=True)


def save_metadata(root, value):
    fd, temp = tempfile.mkstemp(prefix=".metadata-", dir=root)
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(value, file, sort_keys=True, indent=2)
        os.replace(temp, Path(root) / "metadata.json")
    finally:
        Path(temp).unlink(missing_ok=True)


@lru_cache(maxsize=4096)
def _checked_digest(path, signature):
    return sha256_file(path)


def checked_digest(path):
    """Avoid rehashing immutable files on every job in a long-lived process."""
    path = Path(path).resolve()
    st = path.stat()
    signature = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    return _checked_digest(str(path), signature)


def publish_group(root, info, group, entry, sources):
    """Commit a group once; caller holds partition_lock.

    Sources must be staged on the same filesystem. A crash before metadata
    publication leaves uncommitted files that the next publisher may replace.
    """
    if group in info:
        return False
    for name, source in sources.items():
        if Path(source).stat().st_size == 0:
            raise ValueError(f"Empty artifact: {source}")
    for name, source in sources.items():
        os.chmod(source, 0o444)
        os.replace(source, Path(root) / name)
    info[group] = entry
    save_metadata(root, info)
    return True


def matches(root, hashes):
    return bool(hashes) and all(
        (Path(root) / name).is_file() and checked_digest(Path(root) / name) == digest
        for name, digest in hashes.items()
    )

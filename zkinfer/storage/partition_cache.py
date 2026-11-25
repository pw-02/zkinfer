"""One mutable shared partition slot, with locked, private request snapshots."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

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


def matches(root, hashes):
    return bool(hashes) and all(
        (Path(root) / name).is_file() and sha256_file(Path(root) / name) == digest
        for name, digest in hashes.items()
    )

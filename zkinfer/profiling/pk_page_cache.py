"""Optional Linux proving-key page-cache control for isolated experiments."""
import logging
import os
import shutil
import subprocess
import sys
import time


def cached_bytes(path):
    """Return fincore's resident bytes, or None when unavailable."""
    tool = shutil.which("fincore")
    if not tool:
        return None
    try:
        proc = subprocess.run(
            [tool, "--bytes", "--noheadings", "--output", "RES", "--", os.fspath(path)],
            capture_output=True, text=True, check=True, timeout=60,
        )
        value = int(proc.stdout.strip())
        return value if value >= 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def prepare_pk_page_cache(path, *, evict=False, logger=None):
    """Flush and advise eviction only when requested; never drop global caches."""
    logger = logger or logging.getLogger(__name__)
    metrics = {
        "pk_page_cache_eviction_requested": evict,
        "pk_page_cache_eviction_status": "disabled",
    }
    # Keep normal runtime behavior unchanged (including no extra probe in basic mode).
    if not evict and os.environ.get("ZKINFER_PROFILING_MODE") == "basic":
        return metrics
    if evict and (sys.platform != "linux" or not hasattr(os, "posix_fadvise")
                  or not hasattr(os, "POSIX_FADV_DONTNEED")):
        raise RuntimeError("PK page-cache eviction requires Linux posix_fadvise")
    metrics["pk_page_cache_file_bytes"] = os.path.getsize(path)
    probe_start = time.perf_counter()
    before = cached_bytes(path)
    probe_time = time.perf_counter() - probe_start
    metrics["pk_page_cache_cached_bytes_before"] = before
    if evict:
        start = time.perf_counter()
        try:
            # Read-only access also supports immutable cached keys on Linux.
            with open(path, "rb") as stream:
                fd = stream.fileno()
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError as exc:
            raise RuntimeError(f"Failed to flush/advise PK page-cache eviction: {path}") from exc
        metrics["pk_page_cache_eviction_time(s)"] = time.perf_counter() - start
        metrics["pk_page_cache_eviction_status"] = "advised"
        probe_start = time.perf_counter()
        after = cached_bytes(path)
        probe_time += time.perf_counter() - probe_start
        metrics["pk_page_cache_cached_bytes_after"] = after
        logger.info("PK page-cache eviction advised: before=%s after=%s file=%s bytes",
                    before, after, metrics["pk_page_cache_file_bytes"])
    metrics["pk_page_cache_probe_time(s)"] = probe_time
    return metrics

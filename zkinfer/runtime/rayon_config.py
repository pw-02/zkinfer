"""Apply per-process Rayon settings before the native global pool initializes."""
import os


def rayon_environment(threads, base=None):
    env = dict(os.environ if base is None else base)
    # Null means automatic detection, not an inherited explicit override.
    env.pop("RAYON_NUM_THREADS", None)
    env.pop("RAYON_RS_NUM_CPUS", None)
    if threads is not None:
        env["RAYON_NUM_THREADS"] = str(threads)
    return env


def configure_coordinator_rayon(threads):
    env = rayon_environment(threads)
    for key in ("RAYON_NUM_THREADS", "RAYON_RS_NUM_CPUS"):
        if key in env:
            os.environ[key] = env[key]
        else:
            os.environ.pop(key, None)


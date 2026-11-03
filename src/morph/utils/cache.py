"""Small disk caches shared by independent Morph processes."""

import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def cache_root() -> Path:
    """Use MORPH_CACHE_DIR, otherwise the user's XDG cache directory."""
    default = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "morph"
    return Path(os.environ.get("MORPH_CACHE_DIR", default)).expanduser()


@contextmanager
def file_lock(path):
    """Serialize writers and first-use readers of a shared cache file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{path}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def atomic_write(path: Path, data: bytes) -> None:
    """Replace a cache file only after its complete contents have been written."""
    with tempfile.TemporaryDirectory(dir=path.parent) as staging:
        temporary = Path(staging) / path.name
        temporary.write_bytes(data)
        os.replace(temporary, path)

"""Download the original LibINVENT rules only when computing new cuts."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

from morph.utils.cache import atomic_write, cache_root, file_lock

# Byte-identical to the rules used for the published scaffold caches.
_REACTIONS = (
    "https://raw.githubusercontent.com/MolecularAI/Lib-INVENT-dataset/"
    "540c1b015a091d3b79f57505168017baeefe10a5/tutorial/data/reaction.smirks",
    "0eaf5444865fef1c5c6d5b87ec97f9480a4765fde97cfa6dbbe57d718f9f74ef",
)
_FILTERS = (
    "https://raw.githubusercontent.com/biomed-AI/DiffDec/"
    "916ae14207b2783a90336bb8509374535c5791f9/data/filter_conditions.json",
    "ac33112e64f47991be40259aa694219a7f1d47f36a4a130cf329aa98205529b2",
)


def _download_resource(url: str, sha256: str) -> Path:
    path = cache_root() / "libinvent" / sha256 / url.rsplit("/", 1)[-1]
    with file_lock(path):
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
            return path
        try:
            with urlopen(url, timeout=30) as response:
                data = response.read()
        except OSError as exc:
            raise OSError(
                f"Cannot download LibINVENT resource {url}. Run once with network "
                f"access to populate {path}, then the cached rules work offline."
            ) from exc
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError(f"LibINVENT resource failed SHA-256 verification: {url}")
        atomic_write(path, data)
    return path


def libinvent_resources() -> tuple[Path, Path]:
    """Return verified reaction/filter files, fetching pinned sources if missing."""
    return _download_resource(*_REACTIONS), _download_resource(*_FILTERS)

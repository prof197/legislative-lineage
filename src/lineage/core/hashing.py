"""SHA-256 digests of files, as the reports, manifests and releases record them."""

import hashlib
from pathlib import Path

_CHUNK = 1 << 20


def file_sha256(path: Path) -> str:
    """The file's SHA-256, read a megabyte at a time, so a release asset of any size fits."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()

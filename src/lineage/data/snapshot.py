"""The pinned Open States snapshot: which archives, and exactly which bytes.

Archives are generated independently, so the snapshot is a manifest of archives, each with its
own Generated At date, not one dated download. See DESIGN.md#data-snapshot.
"""

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

# The only export format the ingest schema in `queries` was written against.
SUPPORTED_FORMAT = "1.0"
_MANIFEST_VERSION = 1


class SnapshotError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Archive:
    path: str  # relative to the raw directory, POSIX separators
    state: str
    session: str
    generated_at: str  # verbatim from the README; it carries no time zone
    bytes: int
    sha256: str


def _readme_fields(z: zipfile.ZipFile) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in z.read("README").decode().splitlines():
        key, sep, value = line.partition(": ")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def bills_member(z: zipfile.ZipFile) -> str:
    """Name of the archive's single bills file."""
    members = [n for n in z.namelist() if n.endswith("_bills.json")]
    if len(members) != 1:
        raise SnapshotError(f"{z.filename}: expected one *_bills.json member, found {members}")
    return members[0]


def read_archive(raw_dir: Path, path: Path) -> Archive:
    with zipfile.ZipFile(path) as z:
        fields = _readme_fields(z)
    rel = path.relative_to(raw_dir).as_posix()
    try:
        version = fields["JSON Format Version"]
        archive = Archive(
            path=rel,
            state=fields["State"],
            session=fields["Session"],
            generated_at=fields["Generated At"],
            bytes=path.stat().st_size,
            sha256=_sha256(path),
        )
    except KeyError as missing:
        raise SnapshotError(f"{rel}: README lacks {missing}") from None
    if version != SUPPORTED_FORMAT:
        raise SnapshotError(f"{rel}: export format version {version} is not {SUPPORTED_FORMAT}")
    return archive


def scan(raw_dir: Path) -> list[Archive]:
    return [read_archive(raw_dir, p) for p in sorted(raw_dir.glob("*/*.zip"))]


def write_manifest(archives: list[Archive], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"manifest_version": _MANIFEST_VERSION, "archives": [asdict(a) for a in archives]}
    path.write_text(json.dumps(doc, indent=1) + "\n")


def load_manifest(path: Path) -> list[Archive]:
    doc = json.loads(path.read_text())
    if doc.get("manifest_version") != _MANIFEST_VERSION:
        raise SnapshotError(f"{path}: unsupported manifest version {doc.get('manifest_version')}")
    return [Archive(**a) for a in doc["archives"]]


def verify(raw_dir: Path, pinned: list[Archive]) -> None:
    """Raise SnapshotError naming every archive that is missing, changed or not in the manifest."""
    on_disk = {a.path: a for a in scan(raw_dir)}
    expected = {a.path: a for a in pinned}
    problems = [
        *(f"missing: {p}" for p in sorted(expected.keys() - on_disk.keys())),
        *(f"unlisted: {p}" for p in sorted(on_disk.keys() - expected.keys())),
        *(
            f"changed: {p}"
            for p in sorted(expected.keys() & on_disk.keys())
            if expected[p] != on_disk[p]
        ),
    ]
    if problems:
        raise SnapshotError("raw archives do not match the manifest:\n  " + "\n  ".join(problems))

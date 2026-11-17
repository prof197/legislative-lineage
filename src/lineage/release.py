"""The data release the deploy builds from: packing the export, its checksum manifest, pinning
(DESIGN.md#export, "The data release").

Data never enters git, so the site is built from a GitHub Release asset,
`export-<ST>-<tag>.tar.gz`, whose SHA-256 is committed in `site/data-release.json`. Inside the
tarball, `checksums.json` lists every other file's path, size and SHA-256, so a reader can
check any one file, `site/scripts/fetch-data.sh` can check every unpacked file, and two
releases' manifests show exactly which files changed.

The tarball is byte-for-byte reproducible: members sorted by path, every timestamp, owner and
mode fixed, and no name or time in the gzip header, so the same export always gives the same
digest. That holds on one zlib; another zlib's deflate may compress the same tar differently,
which the manifest's per-file digests are indifferent to.
"""

import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from lineage.disk import DEFAULT_FLOOR_BYTES, require_free_space
from lineage.export_schema import Release

# The data release the deploy builds from, committed: its tag and the asset's expected SHA-256.
# A data change reaches the site only as a reviewed commit to this file.
RELEASE_MANIFEST = Path("site/data-release.json")
# The checksum manifest inside the tarball; never a file the export writes.
CHECKSUMS = "checksums.json"

# The tag and asset naming site/scripts/fetch-data.sh accepts.
_STATE = re.compile(r"[A-Z]{2}")
_TAG = r"data-{state}-[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}"


class ReleaseError(Exception):
    pass


class FileSum(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str  # relative to the export's root, `/`-separated
    size: int
    sha256: str


class Checksums(BaseModel):
    """`checksums.json`: every other file in the release, sorted by path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    files: list[FileSum]


def checksums_bytes(checksums: Checksums) -> bytes:
    """The manifest's one rendering: one file to a line, fields in model order, so a reader can
    find a file's entry with grep and a diff of two releases' manifests is a line per file."""
    lines = ",\n".join(json.dumps(f.model_dump()) for f in checksums.files)
    return f'{{"files": [\n{lines}\n]}}\n'.encode()


def sum_export(export: Path) -> Checksums:
    """Every file in `export`, summed, sorted by path."""
    if not (export / "meta.json").is_file():
        raise ReleaseError(f"{export} holds no meta.json; run `uv run lineage export` first")
    files = sorted(
        ((p.relative_to(export).as_posix(), p) for p in export.rglob("*") if p.is_file()),
        key=lambda e: e[0],
    )
    if any(name == CHECKSUMS for name, _ in files):
        raise ReleaseError(f"{export} already holds {CHECKSUMS}, which packing writes")
    sums: list[FileSum] = []
    for name, path in files:
        data = path.read_bytes()
        sums.append(FileSum(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return Checksums(files=sums)


def asset_name(state: str, tag: str) -> str:
    """The release asset's name, refused unless the tag is one of `state`'s data release tags,
    `data-<st>-YYYY-MM-DD`, the only naming the deploy's fetch accepts."""
    if not _STATE.fullmatch(state):
        raise ReleaseError(f"not a state code: {state}")
    if not re.fullmatch(_TAG.format(state=state.lower()), tag):
        raise ReleaseError(
            f"{tag} is not a data release tag for {state}, data-{state.lower()}-YYYY-MM-DD"
        )
    return f"export-{state}-{tag}.tar.gz"


def _bound(checksums: Checksums, manifest: bytes) -> int:
    # Each member's header, a PAX header in case, padding to 512; the tar's end; gzip's
    # worst-case growth on stored blocks.
    raw = sum(f.size for f in checksums.files) + len(manifest)
    raw += (len(checksums.files) + 1) * 3 * 512 + 20 * 512
    return raw + raw // 100 + (1 << 16)


def _member(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    return info


def package(
    export: Path, *, state: str, tag: str, out_dir: Path, floor: int = DEFAULT_FLOOR_BYTES
) -> tuple[Path, Path]:
    """The export as one release asset, `export-<ST>-<tag>.tar.gz`, with `checksums.json`
    inside, and its SHA-256 beside it in the form `sha256sum -c` reads. Nothing is written
    unless the tag is one the deploy accepts, and the asset appears whole or not at all."""
    name = asset_name(state, tag)
    checksums = sum_export(export)
    manifest = checksums_bytes(checksums)
    members: dict[str, FileSum | bytes] = {f.path: f for f in checksums.files}
    members[CHECKSUMS] = manifest
    out_dir.mkdir(parents=True, exist_ok=True)
    require_free_space(out_dir, needed=_bound(checksums, manifest), floor=floor)
    asset = out_dir / name
    part = out_dir / f"{name}.part"
    try:
        # The name is given empty so the gzip header carries none, not the part file's.
        with (
            part.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz,
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
        ):
            for member in sorted(members):
                _add(tar, export, member, members[member])
        part.replace(asset)
    finally:
        part.unlink(missing_ok=True)
    checksum = out_dir / f"{name}.sha256"
    checksum.write_text(f"{_sha256(asset)}  {name}\n")
    return asset, checksum


def _add(tar: tarfile.TarFile, export: Path, name: str, member: FileSum | bytes) -> None:
    if isinstance(member, bytes):
        data = member
    else:
        data = (export / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != member.sha256:
            raise ReleaseError(f"{name} changed while the release was packed")
    tar.addfile(_member(name, len(data)), io.BytesIO(data))


def read_checksums(asset: Path) -> Checksums:
    """The checksum manifest inside a release asset."""
    with tarfile.open(asset) as tar:
        try:
            f = tar.extractfile(CHECKSUMS)
        except KeyError as e:
            raise ReleaseError(f"{asset.name} holds no {CHECKSUMS}") from e
        assert f is not None
        return Checksums.model_validate_json(f.read(), strict=True)


def changed_files(old: Checksums, new: Checksums) -> dict[str, list[str]]:
    """Which files a new release added, changed or removed against an older one, by path."""
    before = {f.path: f for f in old.files}
    after = {f.path: f for f in new.files}
    return {
        "added": sorted(after.keys() - before.keys()),
        "changed": sorted(p for p in after.keys() & before.keys() if after[p] != before[p]),
        "removed": sorted(before.keys() - after.keys()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_release_manifest(asset: Path, *, tag: str, path: Path = RELEASE_MANIFEST) -> Release:
    """Pin `asset` as the release the deploy builds from: the tag, the asset's name and its
    SHA-256, computed from the asset itself. The deploy checks the download against this
    committed digest, not against a checksum file published beside the asset, which anyone able
    to replace the asset could replace too."""
    if not re.fullmatch(rf"export-[A-Z]{{2}}-{re.escape(tag)}\.tar\.gz", asset.name):
        raise ReleaseError(f"{asset.name} is not release {tag}'s asset")
    release = Release(tag=tag, asset=asset.name, sha256=_sha256(asset))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(release.model_dump(), indent=1) + "\n")
    return release

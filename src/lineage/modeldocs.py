"""Model legislation as documents: each source's manifest entry, each collected document, and the
texts kept outside git, one directory per source.

"ModelDoc" names a published model act or bill, to keep it apart from the embedding model and
the model calls elsewhere in this package. A source is an organization's public library of model
legislation, described by what kind of organization it is, never by a reading of its politics,
and a leaked archive is annotated as leaked. A document's text is stored once per version under
its SHA-256, so a revised text replaces the earlier one without erasing it, as bill records do
(DESIGN.md#updates). Texts are never committed, redistributed or shown; only what is recorded
here, and the share of wording a bill matches, reaches the site. See DESIGN.md#model-documents.
"""

import hashlib
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lineage.paths import DATA

MODELDOCS_DIR = DATA / "modeldocs"
SLUG = r"^[a-z0-9]+(-[a-z0-9]+)*$"
DOCS_FILE = "docs.jsonl"
VERSIONS_FILE = "versions.jsonl"


class ModelDocError(Exception):
    pass


class Source(BaseModel):
    """A library of model legislation, as its collector module declares it."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    key: str = Field(pattern=SLUG)
    organization: str = Field(min_length=1)
    # What kind of organization it is, from public filings: "nonprofit membership organization
    # of state legislators, 501(c)(3)". Never a political label.
    descriptor: str = Field(min_length=1)
    library_url: str = Field(pattern=r"^https://")
    terms_url: str | None = None
    terms_note: str = Field(min_length=1)  # what the terms allow, as read on `terms_read_on`
    terms_read_on: date
    leaked: bool = False
    leak_note: str | None = None  # how the documents became public, when leaked

    @model_validator(mode="after")
    def _a_leaked_source_says_how(self) -> Self:
        if self.leaked != (self.leak_note is not None):
            raise ValueError("a leaked source needs a leak_note, and only a leaked one has one")
        return self


class ModelDoc(BaseModel):
    """One collected document, its current version. Its text is at `text_path`."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*/[a-z0-9]+(-[a-z0-9]+)*$")  # source/slug
    source: str = Field(pattern=SLUG)
    title: str = Field(min_length=1)
    url: str = Field(pattern=r"^https?://")
    published: date | None = None  # as the organization dates it, when it does
    collected_on: date
    robots: Literal["allowed", "no_robots_txt"]  # a URL robots.txt disallows is never collected
    leaked: bool
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")  # the bytes as fetched
    media_type: str
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")  # the cleaned text matching reads

    @model_validator(mode="after")
    def _its_id_names_its_source(self) -> Self:
        if self.id.partition("/")[0] != self.source:
            raise ValueError(f"{self.id} is not in source {self.source}")
        return self


class ModelVersion(BaseModel):
    """One text a document has had, and the days it stood."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    doc_id: str
    text_sha256: str
    first_seen: date
    replaced_on: date | None = None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_dir(source: str, root: Path = MODELDOCS_DIR) -> Path:
    return root / source


def text_path(doc: ModelDoc, root: Path = MODELDOCS_DIR) -> Path:
    return source_dir(doc.source, root) / "texts" / f"{doc.text_sha256}.txt"


def raw_path(doc: ModelDoc, root: Path = MODELDOCS_DIR) -> Path:
    return source_dir(doc.source, root) / "raw" / doc.raw_sha256


def _write_once(path: Path, data: bytes) -> None:
    """Content-addressed files are written once; an existing one already holds these bytes."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    partial.replace(path)


def _write_lines(path: Path, rows: Sequence[BaseModel]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    partial.replace(path)


def read_docs(source: Source, root: Path = MODELDOCS_DIR) -> list[ModelDoc]:
    """A source's current documents, by id."""
    path = source_dir(source.key, root) / DOCS_FILE
    if not path.exists():
        return []
    docs = [ModelDoc.model_validate_json(line) for line in path.read_text().splitlines() if line]
    return sorted(docs, key=lambda d: d.id)


def read_versions(source: Source, root: Path = MODELDOCS_DIR) -> list[ModelVersion]:
    path = source_dir(source.key, root) / VERSIONS_FILE
    if not path.exists():
        return []
    return [
        ModelVersion.model_validate_json(line) for line in path.read_text().splitlines() if line
    ]


def store(
    source: Source, doc: ModelDoc, *, raw: bytes, text: str, root: Path = MODELDOCS_DIR
) -> bool:
    """Record a collected document and keep its bytes and text; returns whether anything changed.
    Collecting the same text again changes nothing; a new text becomes the current version and
    closes the one before it, whose text stays on disk."""
    if doc.source != source.key:
        raise ModelDocError(f"{doc.id} is not in source {source.key}")
    if doc.leaked != source.leaked:
        raise ModelDocError(f"{doc.id}: leaked must match its source's")
    if sha256(raw) != doc.raw_sha256 or sha256(text.encode()) != doc.text_sha256:
        raise ModelDocError(f"{doc.id}: the hashes do not match the bytes and text given")
    current = {d.id: d for d in read_docs(source, root)}
    versions = read_versions(source, root)
    was = current.get(doc.id)
    if was is not None and was.text_sha256 == doc.text_sha256:
        return False
    _write_once(raw_path(doc, root), raw)
    _write_once(text_path(doc, root), text.encode())
    if was is not None:
        versions = [
            v.model_copy(update={"replaced_on": doc.collected_on})
            if v.doc_id == doc.id and v.replaced_on is None
            else v
            for v in versions
        ]
    versions.append(
        ModelVersion(doc_id=doc.id, text_sha256=doc.text_sha256, first_seen=doc.collected_on)
    )
    current[doc.id] = doc
    directory = source_dir(source.key, root)
    directory.mkdir(parents=True, exist_ok=True)
    _write_lines(directory / VERSIONS_FILE, versions)
    _write_lines(directory / DOCS_FILE, [current[k] for k in sorted(current)])
    return True


def load_text(doc: ModelDoc, root: Path = MODELDOCS_DIR) -> str:
    """A document's cleaned text, checked against its recorded hash."""
    data = text_path(doc, root).read_bytes()
    if sha256(data) != doc.text_sha256:
        raise ModelDocError(f"{doc.id}: stored text does not match its hash")
    return data.decode()

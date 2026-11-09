"""Embedding models: the ONNX files the semantic stage runs, pinned and fetched once.

Each candidate is pinned to a Hugging Face repository at a commit, and each of its files to a
hash: the SHA-256 Hugging Face records for a large file, the git blob id for a small one. The
files are fetched through the project's own HTTP client (lineage.fetch), paced like every other
request, into `data/models/<name>/`, and a file lands only if its hash matches. fastembed then
loads the model from that directory and never touches the network. The repositories are the
ONNX exports fastembed itself uses. See DESIGN.md#semantic-stage.
"""

import hashlib
import urllib.error
from dataclasses import dataclass
from pathlib import Path

from lineage.disk import DEFAULT_FLOOR_BYTES
from lineage.fetch import FetchError, download

MODELS_DIR = Path("data/models")


@dataclass(frozen=True, slots=True)
class ModelFile:
    name: str
    size: int
    sha256: str | None = None  # large files
    git_blob: str | None = None  # small files: SHA-1 of "blob <size>\0" + content


@dataclass(frozen=True, slots=True)
class Model:
    name: str
    repo: str
    revision: str
    license: str
    dim: int
    max_tokens: int  # the input length the model was trained to, past which fastembed truncates
    onnx: str
    fastembed_name: str  # the name fastembed knows it by, which sets its pooling
    files: tuple[ModelFile, ...]


_TOKENIZER_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

MODELS = {
    m.name: m
    for m in (
        Model(
            name="bge-small-en-v1.5",
            repo="Qdrant/bge-small-en-v1.5-onnx-Q",
            revision="aa8f8b060edb00e03bfdd08813a2949946c8ba55",
            license="MIT",
            dim=384,
            max_tokens=512,
            onnx="model_optimized.onnx",
            fastembed_name="BAAI/bge-small-en-v1.5",
            files=(
                ModelFile(
                    "model_optimized.onnx",
                    66465124,
                    sha256="51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431",
                ),
                ModelFile("config.json", 706, git_blob="0d7726d0cdccb62ee17c03bd1595cff07199b8f8"),
                ModelFile(
                    "special_tokens_map.json",
                    695,
                    git_blob="9bbecc17cabbcbd3112c14d6982b51403b264bfa",
                ),
                ModelFile(
                    "tokenizer.json", 711396, git_blob="688882a79f44442ddc1f60d70334a7ff5df0fb47"
                ),
                ModelFile(
                    "tokenizer_config.json",
                    1242,
                    git_blob="75305659f7795d4549f0e23688b52fa20a32f925",
                ),
            ),
        ),
        Model(
            name="all-MiniLM-L6-v2",
            repo="Qdrant/all-MiniLM-L6-v2-onnx",
            revision="d13954661f83248295ba75c1ed411eef3b7b936e",
            license="Apache-2.0",
            dim=384,
            max_tokens=256,
            onnx="model.onnx",
            fastembed_name="sentence-transformers/all-MiniLM-L6-v2",
            files=(
                ModelFile(
                    "model.onnx",
                    90387630,
                    sha256="bbd7b466f6d58e646fdc2bd5fd67b2f5e93c0b687011bd4548c420f7bd46f0c5",
                ),
                ModelFile("config.json", 650, git_blob="56c8c186de9040d4fea8daac2ca110f9d412bf04"),
                ModelFile(
                    "special_tokens_map.json",
                    695,
                    git_blob="9bbecc17cabbcbd3112c14d6982b51403b264bfa",
                ),
                ModelFile(
                    "tokenizer.json", 711649, git_blob="4eade387f6f3ade6a663a25642ed27843081a772"
                ),
                ModelFile(
                    "tokenizer_config.json",
                    1412,
                    git_blob="954add25cc0792a2226b11d8ed5c9c39578084f0",
                ),
            ),
        ),
    )
}
assert all({f.name for f in m.files} == {m.onnx, *_TOKENIZER_FILES} for m in MODELS.values())


class ModelError(Exception):
    pass


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def _matches(path: Path, file: ModelFile) -> bool:
    if not path.exists() or path.stat().st_size != file.size:
        return False
    data = path.read_bytes()
    if file.sha256 is not None:
        return hashlib.sha256(data).hexdigest() == file.sha256
    return git_blob_id(data) == file.git_blob


def file_url(model: Model, file: ModelFile) -> str:
    return f"https://huggingface.co/{model.repo}/resolve/{model.revision}/{file.name}"


def model_dir(name: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / name


def fetch_model(
    name: str, models_dir: Path = MODELS_DIR, *, floor: int = DEFAULT_FLOOR_BYTES
) -> tuple[int, int]:
    """Fetch the model's missing or changed files; return (files fetched, files present)."""
    model = MODELS[name]
    target = model_dir(name, models_dir)
    fetched = present = 0
    for file in model.files:
        path = target / file.name
        if _matches(path, file):
            present += 1
            continue
        url = file_url(model, file)
        try:
            download(url, path, floor=floor)
        except urllib.error.URLError as e:
            raise FetchError(f"{url}: {e}") from e
        if not _matches(path, file):
            path.unlink()
            raise FetchError(f"{url} does not match its pinned hash")
        fetched += 1
    return fetched, present


def require_model(name: str, models_dir: Path = MODELS_DIR) -> Path:
    """The model's directory, refused unless every pinned file is there and matches."""
    model = MODELS[name]
    target = model_dir(name, models_dir)
    if not all(_matches(target / f.name, f) for f in model.files):
        raise ModelError(f"{name} is missing or changed; run `lineage model fetch {name}`")
    return target

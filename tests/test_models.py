"""Model files: pinned by hash, fetched only through the project's client, refused if changed."""

import hashlib
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from fastembed import TextEmbedding
from typer.testing import CliRunner

from lineage import models
from lineage.cli import app
from lineage.embed import embed
from lineage.fetch import FetchError
from lineage.models import MODELS, Model, ModelError, ModelFile, fetch_model, git_blob_id

ONNX = b"onnx bytes"
TOKENIZER = b'{"tokenizer": true}'


def test_a_git_blob_id_is_what_git_records_for_the_file() -> None:
    # `printf 'hello\n' | git hash-object --stdin`
    assert git_blob_id(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_every_model_is_pinned_to_a_commit_and_every_file_to_a_hash() -> None:
    for model in MODELS.values():
        assert len(model.revision) == 40
        assert int(model.revision, 16) >= 0
        for file in model.files:
            assert (file.sha256 is None) != (file.git_blob is None)
        assert model.dim == 384


def _tiny(monkeypatch: pytest.MonkeyPatch, served: dict[str, bytes]) -> list[str]:
    model = Model(
        "tiny", "org/tiny", "0" * 40, "MIT", 4, 16, "model.onnx", "org/tiny",
        (
            ModelFile("model.onnx", len(ONNX), sha256=hashlib.sha256(ONNX).hexdigest()),
            ModelFile("tokenizer.json", len(TOKENIZER), git_blob=git_blob_id(TOKENIZER)),
        ),
    )  # fmt: skip
    monkeypatch.setitem(MODELS, "tiny", model)
    urls: list[str] = []

    def download(url: str, dest: Path, *, floor: int) -> tuple[int, str]:
        urls.append(url)
        body = served[url.rsplit("/", 1)[1]]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        return len(body), hashlib.sha256(body).hexdigest()

    monkeypatch.setattr(models, "download", download)
    return urls


def test_files_are_fetched_at_the_pinned_revision_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls = _tiny(monkeypatch, {"model.onnx": ONNX, "tokenizer.json": TOKENIZER})

    assert fetch_model("tiny", tmp_path, floor=0) == (2, 0)
    assert fetch_model("tiny", tmp_path, floor=0) == (0, 2)
    assert urls == [
        f"https://huggingface.co/org/tiny/resolve/{'0' * 40}/model.onnx",
        f"https://huggingface.co/org/tiny/resolve/{'0' * 40}/tokenizer.json",
    ]
    assert models.require_model("tiny", tmp_path) == tmp_path / "tiny"


def test_a_file_that_does_not_match_its_hash_is_removed_and_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny(monkeypatch, {"model.onnx": b"other bytes", "tokenizer.json": TOKENIZER})

    with pytest.raises(FetchError, match="pinned hash"):
        fetch_model("tiny", tmp_path, floor=0)

    assert not (tmp_path / "tiny" / "model.onnx").exists()


def test_a_changed_file_is_refused_before_the_model_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny(monkeypatch, {"model.onnx": ONNX, "tokenizer.json": TOKENIZER})
    fetch_model("tiny", tmp_path, floor=0)
    (tmp_path / "tiny" / "tokenizer.json").write_bytes(b'{"tokenizer": fals}')

    with pytest.raises(ModelError, match="lineage model fetch tiny"):
        models.require_model("tiny", tmp_path)


def test_the_cli_refuses_an_unknown_model(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["model", "fetch", "nope", "--models", str(tmp_path)])

    assert result.exit_code == 1
    assert "unknown model nope" in result.output


class _Model:
    def embed(self, texts: list[str], batch_size: int) -> list[list[float]]:
        return [[3.0, 4.0] if t else [0.0, 0.0] for t in texts]


def test_vectors_come_back_unit_length_so_a_dot_product_is_the_cosine() -> None:
    got = embed(cast(TextEmbedding, _Model()), ["a", ""])

    assert got.dtype == np.float32
    assert got.tolist() == [[0.6000000238418579, 0.800000011920929], [0.0, 0.0]]

"""The pair dataset: fetched at a pinned commit, refused if changed, read as text pairs."""

import hashlib
from pathlib import Path

import pytest

from lineage import pair_data
from lineage.fetch import FetchError
from lineage.pair_data import (
    COMMIT,
    FILES,
    PairDataError,
    PinnedFile,
    fetch_pair_data,
    git_blob_id,
    load_split,
)


def _csv(rows: list[tuple[str, str, int]]) -> bytes:
    lines = ["sec_a_id,sec_b_id,sec_a_title,sec_b_title,sec_a_text,sec_b_text,label"]
    lines += [f"a,b,t,u,{x},{y},{label}" for x, y, label in rows]
    return ("\n".join(lines) + "\n").encode()


def _serve(
    monkeypatch: pytest.MonkeyPatch, bodies: dict[str, bytes], pinned: bool = True
) -> list[str]:
    files = tuple(
        PinnedFile(split, f"p/{split}.csv", len(body), git_blob_id(body) if pinned else "0" * 40)
        for split, body in bodies.items()
    )
    monkeypatch.setattr(pair_data, "FILES", files)
    urls: list[str] = []

    def download(url: str, dest: Path, *, floor: int) -> tuple[int, str]:
        urls.append(url)
        body = bodies[url.rsplit("/", 1)[1].removesuffix(".csv")]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        return len(body), hashlib.sha256(body).hexdigest()

    monkeypatch.setattr(pair_data, "download", download)
    return urls


def test_the_pins_name_three_splits_at_one_commit() -> None:
    assert [f.split for f in FILES] == ["train", "val", "test"]
    assert len(COMMIT) == 40
    assert all(len(f.git_blob) == 40 for f in FILES)


def test_a_split_is_fetched_once_and_read_as_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls = _serve(monkeypatch, {"train": _csv([("one two", "one three", 2), ("x", "y", 0)])})

    assert fetch_pair_data(tmp_path, floor=0) == (1, 0)
    assert fetch_pair_data(tmp_path, floor=0) == (0, 1)
    assert urls == [f"https://raw.githubusercontent.com/{pair_data.REPO}/{COMMIT}/p/train.csv"]
    pairs = load_split("train", tmp_path)
    assert [(p.first, p.second, p.label) for p in pairs] == [
        ("one two", "one three", 2),
        ("x", "y", 0),
    ]


def test_a_file_that_does_not_match_its_pin_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, {"train": _csv([("a", "b", 1)])}, pinned=False)

    with pytest.raises(FetchError, match="git blob id"):
        fetch_pair_data(tmp_path, floor=0)
    with pytest.raises(PairDataError, match="lineage pair-data fetch"):
        load_split("train", tmp_path)


def test_a_label_outside_the_scale_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_bytes(_csv([("a", "b", 5)]))

    with pytest.raises(PairDataError, match="outside"):
        pair_data.read_pairs(path)

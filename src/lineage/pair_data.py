"""The labeled pair dataset: human-annotated pairs of congressional bill subsections.

Kim, Griggs, Kim and Oh, "Learning Bill Similarity with Annotated and Augmented Corpora of
Bills", EMNLP 2021 (https://aclanthology.org/2021.emnlp-main.787). Annotators labeled 4,721
pairs of subsections on a five-point scale, from unrelated to identical, split 7:1:2 into train,
validation and test sets. The repository publishes no license, so the files are used locally
for evaluation only: they are never committed, redistributed or downloaded in CI, and only the
metrics computed from them are. `lineage pair-data fetch` downloads them through the project's
own HTTP client at a pinned commit, each file checked against its git blob id. See
DESIGN.md#pair-dataset.
"""

import csv
import hashlib
import urllib.error
from dataclasses import dataclass
from pathlib import Path

from lineage.fetch import FetchError, download

PAIR_DATA_DIR = Path("data/pair_data")
REPO = "hikoseon12/learning-bill-similarity"
COMMIT = "e11e4f4aae8447b3f71fc49d47def1f8ce33d529"
_PREFIX = "data/subsection_pairs/human_annotated_pairs"
LABELS = ("unrelated", "partially_related", "related", "almost_identical", "identical")


@dataclass(frozen=True, slots=True)
class PinnedFile:
    split: str
    path: str  # in the repository
    size: int
    git_blob: str


FILES = (
    PinnedFile(
        "train", f"{_PREFIX}/train_sub_sec_pairs.csv", 7602202,
        "6bc6ab2672dac2bf0bcd9e0c788052f8fd14db01",
    ),
    PinnedFile(
        "val", f"{_PREFIX}/val_sub_sec_pairs.csv", 1044795,
        "a2126b69479ba570eef10e297fb90d26d541ebbc",
    ),
    PinnedFile(
        "test", f"{_PREFIX}/test_sub_sec_pairs.csv", 2242926,
        "252c4a0e322532a7089df8f2fdf387093b982f33",
    ),
)  # fmt: skip
SPLITS = tuple(f.split for f in FILES)


class PairDataError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Pair:
    first: str
    second: str
    label: int  # 0 unrelated .. 4 identical (LABELS)


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def _matches(path: Path, file: PinnedFile) -> bool:
    if not path.exists() or path.stat().st_size != file.size:
        return False
    return git_blob_id(path.read_bytes()) == file.git_blob


def file_url(file: PinnedFile) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{file.path}"


def local_path(file: PinnedFile, directory: Path = PAIR_DATA_DIR) -> Path:
    return directory / f"{file.split}.csv"


def fetch_pair_data(directory: Path = PAIR_DATA_DIR, *, floor: int) -> tuple[int, int]:
    """Fetch the missing or changed split files; return (files fetched, files present)."""
    fetched = present = 0
    for file in FILES:
        path = local_path(file, directory)
        if _matches(path, file):
            present += 1
            continue
        url = file_url(file)
        try:
            download(url, path, floor=floor)
        except urllib.error.URLError as e:
            raise FetchError(f"{url}: {e}") from e
        if not _matches(path, file):
            path.unlink()
            raise FetchError(f"{url} does not match its pinned git blob id")
        fetched += 1
    return fetched, present


def load_split(split: str, directory: Path = PAIR_DATA_DIR) -> list[Pair]:
    """One split's pairs, refused unless its file is there and matches its pin."""
    file = next((f for f in FILES if f.split == split), None)
    if file is None:
        raise PairDataError(f"unknown split {split}; known: {', '.join(SPLITS)}")
    path = local_path(file, directory)
    if not _matches(path, file):
        raise PairDataError(f"{path} is missing or changed; run `lineage pair-data fetch`")
    return read_pairs(path)


def read_pairs(path: Path) -> list[Pair]:
    """The pairs in a split file: each subsection's text and the pair's label.

    The files hold the text already lowercased in part and stripped of punctuation, and cut at
    400 words; titles sit in their own columns and are left out, as a section's heading is
    part of its text in the corpus only when the bill prints it there."""
    csv.field_size_limit(1 << 24)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pairs = [Pair(r["sec_a_text"], r["sec_b_text"], int(r["label"])) for r in rows]
    if any(p.label not in range(len(LABELS)) for p in pairs):
        raise PairDataError(f"{path}: a label outside 0-{len(LABELS) - 1}")
    return pairs

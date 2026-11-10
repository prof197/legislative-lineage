"""A stand-in for the embedding model, for tests of the semantic stage without model files.

A text's vector is its bag of words, hashed into DIM dimensions; a word's MARK suffix is
ignored, so a copy whose words carry it reads the same here while its shingles do not."""

import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.semantic import SemanticSearch
from lineage.vectors import build_plan, semantic_dir, shard_name

DIM = 256
MARK = "Q"


def bag_of_words(texts: Sequence[str]) -> NDArray[np.float32]:
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.split():
            word = word.removesuffix(MARK).lower()
            out[i, zlib.crc32(word.encode()) % DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.where(norms == 0, 1, norms)).astype(np.float32)


def bag_of_words_search(tables: Path) -> SemanticSearch:
    """Plan the tables' windows and embed them with the stand-in, in memory."""
    plan = build_plan(tables, floor_bytes=0)
    with queries.connect_quiet() as con:
        texts = [
            t
            for s in range(plan.shards)
            for t in queries.read_semantic_texts(
                con, parquet_path=semantic_dir(tables) / "texts" / f"{shard_name(s)}.parquet"
            )
        ]
    return SemanticSearch(
        bag_of_words, [bag_of_words(texts)], str(semantic_dir(tables) / "units" / "*.parquet")
    )

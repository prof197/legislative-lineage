from pathlib import Path

import numpy as np

from lineage import queries
from lineage.ann import Neighbors
from lineage.neighbors import build_neighbors, keep_neighbors
from lineage.semantic import K
from lineage.vectors import build_plan, semantic_dir, shard_name, vectors_dir
from tests.fakes import bag_of_words


def test_a_text_is_not_its_own_neighbor_and_weak_neighbors_are_dropped() -> None:
    found = Neighbors(
        np.array([[0, 5, 6, -1], [1, 0, 7, 8]], dtype=np.int64),
        np.array([[1.0, 0.95, 0.5, 0.0], [1.0, 0.99, 0.97, 0.96]], dtype=np.float32),
    )

    got = keep_neighbors(found, np.array([0, 1], dtype=np.int64), 2, 0.9)

    assert got.ids.tolist() == [[5, -1], [0, 7]]
    assert np.allclose(got.scores, [[0.95, 0.0], [0.99, 0.97]])


def test_every_text_gets_its_nearest_texts_above_the_threshold(
    corpus: Path, tmp_path: Path
) -> None:
    tables = tmp_path / "tables"
    tables.mkdir()
    for name in ("sections", "eligible.parquet"):
        (tables / name).symlink_to(corpus / name)
    plan = build_plan(tables, floor_bytes=0)
    out = vectors_dir(tables, "fake")
    out.mkdir(parents=True)
    with queries.connect_quiet() as con:
        for s in range(plan.shards):
            texts = queries.read_semantic_texts(
                con, parquet_path=semantic_dir(tables) / "texts" / f"{shard_name(s)}.parquet"
            )
            np.save(out / f"{shard_name(s)}.npy", bag_of_words(texts).astype(np.float16))

    doc = build_neighbors(tables, model="fake", spec="Flat", nprobe=None, floor_bytes=0)

    ids = np.load(out / "neighbors" / "ids.npy")
    scores = np.load(out / "neighbors" / "scores.npy")
    assert ids.shape == (plan.distinct, K)
    assert (ids != np.arange(plan.distinct)[:, None]).all()
    kept = ids >= 0
    assert (scores[kept] >= 0.89).all()
    assert doc["neighbors_kept"] == int(kept.sum())

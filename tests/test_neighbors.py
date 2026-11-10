from pathlib import Path

import numpy as np
import pytest

from lineage import queries
from lineage.ann import Neighbors
from lineage.memory import MemoryBoundError, current_bytes
from lineage.neighbors import build_neighbors, keep_neighbors, projected_bytes
from lineage.semantic import K
from lineage.vectors import build_plan, load_plan, semantic_dir, shard_name, vectors_dir
from tests.fakes import bag_of_words


def test_a_text_is_not_its_own_neighbor_and_weak_neighbors_are_dropped() -> None:
    found = Neighbors(
        np.array([[0, 5, 6, -1], [1, 0, 7, 8]], dtype=np.int64),
        np.array([[1.0, 0.95, 0.5, 0.0], [1.0, 0.99, 0.97, 0.96]], dtype=np.float32),
    )

    got = keep_neighbors(found, np.array([0, 1], dtype=np.int64), 2, 0.9)

    assert got.ids.tolist() == [[5, -1], [0, 7]]
    assert np.allclose(got.scores, [[0.95, 0.0], [0.99, 0.97]])


@pytest.mark.parametrize("candidates", [None, 30])  # the index's own scores, or re-scored
def test_every_text_gets_its_nearest_texts_above_the_threshold(
    corpus: Path, tmp_path: Path, candidates: int | None
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

    doc = build_neighbors(
        tables, model="fake", spec="Flat", ef_search=None, candidates=candidates, floor_bytes=0
    )

    ids = np.load(out / "neighbors" / "ids.npy")
    scores = np.load(out / "neighbors" / "scores.npy")
    assert ids.shape == (plan.distinct, K)
    assert (ids != np.arange(plan.distinct)[:, None]).all()
    kept = ids >= 0
    assert (scores[kept] >= 0.89).all()
    assert doc["neighbors_kept"] == int(kept.sum())
    assert doc["exact_check"]["texts"] == plan.distinct  # the fixture is smaller than CHECKED
    assert doc["exact_check"]["recall"] == 1.0  # a flat index is exact


def _embedded(corpus: Path, tmp_path: Path) -> Path:
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
    return tables


def test_a_build_projected_over_the_memory_bound_is_not_started(
    corpus: Path, tmp_path: Path
) -> None:
    tables = _embedded(corpus, tmp_path)

    with pytest.raises(MemoryBoundError, match="not started"):
        build_neighbors(
            tables, model="fake", spec="Flat", ef_search=None, floor_bytes=0, bound_bytes=1
        )
    assert not (vectors_dir(tables, "fake") / "neighbors" / "ids.npy").exists()


def test_a_build_that_crosses_the_memory_bound_stops(corpus: Path, tmp_path: Path) -> None:
    tables = _embedded(corpus, tmp_path)
    vectors = np.load(next(vectors_dir(tables, "fake").glob("*.npy")))
    plan = load_plan(tables)
    assert plan is not None
    projected = projected_bytes(plan.distinct, vectors.shape[1])
    assert projected < current_bytes()  # the test process already holds more

    with pytest.raises(MemoryBoundError, match="stopped"):
        build_neighbors(
            tables,
            model="fake",
            spec="Flat",
            ef_search=None,
            floor_bytes=0,
            bound_bytes=projected,
        )


def _report(recall: float, within: bool) -> dict[str, object]:
    return {
        "texts": 10,
        "texts_with_a_neighbor": 5,
        "threshold": 0.89,
        "neighbors_kept": 7,
        "seconds": 1,
        "peak_memory_bytes": 1 << 30,
        "within_memory_bound": within,
        "exact_check": {"texts": 10, "true_neighbors": 7, "recall": recall},
    }


@pytest.mark.parametrize(
    ("recall", "within", "said"),
    [
        (0.985, True, "below 0.99; rerun with a larger --ef-search than 256"),
        (0.999, False, "peak memory crossed the bound"),
    ],
)
def test_the_command_stops_below_the_recall_floor_or_over_the_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recall: float, within: bool, said: str
) -> None:
    from typer.testing import CliRunner

    from lineage import cli

    def build(*args: object, **kwargs: object) -> dict[str, object]:
        return _report(recall, within)

    monkeypatch.setattr(cli, "build_neighbors", build)
    report = tmp_path / "neighbors.json"

    result = CliRunner().invoke(cli.app, ["neighbors", "--report", str(report)])

    assert result.exit_code == 1
    assert said in result.output
    assert report.exists()  # the measurement is kept, to report


def test_the_command_passes_at_the_floor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from lineage import cli

    def build(*args: object, **kwargs: object) -> dict[str, object]:
        return _report(0.99, True)

    monkeypatch.setattr(cli, "build_neighbors", build)

    result = CliRunner().invoke(cli.app, ["neighbors", "--report", str(tmp_path / "n.json")])

    assert result.exit_code == 0, result.output

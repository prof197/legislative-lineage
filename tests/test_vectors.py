import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from lineage import queries, vectors
from lineage.vectors import (
    EmbedError,
    build_plan,
    current_plan,
    digest,
    load_plan,
    missing_shards,
    semantic_dir,
    shard_name,
)

LONG = " ".join(f"w{i}" for i in range(150))  # two windows: words 0-93 and 56-149
TEXTS = {
    # (bill, ordinal): display text
    ("b1", 1): "the department of revenue shall publish the rates each year",
    ("b1", 2): LONG,
    ("b2", 1): "the department of revenue shall publish the rates each year",  # a companion
    ("b3", 1): "this act takes effect on july first",  # boilerplate: not embedded
    ("r1", 1): "commending the state champions",  # an excluded document type
}
BOILERPLATE = ("b3", 1)
EXCLUDED_BILL = "r1"


@pytest.fixture
def tables(tmp_path: Path) -> Path:
    tables = tmp_path / "tables"
    (tables / "sections").mkdir(parents=True)
    keys = sorted(TEXTS)
    halves = {"a.parquet": keys[:2], "b.parquet": keys[2:]}
    with queries.connect_quiet() as con:
        for name, part in halves.items():
            columns = queries.SectionColumns(
                bill_id=[b for b, _ in part],
                ordinal=[o for _, o in part],
                heading=[None] * len(part),
                text=[TEXTS[k] for k in part],
                normalized=[TEXTS[k] for k in part],
                by_heading=[True] * len(part),
                boilerplate=["effective date" if k == BOILERPLATE else None for k in part],
            )
            queries.write_sections(con, parquet_path=tables / "sections" / name, columns=columns)
        bills = sorted({b for b, _ in keys})
        queries.write_eligibility(
            con,
            parquet_path=tables / "eligible.parquet",
            ids=bills,
            eligible=[b != EXCLUDED_BILL for b in bills],
            rules=[1] * len(bills),
        )
    return tables


def _units(tables: Path) -> list[tuple[str, int, int, int]]:
    with queries.connect_quiet() as con:
        rows: list[tuple[str, int, int, int]] = con.execute(
            "SELECT bill_id, ordinal, window_number, text_id FROM read_parquet($p) "
            "ORDER BY bill_id, ordinal, window_number",
            {"p": str(semantic_dir(tables) / "units" / "*.parquet")},
        ).fetchall()
    return rows


def _texts(tables: Path, plan_shards: int) -> list[str]:
    with queries.connect_quiet() as con:
        return [
            t
            for s in range(plan_shards)
            for t in queries.read_semantic_texts(
                con, parquet_path=semantic_dir(tables) / "texts" / f"{shard_name(s)}.parquet"
            )
        ]


def test_each_distinct_window_text_is_planned_once(
    tables: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vectors, "SHARD_TEXTS", 2)

    plan = build_plan(tables, floor_bytes=0)

    words = LONG.split()
    first, second = " ".join(words[:94]), " ".join(words[56:])
    assert (plan.windows, plan.distinct, plan.shards) == (4, 3, 2)
    assert _texts(tables, plan.shards) == [TEXTS[("b1", 1)], first, second]
    # The companion's window shares the first bill's text id.
    assert _units(tables) == [("b1", 1, 0, 0), ("b1", 2, 0, 1), ("b1", 2, 1, 2), ("b2", 1, 0, 0)]
    assert load_plan(tables) == plan


def test_a_plan_is_reused_and_refused_once_the_sections_change(tables: Path) -> None:
    plan, built = current_plan(tables, floor_bytes=0)
    assert built
    assert current_plan(tables, floor_bytes=0) == (plan, False)

    (tables / "sections" / "a.parquet").write_bytes(b"changed")
    with pytest.raises(EmbedError, match="sections or their eligibility changed"):
        current_plan(tables, floor_bytes=0)


class _Model:
    def token_count(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts: list[str], batch_size: int) -> list[list[float]]:
        return [[float(len(t.split())), 1.0] for t in texts]


def test_a_shard_is_embedded_as_float16_rows_and_counted_done(
    tables: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_plan(tables, floor_bytes=0)
    out = vectors.vectors_dir(tables, "m")
    out.mkdir(parents=True)
    monkeypatch.setitem(vectors._WORKER, "model", _Model())  # pyright: ignore[reportPrivateUsage]
    assert missing_shards(tables, "m", plan) == [0]

    count, _ = vectors._embed_shard(  # pyright: ignore[reportPrivateUsage]
        semantic_dir(tables) / "texts" / "00000.parquet", out / "00000.npy"
    )

    got: NDArray[np.float16] = np.load(out / "00000.npy")
    assert count == plan.distinct
    assert got.dtype == np.float16
    assert np.allclose(np.linalg.norm(got.astype(np.float32), axis=1), 1, atol=1e-3)
    assert missing_shards(tables, "m", plan) == []
    assert not list(out.glob("*.partial"))


def test_digests_tell_texts_apart() -> None:
    assert digest("a b") != digest("a  b")
    assert len(digest("a")) == 16


def test_shards_are_embedded_in_a_fixed_random_order() -> None:
    order = vectors.shard_order(list(range(100)))

    assert sorted(order) == list(range(100))
    assert order != list(range(100))
    assert order == vectors.shard_order(list(range(100)))


def test_the_report_counts_a_stopped_run_up_to_its_last_shard(tables: Path) -> None:
    plan = build_plan(tables, floor_bytes=0)
    out = vectors.vectors_dir(tables, "m")
    out.mkdir(parents=True)
    events = [
        {"run": 100.0, "at": 100.0, "event": "start", "processes": 4, "threads": 2},
        {"run": 100.0, "at": 160.0, "event": "shard", "shard": 0, "texts": 3},
        # stopped here, with no end event
        {"run": 500.0, "at": 500.0, "event": "start", "processes": 2, "threads": 4},
        {"run": 500.0, "at": 530.0, "event": "end"},
    ]
    (out / "runs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))

    doc = vectors.report(tables, "m", plan)

    assert (doc["runs"], doc["seconds"], doc["texts_embedded_in_runs"]) == (2, 90, 3)
    assert doc["processes_threads"] == [(2, 4), (4, 2)]


def test_a_change_of_eligibility_makes_the_plan_stale(tables: Path) -> None:
    current_plan(tables, floor_bytes=0)
    with queries.connect_quiet() as con:
        queries.write_eligibility(
            con,
            parquet_path=tables / "eligible.parquet",
            ids=["b1", "b2", "b3", "r1"],
            eligible=[True, False, True, False],
            rules=[1, 1, 1, 1],
        )

    with pytest.raises(EmbedError, match="eligibility changed"):
        current_plan(tables, floor_bytes=0)


def test_a_new_plan_is_refused_over_vectors_from_an_earlier_one(tables: Path) -> None:
    build_plan(tables, floor_bytes=0)
    out = vectors.vectors_dir(tables, "m")
    out.mkdir(parents=True)
    np.save(out / "00000.npy", np.zeros((1, 2), dtype=np.float16))

    with pytest.raises(EmbedError, match="earlier plan"):
        build_plan(tables, floor_bytes=0)

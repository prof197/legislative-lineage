"""The semantic stage's matches as cross-state section pairs: what the neighbors imply, once."""

import json
from pathlib import Path

import numpy as np
import pytest

from lineage import queries
from lineage.semantic import THRESHOLD, K
from lineage.semantic_pairs import (
    SemanticPairsError,
    build_semantic_pairs,
    load_semantic_pairs,
    pairs_dir,
    section_keys,
)
from lineage.vectors import semantic_dir, vectors_dir
from tests.conftest import FAKE_MODEL as MODEL


@pytest.fixture
def tables(embedded_commission: Path) -> Path:
    return embedded_commission


def _expected(tables: Path) -> dict[tuple[tuple[str, int], tuple[str, int]], float]:
    """Every cross-state section pair two windows' texts imply, at its best cosine, by brute
    force: the same text, or one text among the other's kept neighbors."""
    with queries.connect_quiet() as con:
        units = con.execute(
            "SELECT u.bill_id, u.ordinal, u.text_id, b.state "
            "FROM read_parquet($u) u JOIN read_parquet($b) b ON b.id = u.bill_id",
            {
                "u": str(semantic_dir(tables) / "units" / "*.parquet"),
                "b": str(tables / "bills" / "*.parquet"),
            },
        ).fetchall()
    bills, ordinals = section_keys(tables)
    signed = set(zip(bills, ordinals.tolist(), strict=True))
    neighbors = vectors_dir(tables, MODEL) / "neighbors"
    ids, scores = np.load(neighbors / "ids.npy"), np.load(neighbors / "scores.npy")
    cosine: dict[tuple[int, int], float] = {}
    for t in range(ids.shape[0]):
        cosine[(t, t)] = 1.0
        for u, c in zip(ids[t].tolist(), scores[t].tolist(), strict=True):
            if u >= 0:
                for key in ((t, u), (u, t)):
                    cosine[key] = max(cosine.get(key, -1.0), float(c))
    best: dict[tuple[tuple[str, int], tuple[str, int]], float] = {}
    for bill_a, ord_a, text_a, state_a in units:
        for bill_b, ord_b, text_b, state_b in units:
            a, b = (bill_a, ord_a), (bill_b, ord_b)
            if state_a == state_b or a >= b or a not in signed or b not in signed:
                continue
            c = cosine.get((text_a, text_b))
            if c is not None:
                best[(a, b)] = max(best.get((a, b), -1.0), c)
    return best


def test_every_cross_state_pair_the_neighbors_imply_is_stored_once_at_its_best(
    tables: Path,
) -> None:
    report = build_semantic_pairs(tables, model=MODEL, floor_bytes=0)

    pairs = load_semantic_pairs(tables, model=MODEL)
    bills, ordinals = section_keys(tables)

    def key(section: int) -> tuple[str, int]:
        return bills[section], int(ordinals[section])

    got = {
        tuple(sorted((key(a), key(b)))): float(c)
        for a, b, c in zip(
            pairs.left.tolist(), pairs.right.tolist(), pairs.cosine.tolist(), strict=True
        )
    }
    expected = _expected(tables)
    assert expected, "the fixture should have cross-state matches by meaning"
    assert got.keys() == expected.keys()
    for k, c in expected.items():
        assert got[k] == pytest.approx(c, abs=1e-3)
    assert (pairs.left < pairs.right).all()
    assert len(set(zip(pairs.left.tolist(), pairs.right.tolist(), strict=True))) == pairs.left.size
    assert (pairs.cosine >= THRESHOLD - 2**-11).all()
    assert report["pairs"] == pairs.left.size


def test_each_pair_names_the_window_pair_that_gave_its_best_cosine(tables: Path) -> None:
    build_semantic_pairs(tables, model=MODEL, floor_bytes=0)

    pairs = load_semantic_pairs(tables, model=MODEL)
    bills, ordinals = section_keys(tables)
    with queries.connect_quiet() as con:
        rows = con.execute(
            "SELECT bill_id, ordinal, window_number, text_id FROM read_parquet($u)",
            {"u": str(semantic_dir(tables) / "units" / "*.parquet")},
        ).fetchall()
    text_of = {(b, o, w): t for b, o, w, t in rows}
    vectors = np.concatenate(
        [np.load(p).astype(np.float32) for p in sorted(vectors_dir(tables, MODEL).glob("*.npy"))]
    )
    for a, b, c, wa, wb in zip(
        pairs.left.tolist(),
        pairs.right.tolist(),
        pairs.cosine.tolist(),
        pairs.left_window.tolist(),
        pairs.right_window.tolist(),
        strict=True,
    ):
        ta = text_of[(bills[a], int(ordinals[a]), wa)]
        tb = text_of[(bills[b], int(ordinals[b]), wb)]
        assert float(vectors[ta] @ vectors[tb]) == pytest.approx(c, abs=1e-2)


@pytest.mark.parametrize(
    "changed", [{"model": "other"}, {"k": K + 1}, {"threshold": THRESHOLD + 0.01}]
)
def test_a_table_built_under_another_model_k_or_threshold_refuses_to_load(
    tables: Path, changed: dict[str, object]
) -> None:
    build_semantic_pairs(tables, model=MODEL, floor_bytes=0)
    path = pairs_dir(tables, MODEL) / "fingerprint.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | changed))

    with pytest.raises(SemanticPairsError, match="lineage semantic-pairs"):
        load_semantic_pairs(tables, model=MODEL)


def test_neighbors_kept_at_another_k_are_refused(tables: Path) -> None:
    with pytest.raises(SemanticPairsError, match="K"):
        build_semantic_pairs(tables, model=MODEL, floor_bytes=0, k=K + 1)
    assert not pairs_dir(tables, MODEL).exists()

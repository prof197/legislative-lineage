import numpy as np
from numpy.typing import NDArray

from lineage.ann import (
    add,
    exact_search,
    recall_at,
    rescore,
    search,
    set_search,
    train_index,
)

SEED = 7


def _unit(n: int, dim: int, rng: np.random.Generator) -> NDArray[np.float32]:
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_exact_search_across_blocks_matches_one_matrix_product() -> None:
    rng = np.random.default_rng(SEED)
    corpus = _unit(500, 16, rng)
    queries = corpus[:20] + 0.01 * _unit(20, 16, rng)

    got = exact_search(
        [corpus[:123].astype(np.float16), corpus[123:].astype(np.float16)], queries, 5
    )

    want = np.argsort(-(queries @ corpus.T), axis=1)[:, :5]
    assert recall_at(got.ids, want) == 1.0
    assert got.ids[:, 0].tolist() == list(range(20))
    assert (np.diff(got.scores, axis=1) <= 1e-6).all()


def test_a_corpus_smaller_than_k_pads_with_no_neighbor() -> None:
    rng = np.random.default_rng(SEED)
    corpus = _unit(3, 8, rng)

    got = exact_search([corpus], corpus[:1], 5)

    assert got.ids[0, 3:].tolist() == [-1, -1]
    assert np.isneginf(got.scores[0, 3:]).all()


def test_rescoring_orders_candidates_by_their_stored_vectors() -> None:
    rng = np.random.default_rng(SEED)
    stored = _unit(10, 8, rng)
    query = stored[4:5]
    candidates = np.array([[1, 4, -1, 7]], dtype=np.int64)

    got = rescore(candidates, query, lambda ids: stored[ids], 3)

    assert got.ids[0, 0] == 4
    assert np.isclose(got.scores[0, 0], 1.0)
    assert -1 not in got.ids[0, :3].tolist()


def test_recall_counts_only_real_neighbors() -> None:
    truth = np.array([[1, 2, -1], [3, 4, 5]], dtype=np.int64)
    found = np.array([[2, 9, 9], [3, 4, 6]], dtype=np.int64)

    assert recall_at(found, truth) == 3 / 5


def test_approximate_indexes_find_near_copies() -> None:
    rng = np.random.default_rng(SEED)
    corpus = _unit(4000, 32, rng)
    queries = corpus[:50] + 0.05 * _unit(50, 32, rng)
    truth = np.arange(50, dtype=np.int64)[:, None]

    hnsw = train_index("HNSW16,SQfp16", 32, corpus)
    add(hnsw, corpus)
    set_search(hnsw, ef_search=64)
    ivf = train_index("IVF16,PQ8", 32, corpus)
    add(ivf, corpus)
    set_search(ivf, nprobe=8)

    assert recall_at(search(hnsw, queries, 1).ids, truth) == 1.0
    approx = search(ivf, queries, 10).ids
    assert recall_at(rescore(approx, queries, lambda ids: corpus[ids], 1).ids, truth) == 1.0

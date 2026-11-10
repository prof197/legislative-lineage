"""`lineage eval pair-data`: the matching stages scored on the labeled pair dataset.

Each pair of subsections is scored the way the pipeline scores a pair of sections:

- the near-duplicate stage cuts each text into its windows, shingles, stoplists and signs
  them, and finds the pair when a pair of windows shares a band key and verifies at the
  threshold, as the build does;
- the semantic stage embeds each text's windows, 94 words without overlap, and scores the pair
  by its best cosine between windows.

A pair counts as positive when annotators labeled it related or closer (2 to 4 on the 0 to 4
scale). Each model's cosine threshold is set on the train and validation splits, never the test
split, as the lowest cosine whose precision there reaches the target, so it gives the most
recall at that precision; the test split reports what it gives. Only metrics are written; the
dataset is not committed (lineage.pair_data). See DESIGN.md#pair-dataset.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lineage.boilerplate import boilerplate_kind
from lineage.embed import embed, load_model
from lineage.lsh import band_keys
from lineage.minhash import permutations, signatures
from lineage.models import MODELS_DIR
from lineage.pair_data import LABELS, PAIR_DATA_DIR, SPLITS, Pair, load_split
from lineage.sections import normalize
from lineage.shingles import section_shingles
from lineage.stoplist import apply_stoplist, load_stoplist
from lineage.verify import THRESHOLD, min_agreements
from lineage.windows import WINDOW_WORDS, display_windows, normalized_windows

COMMAND = "uv run lineage eval pair-data"
REPORT = Path("reports/pair_data.json")
POSITIVE_FROM = 2  # related
# The precision each model's threshold is set at; recall is reported at the others too.
TARGET_PRECISION = 0.9
PRECISIONS = (0.8, 0.9, 0.95)
SEMANTIC_STRIDE = WINDOW_WORDS  # the semantic stage's unit: windows without overlap
DEV_SPLITS = ("train", "val")


@dataclass(frozen=True, slots=True)
class NearDuplicate:
    agreements: int  # the best window pair's agreeing positions; 0 when a side is unsigned
    proposed: bool  # some pair of windows shares a band key
    signed: bool = True  # False when a side is boilerplate, too short or wholly stoplisted


def _signed(text: str, stoplist: NDArray[np.uint32]) -> NDArray[np.uint32] | None:
    normalized = normalize(text)
    if boilerplate_kind(normalized) is not None:
        return None
    sets = [
        s
        for s in apply_stoplist(section_shingles(normalized_windows(text, normalized)), stoplist)
        if s.size
    ]
    if not sets:
        return None
    return signatures(sets, permutations())


def near_duplicate_scores(
    pairs: Sequence[Pair], stoplist: NDArray[np.uint32]
) -> list[NearDuplicate]:
    out: list[NearDuplicate] = []
    for pair in pairs:
        a, b = _signed(pair.first, stoplist), _signed(pair.second, stoplist)
        if a is None or b is None:
            out.append(NearDuplicate(0, False, signed=False))
            continue
        agree = (a[:, None, :] == b[None, :, :]).sum(axis=2)
        keys_a, keys_b = band_keys(a), band_keys(b)
        shared = (keys_a[:, None, :] == keys_b[None, :, :]).any(axis=2)
        out.append(NearDuplicate(int(agree.max()), bool(shared.any())))
    return out


def best_cosines(
    pairs: Sequence[Pair], embed_texts: Callable[[Sequence[str]], NDArray[np.float32]]
) -> NDArray[np.float32]:
    """Each pair's best cosine between its two texts' windows; each distinct window is
    embedded once, by `embed_texts`, which returns unit-length rows."""
    windows = [
        (
            display_windows(p.first, size=WINDOW_WORDS, stride=SEMANTIC_STRIDE),
            display_windows(p.second, size=WINDOW_WORDS, stride=SEMANTIC_STRIDE),
        )
        for p in pairs
    ]
    distinct = sorted({w for a, b in windows for w in (*a, *b)})
    at = {text: i for i, text in enumerate(distinct)}
    vectors = embed_texts(distinct)
    scores = np.empty(len(pairs), dtype=np.float32)
    for i, (a, b) in enumerate(windows):
        scores[i] = (vectors[[at[w] for w in a]] @ vectors[[at[w] for w in b]].T).max()
    return scores


def semantic_scores(
    pairs: Sequence[Pair], model_name: str, models_dir: Path = MODELS_DIR
) -> NDArray[np.float32]:
    model = load_model(model_name, models_dir)
    return best_cosines(pairs, lambda texts: embed(model, texts))


def precision_recall(
    scores: NDArray[np.float32], positive: NDArray[np.bool_], threshold: float
) -> tuple[float, float]:
    chosen = scores >= threshold
    found = int((chosen & positive).sum())
    precision = found / int(chosen.sum()) if chosen.any() else 1.0
    recall = found / int(positive.sum()) if positive.any() else 0.0
    return precision, recall


def threshold_at_precision(
    scores: NDArray[np.float32], positive: NDArray[np.bool_], precision: float
) -> float | None:
    """The lowest score threshold whose precision reaches `precision`: the most recall there."""
    order = np.argsort(-scores, kind="stable")
    ranked, hits = scores[order], np.cumsum(positive[order])
    # A threshold keeps every score at or above it, so only the last of a run of ties counts.
    last = np.r_[ranked[1:] != ranked[:-1], True]
    kept = np.arange(1, scores.size + 1)
    meets = last & (hits / kept >= precision)
    return float(ranked[np.flatnonzero(meets)[-1]]) if meets.any() else None


def roc_auc(scores: NDArray[np.float32], positive: NDArray[np.bool_]) -> float:
    """The chance a positive pair outscores a negative one, ties counting half."""
    ranks = _average_ranks(scores)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(values: NDArray[np.float32]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1)
    sorted_values = values[order]
    starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    ends = np.r_[starts[1:], values.size]
    for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
        if e - s > 1:
            ranks[order[s:e]] = (s + 1 + e) / 2
    return ranks


def _by_label(labels: NDArray[np.int64], found: NDArray[np.bool_]) -> dict[str, dict[str, Any]]:
    return {
        name: {"pairs": int((labels == i).sum()), "found": int((found & (labels == i)).sum())}
        for i, name in enumerate(LABELS)
    }


def _semantic(
    name: str,
    dev: Sequence[Pair],
    test: Sequence[Pair],
    positive: dict[str, NDArray[np.bool_]],
    labels: NDArray[np.int64],
    near_found: NDArray[np.bool_],
    models_dir: Path,
) -> dict[str, Any]:
    dev_scores = semantic_scores(dev, name, models_dir)
    test_scores = semantic_scores(test, name, models_dir)
    at: dict[str, Any] = {}
    for target in PRECISIONS:
        cut = threshold_at_precision(dev_scores, positive["dev"], target)
        entry: dict[str, Any] = {"threshold": None if cut is None else round(cut, 4)}
        if cut is not None:
            found = test_scores >= cut
            p, r = precision_recall(test_scores, positive["test"], cut)
            entry |= {"test_precision": round(p, 4), "test_recall": round(r, 4)}
            if target == TARGET_PRECISION:
                either = (found | near_found).astype(np.float32)
                p, r = precision_recall(either, positive["test"], 1.0)
                entry["by_label"] = _by_label(labels, found)
                entry["with_near_duplicate"] = {
                    "test_precision": round(p, 4),
                    "test_recall": round(r, 4),
                    "by_label": _by_label(labels, found | near_found),
                }
        at[str(target)] = entry
    return {
        "unit": f"{WINDOW_WORDS}-word windows, stride {SEMANTIC_STRIDE}",
        "dev_roc_auc": round(roc_auc(dev_scores, positive["dev"]), 4),
        "test_roc_auc": round(roc_auc(test_scores, positive["test"]), 4),
        "at_precision": at,
    }


def evaluate(
    tables_dir: Path,
    models: Sequence[str] = (),
    *,
    data_dir: Path = PAIR_DATA_DIR,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    splits = {s: load_split(s, data_dir) for s in SPLITS}
    dev = [p for s in DEV_SPLITS for p in splits[s]]
    test = splits["test"]
    labels = np.array([p.label for p in test], dtype=np.int64)
    positive = {
        "dev": np.array([p.label >= POSITIVE_FROM for p in dev]),
        "test": labels >= POSITIVE_FROM,
    }
    near = near_duplicate_scores(test, load_stoplist(tables_dir))
    need = min_agreements(THRESHOLD)
    found = np.array([n.proposed and n.agreements >= need for n in near])
    precision, recall = precision_recall(found.astype(np.float32), positive["test"], 1.0)
    agreements = np.array([n.agreements for n in near], dtype=np.float32)
    return {
        "command": COMMAND,
        "dataset": {
            "paper": "Kim, Griggs, Kim and Oh, Learning Bill Similarity with Annotated and "
            "Augmented Corpora of Bills, EMNLP 2021, https://aclanthology.org/2021.emnlp-main.787",
            "pairs": {s: len(v) for s, v in splits.items()},
            "positive": f"label {POSITIVE_FROM} ({LABELS[POSITIVE_FROM]}) or higher",
        },
        "near_duplicate": {
            "split": "test",
            "verification_threshold": THRESHOLD,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "roc_auc_of_best_agreements": round(roc_auc(agreements, positive["test"]), 4),
            "by_label": _by_label(labels, found),
            "unsigned_pairs": sum(1 for n in near if not n.signed),
        },
        "semantic": {
            "threshold_set_on": list(DEV_SPLITS),
            "target_precision": TARGET_PRECISION,
            "models": {
                name: _semantic(name, dev, test, positive, labels, found, models_dir)
                for name in models
            },
        },
    }


def write_report(doc: dict[str, Any], path: Path = REPORT) -> None:
    path.write_text(json.dumps(doc, indent=1) + "\n")

"""Origin recall: the enactments an answer key names, found through origin matches."""

import json
from pathlib import Path
from typing import Any

import pytest

from lineage.acts import ACTS
from lineage.answer_key import ANSWER_KEY, COLUMNS, render_answer_key
from lineage.commission import RESULTS
from lineage.intervals import wilson
from lineage.origin_recall import (
    KEYS,
    AnswerKey,
    OriginRecallError,
    evaluate,
    load_key,
    matches_from_commission,
    origin_recall,
    same_slug,
)
from lineage.origins import OriginMatch

ROOT = Path(__file__).parents[1]
SOURCE = "commission"


def _v1() -> dict[str, Any]:
    return json.loads((ROOT / RESULTS).read_text())["commission"]


def _development() -> AnswerKey:
    return AnswerKey("development", ROOT / ANSWER_KEY, sealed=False)


def _counts(cell: dict[str, Any]) -> tuple[int, int]:
    return cell["found"], cell["enactments"]


def test_v1s_development_figure_is_reproduced_from_its_own_matches() -> None:
    """v1's commission results, put in the shape origin matching writes, give v1's recall: the
    same 33 of 36, per act, per stratum and per stage, enactment by enactment."""
    v1 = _v1()
    matches = matches_from_commission(v1, lambda act: f"{SOURCE}/{act}")
    section = origin_recall(matches, _development(), same_slug(ACTS, SOURCE))

    assert _counts(section["recall"]["pooled"]) == (33, 36)
    assert section["recall"]["pooled"]["interval_95"] == [round(v, 4) for v in wilson(33, 36)]
    for breakdown in ("by_act", "by_stratum"):
        assert {k: _counts(c) for k, c in section["recall"][breakdown].items()} == {
            k: _counts(c["all"]) for k, c in v1["recall"][breakdown].items()
        }
    for stage in ("near_duplicate", "semantic", "both"):
        mine, theirs = section["stage_ablation"][stage], v1["stage_ablation"][stage]
        assert _counts(mine["pooled"]) == _counts(theirs["pooled"]["all"])
        assert {a: _counts(c) for a, c in mine["by_act"].items()} == {
            a: _counts(c["all"]) for a, c in theirs["by_act"].items()
        }
    assert _counts(section["stage_ablation"]["near_duplicate"]["pooled"]) == (31, 36)
    # v1 lists the enactments act by act, the key in the lists' order.
    assert sorted(
        (e["act"], e["state"], e["year"], e["listed"], e["found"], *e["found_by_stage"].values())
        for e in section["enactments"]
    ) == sorted(
        (e["act"], e["state"], e["year"], e["bill_cell"], e["found"], *e["found_by_stage"].values())
        for e in v1["enactments"]
    )
    assert [r["found"] for r in section["recall_by_min_share"]] == [
        r["found"] for r in v1["recall_by_min_share"]
    ]
    # v1 lists the enactments it leaves out; rows listed only as introduced are not enactments.
    assert [e for e in section["excluded"] if e["why"] != "listed as introduced"] == v1["excluded"]


# act, jurisdiction, listed, listed status, counted, rule, bill id; the rest as the key leaves it.
ROWS = [
    ("widget-registration", "Utah", "HB 12/SB 7", "Enacted", "yes", "in_year", "ut-hb12"),
    ("widget-registration", "Utah", "HB 12/SB 7", "Enacted", "yes", "in_year", "ut-sb7"),
    ("widget-registration", "Idaho", "S 1001", "Introduced", "no: listed as introduced", "", ""),
    ("widget-registration", "Ohio", "SB 9", "Enacted - Substantially Similiar", "yes", "in_year",
     "oh-sb9"),
    ("widget-registration", "Oregon", "SB 3", "Enacted", "yes", "", ""),
    ("gadget-safety", "Texas", "HB 5", "Enacted", "yes", "in_year", "tx-hb5"),
    ("gadget-safety", "Iowa", "SF 2", "Enacted", "yes", "in_year", "ia-sf2"),
]  # fmt: skip
FIELDS = ("act", "jurisdiction", "listed", "listed_status", "counted", "rule", "bill_id")
BLANK = dict.fromkeys(COLUMNS, "") | {"year": "2098"}
KEY = render_answer_key(
    [BLANK | dict(zip(FIELDS, r, strict=True)) for r in ROWS],
    "uv run lineage eval answer-key; ULC Bill List, read 2099-01-02",
)
DOCS = {
    "widget-registration": frozenset({"lib/widget-registration", "lib/widget-registration-2097"}),
    "gadget-safety": frozenset({"lib/gadget-safety"}),
}


def _match(bill: str, doc: str, share: float, *stages: str) -> OriginMatch:
    return OriginMatch.model_validate(
        {
            "bill_id": bill,
            "doc_id": doc,
            "stages": list(stages) or ["near_duplicate"],
            "model_share": share,
            "bill_share": 0.5,
            "spans": [],
        }
    )


@pytest.fixture
def key(tmp_path: Path) -> AnswerKey:
    path = tmp_path / "key.csv"
    path.write_text(KEY)
    return AnswerKey("toy", path, sealed=False)


def test_an_enactment_is_found_through_any_of_its_bills_and_any_of_its_acts_documents(
    key: AnswerKey,
) -> None:
    matches = [
        _match("ut-sb7", "lib/widget-registration-2097", 0.4, "semantic"),
        _match("ut-sb7", "lib/widget-registration", 0.2),
        _match("oh-sb9", "lib/widget-registration", 0.0),  # matched, at no share of the model
        _match("tx-hb5", "lib/widget-registration", 0.9),  # another act's document
        _match("ia-sf2", "lib/other-act", 0.9),  # a document no act names
    ]
    section = evaluate(matches, load_key(key.path), DOCS)
    by_state = {e["state"]: e for e in section["enactments"]}

    utah = by_state["UT"]
    assert utah["found"]
    assert utah["found_by"] == "ut-sb7"
    assert utah["model_share"] == 0.4  # the act's best document stands
    assert utah["found_by_stage"] == {"near_duplicate": False, "semantic": True}
    assert by_state["OH"]["miss_reason"] == "below_threshold"
    assert by_state["OH"]["stratum"] == "substantially_similar"
    assert by_state["OR"]["miss_reason"] == "not_in_corpus"
    assert by_state["TX"]["miss_reason"] == "not_matched"
    assert by_state["IA"]["miss_reason"] == "not_matched"
    pooled = section["recall"]["pooled"]
    assert (pooled["enactments"], pooled["found"], pooled["recall"]) == (5, 1, 0.2)
    assert pooled["interval_95"] == [round(v, 4) for v in wilson(1, 5)]
    assert section["recall"]["by_act"]["gadget-safety"]["found"] == 0
    assert section["recall"]["by_stratum"]["substantially_similar"]["enactments"] == 1
    assert section["stage_ablation"]["near_duplicate"]["pooled"]["found"] == 0
    assert section["miss_reasons"] == {"not_in_corpus": 1, "not_matched": 2, "below_threshold": 1}
    assert section["excluded"] == [
        {"act": "widget-registration", "jurisdiction": "Idaho", "year": 2098,
         "why": "listed as introduced"}
    ]  # fmt: skip
    assert [r["found"] for r in section["recall_by_min_share"]] == [1, 1, 1, 0]


def test_every_act_of_the_key_needs_a_document(key: AnswerKey) -> None:
    with pytest.raises(OriginRecallError, match="gadget-safety"):
        evaluate([], load_key(key.path), {"widget-registration": DOCS["widget-registration"]})


def test_the_held_out_key_is_sealed_and_refused_without_final() -> None:
    held_out = AnswerKey("held_out", ROOT / KEYS["held_out"].path, sealed=True)
    assert KEYS["held_out"].sealed
    assert not KEYS["development"].sealed
    docs = same_slug({e.act for e in load_key(held_out.path).enactments}, SOURCE)

    with pytest.raises(OriginRecallError, match="sealed"):
        origin_recall([], held_out, docs)
    with pytest.raises(OriginRecallError, match="configuration"):
        origin_recall([], held_out, docs, final=True)
    with pytest.raises(OriginRecallError, match="not sealed"):
        origin_recall([], _development(), same_slug(ACTS, SOURCE), final=True)


def test_a_sealed_key_runs_once_per_configuration(key: AnswerKey) -> None:
    sealed = AnswerKey("toy", key.path, sealed=True)
    first = origin_recall(
        [_match("ut-hb12", "lib/widget-registration", 0.5)],
        sealed, DOCS, final=True, configuration={"semantic_threshold": 0.94},
    )  # fmt: skip
    assert first["earlier_runs"] == []
    assert first["configuration"] == {"semantic_threshold": 0.94}

    with pytest.raises(OriginRecallError, match="already run"):
        origin_recall([], sealed, DOCS, final=True, configuration={"semantic_threshold": 0.94},
                      previous=first)  # fmt: skip
    second = origin_recall(
        [], sealed, DOCS, final=True, configuration={"semantic_threshold": 0.95}, previous=first
    )
    assert [r["configuration"] for r in second["earlier_runs"]] == [{"semantic_threshold": 0.94}]
    assert second["earlier_runs"][0]["found"] == ["widget-registration UT"]
    with pytest.raises(OriginRecallError, match="already run"):
        origin_recall([], sealed, DOCS, final=True, configuration={"semantic_threshold": 0.94},
                      previous=second)  # fmt: skip


def test_the_held_out_key_counts_the_enactments_v1_counted() -> None:
    loaded = load_key(ROOT / KEYS["held_out"].path)
    v1 = json.loads((ROOT / RESULTS).read_text())["commission_held_out"]

    assert len(loaded.enactments) == v1["recall"]["pooled"]["all"]["enactments"] == 19
    assert {e.act for e in loaded.enactments} == set(v1["recall"]["by_act"])

"""The commission recall eval: miss reasons stage by stage, and the pipeline regression test.

The headline number queries with each act's own text, which cannot be committed. CI instead
runs the same eval on a committed fixture (`evals/fixture.py`): the enacting bills among
distractors, each act queried with its earliest enacted state version. That is a regression
test of the pipeline, not the recall the README reports.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from lineage import queries
from lineage.acts import ACTS, HELD_OUT_ACTS
from lineage.cli import app
from lineage.commission import (
    MISS_REASONS,
    CommissionError,
    Probe,
    Relative,
    check_held_out_configuration,
    evaluate,
    probe,
    run_configuration,
    target_reason,
    with_earlier_runs,
)
from lineage.lsh import BANDS, BUCKET_CAP, ROWS
from lineage.minhash import NUM_PERM
from lineage.queries import EnactmentBill
from lineage.semantic import BUILD_THRESHOLD, THRESHOLD
from lineage.sessions import load_sessions
from lineage.ulc import ENACTMENTS, load_enactments
from tests.conftest import COMMISSION
from tests.fakes import bag_of_words_search
from tests.test_acts import ACT

RNG = np.random.default_rng(7)


def _random(n: int) -> np.ndarray:
    return RNG.integers(0, 2**32, size=(n, NUM_PERM), dtype=np.uint32)


def _sizes(n: int, size: int) -> np.ndarray:
    return np.full((n, BANDS), size, dtype=np.int64)


def test_an_identical_section_is_verified() -> None:
    query = _random(2)

    got = probe(query, _sizes(2, 3), query[1:].copy())

    assert got == Probe("verified", 1.0)


def test_a_pair_sharing_one_band_but_little_else_fails_verification() -> None:
    query = _random(1)
    target = _random(1)
    target[0, :ROWS] = query[0, :ROWS]  # the first band's key is shared

    got = probe(query, _sizes(1, 3), target)

    assert got.stage == "verification"
    assert got.best_estimate == ROWS / NUM_PERM


def test_a_match_only_in_buckets_over_the_cap_is_lost_to_the_cap() -> None:
    query = _random(1)

    got = probe(query, _sizes(1, BUCKET_CAP + 1), query.copy())

    assert got.stage == "bucket_cap"


def test_a_section_sharing_no_band_was_never_proposed() -> None:
    assert probe(_random(1), _sizes(1, 3), _random(4)).stage == "not_proposed"


BILL = EnactmentBill(
    id="b",
    state="UT",
    session="2098",
    identifier="HB 12",
    status="passed",
    status_date=None,
    passage_chambers=2,
    text_chars=5000,
    amendment_only=False,
    eligible=True,
    family="f",
)


def test_reasons_follow_the_pipeline_from_source_text_to_the_threshold() -> None:
    proposed = Probe("verification", 0.4)

    def reason(bill: EnactmentBill, rows: int = 3, probed: Probe | None = proposed) -> str | None:
        return target_reason(bill, signed_sections=rows, probed=probed, relative=None)

    assert reason(replace(BILL, text_chars=0, amendment_only=True)) == "no_text_in_source"
    assert reason(replace(BILL, amendment_only=True, eligible=False)) == "amendment_only"
    assert reason(replace(BILL, eligible=False)) == "excluded_type"
    assert reason(BILL, rows=0, probed=None) == "no_usable_text"
    assert reason(BILL) == "verification"
    assert reason(BILL, probed=Probe("not_proposed", 0.1)) == "not_proposed"
    low = Relative(3, 0.1, "HB 12")
    assert target_reason(BILL, signed_sections=3, probed=None, relative=low, min_share=0.2) == (
        "below_threshold"
    )
    assert list(MISS_REASONS)[-1] == "below_threshold"


def test_found_is_a_score_above_the_threshold_whatever_the_rank() -> None:
    # A uniform act's relatives can outnumber any fixed rank once each state's introduced,
    # failed and enacted versions count, so rank is reported but does not decide.
    far = Relative(400, 0.25, "HB 12")

    assert target_reason(BILL, signed_sections=3, probed=None, relative=far, min_share=0.2) is None
    assert target_reason(BILL, signed_sections=3, probed=None, relative=far) is None


def _raw_text(tables: Path, bill_id: str) -> str:
    with queries.connect_quiet() as con:
        row = con.execute(
            "SELECT text FROM read_parquet($p) WHERE id = $i",
            {"p": str(tables / "bills" / "*.parquet"), "i": bill_id},
        ).fetchall()
    return str(row[0][0])


# Measured on the fixture. A pipeline change that loses one of these fails here; one that finds
# more should add it, so the floor only rises. Citation headings (DESIGN.md#sections) added
# Colorado's two enactments, Idaho's, Rhode Island's and New York's, whose chapters had been
# one section each.
REGRESSION_FOUND = {
    ("electronic-estate-planning-documents", "CO", 2024),
    ("electronic-estate-planning-documents", "MN", 2026),
    ("electronic-estate-planning-documents", "UT", 2026),
    ("public-expression-protection", "IA", 2025),
    ("public-expression-protection", "ID", 2025),
    ("public-expression-protection", "MT", 2025),
    ("special-deposits", "CO", 2024),
    ("special-deposits", "NY", 2026),
    ("special-deposits", "RI", 2026),
    ("special-deposits", "UT", 2025),
}


def test_pipeline_regression_the_enacting_bills_are_found_among_distractors(
    commission_corpus: Path,
) -> None:
    build = json.loads((COMMISSION / "build.json").read_text())
    act_texts = {act: _raw_text(commission_corpus, bill) for act, bill in build["queries"].items()}
    snapshot = load_enactments(ENACTMENTS)
    with queries.connect_quiet() as con:
        own = {
            (act, row[0], row[1])
            for act, bill in build["queries"].items()
            for row in con.execute(
                "SELECT state, identifier FROM read_parquet($p) WHERE id = $i",
                {"p": str(commission_corpus / "bills" / "*.parquet"), "i": bill},
            ).fetchall()
        }
    # Each query's own enactment would find itself, so it is left out.
    rows = [
        r
        for r in snapshot.rows
        if not any(r.act == a and r.state == s and i in r.bill_numbers for a, s, i in own)
    ]

    result = evaluate(
        act_texts, replace(snapshot, rows=rows), commission_corpus,
        load_sessions(COMMISSION / "sessions.csv"),
    )  # fmt: skip

    found = {(e["act"], e["state"], e["year"]) for e in result["enactments"] if e["found"]}
    assert found >= REGRESSION_FOUND, f"lost: {sorted(REGRESSION_FOUND - found)}"
    for e in result["enactments"]:
        assert e["found"] or e["miss_reason"] in MISS_REASONS
        if not e["with_text"] and e["bills"]:
            assert e["miss_reason"] == "no_text_in_source", e


def test_the_stage_ablation_counts_each_stage_alone_and_both(commission_corpus: Path) -> None:
    build = json.loads((COMMISSION / "build.json").read_text())
    act_texts = {act: _raw_text(commission_corpus, bill) for act, bill in build["queries"].items()}
    snapshot = load_enactments(ENACTMENTS)
    sessions = load_sessions(COMMISSION / "sessions.csv")

    near = evaluate(act_texts, snapshot, commission_corpus, sessions)
    both = evaluate(
        act_texts, snapshot, commission_corpus, sessions,
        semantic=bag_of_words_search(commission_corpus),
    )  # fmt: skip

    assert near["stage_ablation"] is None
    ablation = both["stage_ablation"]
    found = {
        s: ablation[s]["pooled"]["all"]["found"] for s in ("near_duplicate", "semantic", "both")
    }
    assert found["near_duplicate"] == near["recall"]["pooled"]["all"]["found"]
    assert found["both"] == both["recall"]["pooled"]["all"]["found"]
    # each stage broken down as recall is, so a table can set a stage beside another run's
    assert ablation["near_duplicate"]["by_act"] == near["recall"]["by_act"]
    assert ablation["near_duplicate"]["by_stratum"] == near["recall"]["by_stratum"]
    assert ablation["both"] == both["recall"]
    assert found["both"] >= max(found["near_duplicate"], found["semantic"])
    for e in both["enactments"]:
        assert e["found"] == any(e["found_by_stage"].values())
        for bill in e["bills"]:
            if "semantic" in bill["stages"]:
                assert bill["best_cosine"] >= THRESHOLD - 1e-4


def test_the_command_writes_its_section_of_the_results(
    commission_corpus: Path, tmp_path: Path
) -> None:
    acts = tmp_path / "acts"
    acts.mkdir()
    for act in ACTS:
        (acts / f"{act}.txt").write_text(ACT)
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"other": {"kept": True}}))

    out = CliRunner().invoke(
        app,
        [
            "eval", "commission", "--no-semantic", "--tables", str(commission_corpus),
            "--acts", str(acts),
            "--sessions", str(COMMISSION / "sessions.csv"), "--results", str(results),
        ],
    )  # fmt: skip

    assert out.exit_code == 0, out.output
    doc = json.loads(results.read_text())
    assert doc["other"] == {"kept": True}
    section = doc["commission"]
    assert section["command"] == "uv run lineage eval commission"
    assert set(section["queries"]) == set(ACTS)
    assert section["threshold"]["status"] == "approved"
    for act in ACTS:
        relatives = section["queries"][act]["relatives"]
        assert [r["rank"] for r in relatives] == list(range(1, len(relatives) + 1))
    assert len(section["queries"][ACTS[0]]["file_sha256"]) == 64
    assert section["enactments_file"]["read"].startswith("ULC Bill List, read ")
    assert "recall" in out.output


def test_the_command_says_which_act_text_is_missing(
    commission_corpus: Path, tmp_path: Path
) -> None:
    out = CliRunner().invoke(
        app,
        [
            "eval",
            "commission",
            "--no-semantic",
            "--tables",
            str(commission_corpus),
            "--acts",
            str(tmp_path),
        ],
    )

    assert out.exit_code == 1
    assert ACTS[0] in out.output


def test_the_held_out_key_is_not_evaluated_without_final(
    commission_corpus: Path, tmp_path: Path
) -> None:
    results = tmp_path / "results.json"

    out = CliRunner().invoke(
        app,
        [
            "eval", "commission", "--held-out", "--tables", str(commission_corpus),
            "--results", str(results),
        ],
    )  # fmt: skip

    assert out.exit_code == 1
    assert "--final" in out.output
    assert not results.exists()


def test_final_applies_only_to_the_held_out_key(commission_corpus: Path, tmp_path: Path) -> None:
    out = CliRunner().invoke(
        app, ["eval", "commission", "--final", "--tables", str(commission_corpus)]
    )

    assert out.exit_code == 1
    assert "--held-out" in out.output


def test_the_final_run_reads_the_held_out_acts(commission_corpus: Path, tmp_path: Path) -> None:
    out = CliRunner().invoke(
        app,
        [
            "eval", "commission", "--held-out", "--final", "--no-semantic",
            "--tables", str(commission_corpus),
            "--acts", str(tmp_path),
        ],
    )  # fmt: skip

    assert out.exit_code == 1
    assert HELD_OUT_ACTS[0] in out.output


@pytest.mark.parametrize("reason", MISS_REASONS)
def test_every_miss_reason_is_documented(reason: str) -> None:
    design = (Path(__file__).parents[1] / "DESIGN.md").read_text()
    assert f"`{reason}`" in design


def test_recall_at_a_stricter_semantic_threshold_matches_the_search_and_never_rises(
    commission_corpus: Path,
) -> None:
    build = json.loads((COMMISSION / "build.json").read_text())
    act_texts = {act: _raw_text(commission_corpus, bill) for act, bill in build["queries"].items()}
    snapshot = load_enactments(ENACTMENTS)
    sessions = load_sessions(COMMISSION / "sessions.csv")
    thresholds = (BUILD_THRESHOLD, THRESHOLD, 0.95, 0.99, 1.01)

    got = evaluate(
        act_texts, snapshot, commission_corpus, sessions,
        semantic=bag_of_words_search(commission_corpus), semantic_thresholds=thresholds,
    )  # fmt: skip
    alone = evaluate(
        act_texts, snapshot, commission_corpus, sessions,
        semantic=bag_of_words_search(commission_corpus),
    )  # fmt: skip

    found = [r["found"] for r in got["recall_by_semantic_threshold"]]
    assert found[1] == got["recall"]["pooled"]["all"]["found"]
    assert found == sorted(found, reverse=True)
    # Above every cosine, only the near-duplicate stage finds anything.
    assert found[-1] == got["stage_ablation"]["near_duplicate"]["pooled"]["all"]["found"]
    for e in got["enactments"]:
        assert e["found_at_semantic_threshold"][str(THRESHOLD)] == e["found"]
    # Searching lower for the options leaves the results at the stage's threshold as they were.
    assert got["recall"] == alone["recall"]
    assert got["queries"] == alone["queries"]
    with pytest.raises(CommissionError, match="under the build"):
        evaluate(
            act_texts, snapshot, commission_corpus, sessions,
            semantic=bag_of_words_search(commission_corpus), semantic_thresholds=(0.5,),
        )  # fmt: skip


def _held_out_run(threshold: float, found: list[tuple[str, str]]) -> dict[str, Any]:
    enactments = [{"act": a, "state": st, "found": True} for a, st in found]
    pooled = {"all": {"enactments": 3, "found": len(found)}}
    return {
        "semantic_threshold": threshold,
        "state_boilerplate_flag": {"sections": 2, "both_stages_exempt_from": BUILD_THRESHOLD},
        "recall": {"pooled": pooled},
        "stage_ablation": {"near_duplicate": {"pooled": pooled}},
        "miss_reasons": {"not_proposed": 3 - len(found)},
        "enactments": [*enactments, {"act": "x", "state": "IA", "found": False}],
    }


def test_a_held_out_run_keeps_the_earlier_runs_and_refuses_one_already_run() -> None:
    first = with_earlier_runs(_held_out_run(0.93, [("x", "PA"), ("x", "NV")]), None)
    assert first["earlier_runs"] == []
    second = with_earlier_runs(_held_out_run(0.94, [("x", "PA")]), first)

    assert [r["semantic_threshold"] for r in second["earlier_runs"]] == [0.93]
    assert second["earlier_runs"][0]["found"] == ["x NV", "x PA"]
    assert second["earlier_runs"][0]["recall"] == first["recall"]["pooled"]
    flagged = {("b", 1), ("b", 2)}
    now = run_configuration(True, flagged)
    assert now == {k: second[k] for k in ("semantic_threshold", "state_boilerplate_flag")} | {
        "semantic_threshold": THRESHOLD
    }
    # Run at the current configuration, and at the one before it, nothing more is allowed.
    with pytest.raises(CommissionError, match="already run"):
        check_held_out_configuration(with_earlier_runs(_held_out_run(THRESHOLD, []), None), now)
    with pytest.raises(CommissionError, match="already run"):
        check_held_out_configuration(
            second, {k: first[k] for k in ("semantic_threshold", "state_boilerplate_flag")}
        )
    check_held_out_configuration(None, now)
    check_held_out_configuration(second, run_configuration(True, {("b", 1)}))

"""The commission recall eval: miss reasons stage by stage, and the pipeline regression test.

The headline number queries with each act's own text, which cannot be committed. CI instead
runs the same eval on a committed fixture (`evals/fixture.py`): the enacting bills among
distractors, each act queried with its earliest enacted state version. That is a regression
test of the pipeline, not the recall the README reports.
"""

import json
from dataclasses import replace
from pathlib import Path

import duckdb
import numpy as np
import pytest
from typer.testing import CliRunner

from lineage.acts import ACTS
from lineage.cli import app
from lineage.commission import MISS_REASONS, Probe, Relative, evaluate, probe, target_reason
from lineage.lsh import BANDS, BUCKET_CAP, ROWS
from lineage.minhash import NUM_PERM
from lineage.queries import EnactmentBill
from lineage.sessions import load_sessions
from lineage.ulc import ENACTMENTS, load_enactments
from tests.conftest import COMMISSION
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


def test_reasons_follow_the_pipeline_from_source_text_to_cutoff() -> None:
    proposed = Probe("verification", 0.4)

    def reason(bill: EnactmentBill, rows: int = 3, probed: Probe | None = proposed) -> str | None:
        return target_reason(bill, signed_rows=rows, probed=probed, relative=None)

    assert reason(replace(BILL, text_chars=0, amendment_only=True)) == "no_text_in_source"
    assert reason(replace(BILL, amendment_only=True, eligible=False)) == "amendment_only"
    assert reason(replace(BILL, eligible=False)) == "excluded_type"
    assert reason(BILL, rows=0, probed=None) == "no_usable_text"
    assert reason(BILL) == "verification"
    assert reason(BILL, probed=Probe("not_proposed", 0.1)) == "not_proposed"
    ranked = target_reason(BILL, signed_rows=3, probed=None, relative=Relative(51, 0.1, "HB 12"))
    assert ranked == "below_cutoff"
    found = target_reason(BILL, signed_rows=3, probed=None, relative=Relative(50, 0.1, "HB 12"))
    assert found is None
    assert list(MISS_REASONS)[-1] == "below_cutoff"


def _raw_text(tables: Path, bill_id: str) -> str:
    with duckdb.connect() as con:
        row = con.execute(
            "SELECT raw_text FROM read_parquet($p) WHERE id = $i",
            {"p": str(tables / "bills" / "*.parquet"), "i": bill_id},
        ).fetchall()
    return str(row[0][0])


# Measured on the fixture when the test was written. A pipeline change that loses one of these
# fails here; one that finds more should add it, so the floor only rises.
REGRESSION_FOUND = {
    ("electronic-estate-planning-documents", "MN", 2026),
    ("electronic-estate-planning-documents", "UT", 2026),
    ("public-expression-protection", "IA", 2025),
    ("public-expression-protection", "MT", 2025),
    ("special-deposits", "UT", 2025),
}


def test_pipeline_regression_the_enacting_bills_are_found_among_distractors(
    commission_corpus: Path,
) -> None:
    build = json.loads((COMMISSION / "build.json").read_text())
    queries = {act: _raw_text(commission_corpus, bill) for act, bill in build["queries"].items()}
    snapshot = load_enactments(ENACTMENTS)
    with duckdb.connect() as con:
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
        queries, replace(snapshot, rows=rows), commission_corpus,
        load_sessions(COMMISSION / "sessions.csv"),
    )  # fmt: skip

    found = {(e["act"], e["state"], e["year"]) for e in result["enactments"] if e["found"]}
    assert found >= REGRESSION_FOUND, f"lost: {sorted(REGRESSION_FOUND - found)}"
    for e in result["enactments"]:
        assert e["found"] or e["miss_reason"] in MISS_REASONS
        if not e["with_text"] and e["bills"]:
            assert e["miss_reason"] == "no_text_in_source", e


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
            "eval", "commission", "--tables", str(commission_corpus), "--acts", str(acts),
            "--sessions", str(COMMISSION / "sessions.csv"), "--results", str(results),
        ],
    )  # fmt: skip

    assert out.exit_code == 0, out.output
    doc = json.loads(results.read_text())
    assert doc["other"] == {"kept": True}
    section = doc["commission"]
    assert section["command"] == "uv run lineage eval commission"
    assert set(section["queries"]) == set(ACTS)
    assert len(section["queries"][ACTS[0]]["file_sha256"]) == 64
    assert section["enactments_file"]["read"].startswith("ULC Bill List, read ")
    assert "recall" in out.output


def test_the_command_says_which_act_text_is_missing(
    commission_corpus: Path, tmp_path: Path
) -> None:
    out = CliRunner().invoke(
        app, ["eval", "commission", "--tables", str(commission_corpus), "--acts", str(tmp_path)]
    )

    assert out.exit_code == 1
    assert ACTS[0] in out.output


@pytest.mark.parametrize("reason", MISS_REASONS)
def test_every_miss_reason_is_documented(reason: str) -> None:
    design = (Path(__file__).parents[1] / "DESIGN.md").read_text()
    assert f"`{reason}`" in design

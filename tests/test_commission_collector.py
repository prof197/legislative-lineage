"""The commission's collector (DESIGN.md#model-documents): the acts on disk with provenance, then
the rest from the site, against recorded responses (tests/fixtures/collect/), never the network.
Every act text here is synthetic; no commission text is committed."""

import json
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.origins.collect import CollectContext, CollectReport, collectors
from lineage.origins.collect_base import Fetched
from lineage.origins.collectors import commission
from lineage.origins.collectors.commission import (
    DOCX,
    SOURCE,
    act_page,
    collect_from,
    enacted_in_corpus_years,
)
from lineage.origins.modeldocs import load_text, raw_path, read_docs, read_versions, sha256
from lineage.origins.modeltext import clean_act
from tests.recorded_site import SYNTHETIC_ACT, RecordedSite, docx_bytes, recorded_site

DAY = date(2026, 11, 15)
SITE = "https://www.uniformlaws.org"
HEALTH_CARE_FILES = f"{SITE}/HigherLogic/System/DownloadDocumentFile.ashx?DocumentFileKey="
WITHHELD = (
    f"{SITE}/viewdocument/withheld-final-act?CommunityKey=00000000-0000-4000-8000-0000000000a2"
    "&tab=librarydocuments"
)
WANTED = {
    "health-care-decisions",
    "antitrust-pre-merger-notification",
    "partition-of-lantern-property",
    "harbor-lights",
    "pewter-lamp",
    "assignment-for-benefit-of-creditors",
    "never-listed",
}


def _disk(tmp_path: Path) -> tuple[Path, Path]:
    """Synthetic texts saved as two of the acts lineage.evals.acts knows; the other seven are
    absent."""
    acts, held_out = tmp_path / "acts", tmp_path / "acts" / "held-out"
    held_out.mkdir(parents=True)
    text = SYNTHETIC_ACT.read_text()
    (acts / "public-expression-protection.txt").write_text(text)
    (held_out / "assignment-for-benefit-of-creditors.txt").write_text(
        text.replace("Lantern", "Lamp")
    )
    return acts, held_out


def _run(tmp_path: Path, site: RecordedSite, *, on_site: bool, on: date = DAY) -> CollectReport:
    if not (tmp_path / "acts").exists():
        _disk(tmp_path)
    acts, held_out = tmp_path / "acts", tmp_path / "acts" / "held-out"
    return collect_from(
        CollectContext(root=tmp_path / "modeldocs", on=on),
        fetch=site.fetch,
        robots_reader=site.read_robots,
        acts_dir=acts,
        held_out_dir=held_out,
        site=on_site,
        wanted=WANTED,
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_the_acts_on_disk_are_collected_with_provenance_and_nothing_fetched(
    tmp_path: Path,
) -> None:
    site = recorded_site()
    report = _run(tmp_path, site, on_site=False)

    assert (report.found, report.collected, report.unchanged) == (9, 2, 0)
    assert report.blocked == []
    # The seven acts not on disk, the two reserve acts among them, are reported, not silently
    # skipped.
    assert len(report.failed) == 7
    assert any(f.reason.startswith("health-care-decisions.txt") for f in report.failed)
    assert all("not on disk" in f.reason for f in report.failed)
    assert site.requests == []
    assert site.robots_reads == [("https", "www.uniformlaws.org")]

    root = tmp_path / "modeldocs"
    docs = {d.id: d for d in read_docs(SOURCE, root)}
    assert sorted(docs) == [
        "commission/assignment-for-benefit-of-creditors",
        "commission/public-expression-protection",
    ]
    doc = docs["commission/public-expression-protection"]
    raw = (tmp_path / "acts" / "public-expression-protection.txt").read_bytes()
    assert doc.title == "Public Expression Protection Act"
    assert doc.url == act_page("4f486460-199c-49d7-9fac-05570be1e7b1")
    assert (doc.robots, doc.leaked, doc.media_type) == ("allowed", False, "text/plain")
    assert doc.collected_on == DAY
    assert doc.raw_sha256 == sha256(raw)
    assert raw_path(doc, root).read_bytes() == raw
    assert load_text(doc, root) == clean_act(raw.decode())


def test_a_rerun_changes_nothing(tmp_path: Path) -> None:
    first = _run(tmp_path, recorded_site(), on_site=True)
    before = _tree(tmp_path / "modeldocs")
    again = _run(tmp_path, recorded_site(), on_site=True, on=date(2026, 11, 16))

    assert _tree(tmp_path / "modeldocs") == before
    assert again.collected == 0
    assert again.unchanged == first.collected
    assert (again.blocked, again.failed) == (first.blocked, first.failed)


def test_the_rest_come_from_the_site_and_every_one_found_is_accounted_for(
    tmp_path: Path,
) -> None:
    site = recorded_site()
    report = _run(tmp_path, site, on_site=True)

    assert (report.found, report.collected, report.unchanged) == (8, 4, 0)
    assert report.blocked == [WITHHELD]
    failed = {f.url: f.reason for f in report.failed}
    harbor = f"{SITE}/committees/community-home?CommunityKey=00000000-0000-4000-8000-0000000000b2"
    pewter = f"{SITE}/committees/community-home?CommunityKey=00000000-0000-4000-8000-0000000000d1"
    assert failed[harbor] == "no Final Act document on the act's page"
    assert "HTTP 404" in failed[pewter]
    assert (
        "never-listed: enacted in the corpus years but not in the catalog"
        in failed[f"{SITE}/acts/catalog/current"]
    )

    asked = [url for url, _ in site.requests]
    # Robots first: the disallowed library page is never asked for. The edition without
    # comments is chosen over the PDF; an act on disk, or not wanted, is never fetched.
    assert WITHHELD not in asked
    assert f"{HEALTH_CARE_FILES}00000000-0000-4000-8000-00000000c002&forceDialog=0" in asked
    assert not any("c001" in url or "c003" in url for url in asked)
    assert not any("0000000000a1" in url or "b7e5e644" in url for url in asked)
    assert all(interval >= 1.0 for _, interval in site.requests)
    assert len(site.robots_reads) == 1

    docs = {d.id: d for d in read_docs(SOURCE, tmp_path / "modeldocs")}
    assert sorted(docs) == [
        "commission/assignment-for-benefit-of-creditors",
        "commission/health-care-decisions",
        "commission/partition-of-lantern-property",
        "commission/public-expression-protection",
    ]
    word = docs["commission/health-care-decisions"]
    assert (word.title, word.media_type, word.robots) == (
        "Health-Care Decisions Act",
        DOCX,
        "allowed",
    )
    assert (
        word.url
        == f"{SITE}/committees/community-home?CommunityKey=00000000-0000-4000-8000-0000000000b1"
    )
    expected = clean_act(SYNTHETIC_ACT.read_text())
    assert load_text(word, tmp_path / "modeldocs") == expected
    pdf = docs["commission/partition-of-lantern-property"]
    assert pdf.media_type == "application/pdf"
    assert load_text(pdf, tmp_path / "modeldocs").split() == expected.split()


def test_a_revised_file_on_the_site_becomes_a_new_version(tmp_path: Path) -> None:
    _run(tmp_path, recorded_site(), on_site=True)
    site = recorded_site()
    file_url = f"{HEALTH_CARE_FILES}00000000-0000-4000-8000-00000000c002&forceDialog=0"
    site.overrides[file_url] = docx_bytes(SYNTHETIC_ACT.read_text().replace("Lantern", "Torch"))
    report = _run(tmp_path, site, on_site=True, on=date(2026, 12, 1))

    assert report.collected == 1
    versions = [
        v
        for v in read_versions(SOURCE, tmp_path / "modeldocs")
        if v.doc_id == "commission/health-care-decisions"
    ]
    assert [(v.first_seen, v.replaced_on) for v in versions] == [
        (DAY, date(2026, 12, 1)),
        (date(2026, 12, 1), None),
    ]


def test_an_unreadable_file_fails_its_act_only(tmp_path: Path) -> None:
    site = recorded_site()
    site.overrides[f"{HEALTH_CARE_FILES}00000000-0000-4000-8000-00000000c002&forceDialog=0"] = b"x"
    report = _run(tmp_path, site, on_site=True)

    assert report.collected == 3
    assert any("unreadable Word document" in f.reason for f in report.failed)


def test_the_source_and_the_acts_it_looks_for() -> None:
    assert collectors()["commission"].source == SOURCE
    # The terms as read, stated for the sources page: what they allow, and why fetching is off.
    assert "PENDING" not in SOURCE.terms_note
    assert "noncommercial" in SOURCE.terms_note
    assert "fetching from the site stays off" in SOURCE.terms_note
    assert SOURCE.terms_read_on == date(2026, 11, 17)
    assert not SOURCE.leaked
    # From the committed enactment lists: the development, held-out and reserve acts.
    assert {"public-expression-protection", "directed-trust", "health-care-decisions"} <= (
        enacted_in_corpus_years()
    )


def test_the_command_collects_the_acts_on_disk_without_fetching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acts, held_out = _disk(tmp_path)
    site = recorded_site()

    def no_fetch(url: str, interval: float) -> Fetched:
        raise AssertionError(f"fetched {url} with the site off")

    monkeypatch.setattr(commission, "ACTS_DIR", acts)
    monkeypatch.setattr(commission, "HELD_OUT_DIR", held_out)
    monkeypatch.setattr(commission, "read_robots", site.read_robots)
    monkeypatch.setattr(commission, "fetch_url", no_fetch)
    args = ["collect-models", "commission", "--modeldocs", str(tmp_path / "modeldocs"),
            "--reports", str(tmp_path / "reports"), "--min-free-gib", "0"]  # fmt: skip
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "9 found: 2 collected, 0 unchanged, 0 blocked by robots.txt, 7 failed" in first.output
    report = json.loads((tmp_path / "reports" / "commission.json").read_text())
    # The report names no local path: it is committed.
    assert str(tmp_path) not in json.dumps(report)
    again = CliRunner().invoke(app, args)
    assert "9 found: 0 collected, 2 unchanged" in again.output

"""ALEC's collector (DESIGN.md#model-documents): the model policy library found through the
sitemaps, each policy's type and status read from its page, its text from the site's API or an
attached file, against synthetic responses (tests/fixtures/collect/alec/), never the network.
Every text here is invented; no ALEC text is committed.

The real-sample tests read collected texts from the local data directory (LINEAGE_DATA) and are
skipped until integration has collected them; they assert counts only, never print a text."""

import json
import re
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.core.robots import Robots, parse_robots
from lineage.origins.collect import CollectContext, CollectReport, collectors
from lineage.origins.collect_base import Fetched, provenance
from lineage.origins.collectors import alec
from lineage.origins.collectors.alec import (
    DOCX,
    LIBRARY,
    SOURCE,
    ApiPolicy,
    api_policy,
    attachment,
    body_kind,
    clean_policy,
    collect_from,
    has_law,
    local_measure,
    policy_start,
)
from lineage.origins.modeldocs import (
    MODELDOCS_DIR,
    ModelDoc,
    load_text,
    raw_path,
    read_docs,
    read_versions,
    sha256,
    store,
)
from lineage.origins.modeltext import ModelTextError, docx_text
from lineage.text.extract import html_text, pdf_text
from tests.recorded_site import COLLECT, RecordedSite, docx_bytes

DAY = date(2026, 11, 15)
SITE = "https://alec.org"
FIXTURES = COLLECT / "alec"
AMENDMENT = FIXTURES / "synthetic_amendment.txt"
FILE = f"{SITE}/wp-content/uploads/2026/01/Synthetic-Lantern-Amendments.docx"
WITHHELD = f"{SITE}/model-policy/withheld-lamp-act/"

LANTERN = (
    "BE IT ENACTED BY THE LEGISLATURE OF THE STATE OF\n"
    "Section 1. {Title}\n"
    "This Act may be cited as the Synthetic Lantern Registry Act.\n"
    "Section 2. {Definitions}\n"
    "(A) “Lantern” means a portable lamp kept under.\n"
    "(B) “Registry” means the list kept by the [secretary of state].\n"
    "Section 3. {Registration}\n"
    "A keeper shall register each lantern within [60] days with.\n"
    "A fee of twelve dollars applies.\n"
)


def policy(slug: str) -> str:
    return f"{SITE}/model-policy/{slug}/"


def api(slug: str) -> str:
    return (
        f"{SITE}/wp-json/wp/v2/model-policy?slug={slug}&_fields=id%2Cslug%2Clink%2Ctitle%2Ccontent"
    )


def recorded() -> RecordedSite:
    """The fixture's responses; a `json` entry is a body written as JSON."""
    raw: dict[str, dict[str, object]] = json.loads((FIXTURES / "site.json").read_text())
    raw.pop("_note")
    responses = {
        url: {k: (json.dumps(v) if k == "json" else str(v)) for k, v in entry.items()}
        for url, entry in raw.items()
    }
    for entry in responses.values():
        if "json" in entry:
            entry["body"] = entry.pop("json")
    return RecordedSite(responses, (FIXTURES / "robots.txt").read_bytes())


def _run(
    tmp_path: Path, site: RecordedSite, *, on: date = DAY, on_site: bool = True
) -> CollectReport:
    return collect_from(
        CollectContext(root=tmp_path / "modeldocs", on=on),
        fetch=site.fetch,
        robots_reader=site.read_robots,
        site=on_site,
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_the_library_is_collected_with_provenance_and_every_policy_found_accounted_for(
    tmp_path: Path,
) -> None:
    site = recorded()
    report = _run(tmp_path, site)

    # Ten policies of the kinds collected. The statements, the ordinances, the drafts and the
    # list of a meeting's policies are not model state legislation: neither collected nor
    # counted, whether the page's Type says so or, on an older page without one, its body does.
    assert (report.found, report.collected, report.unchanged) == (10, 6, 0)
    assert report.blocked == [WITHHELD]
    failed = {f.url: f.reason for f in report.failed}
    assert failed == {
        policy("hollow-lamp-act"): "the API lists no policy hollow-lamp-act",
        policy("missing-lamp-act"): "HTTP 404",
        policy("renamed-lamp-act"): "no section, clause or provision in the body",
    }

    asked = [url for url, _ in site.requests]
    # robots.txt first: the disallowed page is never asked for. A policy its page rules out
    # costs that page only; the archive's own page and its pages of cards are skipped.
    assert WITHHELD not in asked
    assert policy("statement-of-principles-on-lanterns") in asked
    assert not any(
        slug in url and "wp-json" in url
        for url in asked
        for slug in ("statement-of-principles-on-lanterns", "draft-", "lamp-post")
    )
    assert f"{SITE}/model-policy/" not in asked
    assert f"{SITE}/model-policy/page/2/" not in asked
    assert f"{SITE}/post-sitemap.xml" not in asked
    assert all(interval >= 1.0 for _, interval in site.requests)
    assert site.robots_reads == [("https", "alec.org")]

    root = tmp_path / "modeldocs"
    docs = {d.id: d for d in read_docs(SOURCE, root)}
    assert sorted(docs) == [
        "alec/amendments-to-lantern-registry-act",
        "alec/headed-lamp-act",
        "alec/lamp-keepers-recital",
        "alec/lantern-registry-act",
        "alec/older-lamp-act",
        "alec/resolution-in-support-of-lamp-week",
    ]
    lantern = docs["alec/lantern-registry-act"]
    assert (lantern.title, lantern.url) == ("Lantern Registry Act", policy("lantern-registry-act"))
    assert (lantern.robots, lantern.leaked, lantern.media_type) == (
        "allowed",
        False,
        "application/json",
    )
    # The organization's own date: the latest it gives, the amendment's over the finalizing.
    assert (lantern.published, lantern.collected_on) == (date(2018, 9, 18), DAY)
    body = site.fetch(api("lantern-registry-act"), 1.0).body
    assert lantern.raw_sha256 == sha256(body)
    assert raw_path(lantern, root).read_bytes() == body
    assert load_text(lantern, root) == LANTERN

    resolution = docs["alec/resolution-in-support-of-lamp-week"]
    # The API title is HTML: entities decoded, a no-break space read as a space.
    assert resolution.title == "Resolution in Support of Lamp Week \u2013 Invented"
    assert resolution.published == date(2018, 9, 18)
    assert load_text(resolution, root).split("\n")[0].startswith("WHEREAS, invented lamps")

    amendment = docs["alec/amendments-to-lantern-registry-act"]
    assert (amendment.media_type, amendment.published) == (DOCX, date(2026, 3, 3))
    assert amendment.raw_sha256 == sha256(docx_bytes(AMENDMENT.read_text()))
    assert load_text(amendment, root) == (
        "Amendments to the Synthetic Lantern Registry Act\n"
        "Section 3. {Registration}\nA keeper shall register each lantern within [90] days with.\n"
    )
    assert FILE in asked


def test_older_layouts_are_read_by_what_their_pages_and_bodies_give(tmp_path: Path) -> None:
    site = recorded()
    _run(tmp_path, site)
    root = tmp_path / "modeldocs"
    docs = {d.id: d for d in read_docs(SOURCE, root)}

    # A Type and a Date Amended but no Status: final. Its summary, the label that ends it, the
    # drafter's placeholders, unpaired brackets and the source line go.
    older = docs["alec/older-lamp-act"]
    assert older.published == date(2013, 1, 29)
    assert load_text(older, root) == (
        "Sec 1. {Short Title.} This Act may be cited as the Older Lamp Act.\n"
        "Section 2. {Registry.} The shall keep the registry of lamps under.\n"
    )
    # No Type: its body's label makes it a resolution, read from the first recital.
    recital = docs["alec/lamp-keepers-recital"]
    assert recital.published is None
    assert load_text(recital, root) == (
        "WHERAS, invented keepers tend invented lamps; and\n"
        "THEREFORE BE IT RESOLVED, that the legislature of thanks them.\n"
    )
    # No section numbers: the body is the policy from its first line.
    assert load_text(docs["alec/headed-lamp-act"], root) == (
        "Definitions\n"
        "(1) “Lamp” means an invented light.\n"
        "Keeper Operations\n"
        "A keeper shall trim each lamp.\n"
        "A keeper may not sell oil.\n"
    )
    # A page ruled out by what its body is still cost the API request that told it.
    asked = [url for url, _ in site.requests]
    for slug in ("lamp-policies-annual-meeting", "statement-of-principles-on-wicks"):
        assert api(slug) in asked
    assert not any("draft-wick-act" in url and "wp-json" in url for url in asked)


def test_a_policy_stored_before_and_now_out_of_scope_is_withdrawn(tmp_path: Path) -> None:
    root = tmp_path / "modeldocs"
    slug = "lamp-lighting-ordinance"  # its page types it a model policy; its text is local
    text = "Section 1. A lamp.\n"
    earlier = provenance(
        SOURCE, slug=slug, title="Lamp Lighting Ordinance", url=policy(slug), raw=b"raw",
        media_type="application/json", text=text, robots="allowed", collected_on=date(2026, 11, 1),
    )  # fmt: skip
    store(SOURCE, earlier, raw=b"raw", text=text, root=root)

    report = _run(tmp_path, recorded())

    assert report.withdrawn == [f"alec/{slug}"]
    assert f"alec/{slug}" not in {d.id for d in read_docs(SOURCE, root)}
    (version,) = [v for v in read_versions(SOURCE, root) if v.doc_id == f"alec/{slug}"]
    assert version.replaced_on == DAY  # closed, its text kept
    assert _run(tmp_path, recorded()).withdrawn == []


def test_a_rerun_changes_nothing(tmp_path: Path) -> None:
    first = _run(tmp_path, recorded())
    before = _tree(tmp_path / "modeldocs")
    again = _run(tmp_path, recorded(), on=date(2026, 11, 16))

    assert _tree(tmp_path / "modeldocs") == before
    assert again.collected == 0
    assert again.unchanged == first.collected
    assert (again.found, again.blocked, again.failed) == (first.found, first.blocked, first.failed)


def test_a_revised_text_becomes_a_new_version(tmp_path: Path) -> None:
    _run(tmp_path, recorded())
    site = recorded()
    site.overrides[FILE] = docx_bytes(AMENDMENT.read_text().replace("[90]", "[120]"))
    report = _run(tmp_path, site, on=date(2026, 12, 1))

    assert (report.collected, report.unchanged) == (1, 5)
    versions = [
        v
        for v in read_versions(SOURCE, tmp_path / "modeldocs")
        if v.doc_id == "alec/amendments-to-lantern-registry-act"
    ]
    assert [(v.first_seen, v.replaced_on) for v in versions] == [
        (DAY, date(2026, 12, 1)),
        (date(2026, 12, 1), None),
    ]


def test_an_unreadable_file_fails_its_policy_only(tmp_path: Path) -> None:
    site = recorded()
    site.overrides[FILE] = b"not a Word file"
    report = _run(tmp_path, site)

    assert (report.found, report.collected) == (10, 5)
    reason = {f.url: f.reason for f in report.failed}[policy("amendments-to-lantern-registry-act")]
    assert "unreadable Word document" in reason


def test_a_host_whose_robots_txt_could_not_be_read_fails_and_nothing_is_fetched(
    tmp_path: Path,
) -> None:
    site = recorded()

    def overloaded(scheme: str, host: str) -> Robots:
        url = f"{scheme}://{host}/robots.txt"
        return parse_robots(url, 503, b"", "2026-11-15T00:00:00+00:00")

    report = collect_from(
        CollectContext(root=tmp_path / "modeldocs", on=DAY),
        fetch=site.fetch,
        robots_reader=overloaded,
        site=True,
    )

    assert site.requests == []
    assert (report.found, report.collected) == (1, 0)
    assert "robots.txt could not be read" in report.failed[0].reason


def test_with_fetching_off_nothing_is_asked_and_the_library_is_reported_unread(
    tmp_path: Path,
) -> None:
    site = recorded()
    report = _run(tmp_path, site, on_site=False)

    assert (site.requests, site.robots_reads) == ([], [])
    assert (report.found, report.collected) == (1, 0)
    assert report.failed[0].url == LIBRARY
    assert "fetching from the site is off" in report.failed[0].reason


def test_the_source_entry() -> None:
    assert collectors()["alec"].source == SOURCE
    assert alec.SITE_FETCH is True
    assert SOURCE.library_url == "https://alec.org/model-policy/"
    assert SOURCE.members_source == "https://alec.org/about/"
    assert SOURCE.terms_url == "https://alec.org/terms-of-use/"
    assert SOURCE.terms_read_on == date(2026, 11, 15)
    assert "501(c)(3)" in SOURCE.descriptor
    assert not SOURCE.leaked


def test_the_command_reports_the_library_unread_without_fetching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_fetch(url: str, interval: float) -> Fetched:
        raise AssertionError(f"fetched {url} with the site off")

    monkeypatch.setattr(alec, "fetch_url", no_fetch)
    monkeypatch.setattr(alec, "read_robots", recorded().read_robots)
    monkeypatch.setattr(alec, "SITE_FETCH", False)
    args = ["collect-models", "alec", "--modeldocs", str(tmp_path / "modeldocs"),
            "--reports", str(tmp_path / "reports"), "--min-free-gib", "0"]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "1 found: 0 collected, 0 unchanged, 0 blocked by robots.txt, 1 failed" in result.output
    report = json.loads((tmp_path / "reports" / "alec.json").read_text())
    assert str(tmp_path) not in json.dumps(report)


# ---- cleaning -------------------------------------------------------------------------------


def test_a_policy_keeps_its_sections_and_loses_labels_notes_approvals_and_instructions() -> None:
    raw = json.loads((FIXTURES / "site.json").read_text())[api("lantern-registry-act")]["json"]
    body = html_text(raw[0]["content"]["rendered"].encode(), "utf-8")
    assert clean_policy(body) == LANTERN
    assert clean_policy(LANTERN) == LANTERN


def test_a_resolution_keeps_its_clauses_from_the_first_whereas() -> None:
    given = (
        "Model Resolution\nSummary\nAn invented summary.\n"
        "WHEREAS, lamps are invented; and\n"
        "NOW, THEREFORE, BE IT RESOLVED, that [state] supports them.\n"
        "BE IT FURTHER RESOLVED, that copies go to [insert recipients].\n"
        "Adopted by the Invented Task Force on May 1, 2020.\n"
    )
    assert clean_policy(given) == (
        "WHEREAS, lamps are invented; and\n"
        "NOW, THEREFORE, BE IT RESOLVED, that [state] supports them.\n"
        "BE IT FURTHER RESOLVED, that copies go to.\n"
    )


@pytest.mark.parametrize(
    ("given", "cleaned"),
    [
        ("under {insert citation} and", "under and"),
        ("kept by {name of agency}.", "kept by."),
        ("under [reference any state lamp code].", "under."),
        ("a fee. {Drafting Note: optional.}", "a fee."),
        ("filed with [cite the code] here", "filed with here"),
    ],
)
def test_instructions_to_the_drafter_go(given: str, cleaned: str) -> None:
    assert clean_policy(f"Section 1. {{Title}}\n{given}\n") == f"Section 1. {{Title}}\n{cleaned}\n"


@pytest.mark.parametrize(
    "line",
    [
        "Approved by ALEC Board of Directors in 1995.",
        "Reapproved by ALEC Board of Directors on January 28, 2013.",
        "(Adopted by the Invented Task Force at the Annual Meeting, July 2015.)",
        "Drafting Note: A state may rename the registry.",
        "{Drafting note: a state may waive it.}",
        "Note: an invented note.",
    ],
)
def test_notes_and_approval_lines_go(line: str) -> None:
    assert clean_policy(f"Section 1. {{Title}}\nA rule.\n{line}\n") == (
        "Section 1. {Title}\nA rule.\n"
    )


@pytest.mark.parametrize(
    "kept",
    [
        "Section 2. {Definitions}",
        "The [secretary of state] keeps it within [60] days.",
        "(C) Approved by the board of directors of the corporation.",
        "The court [shall][may] dismiss it.",
    ],
)
def test_enacted_text_stays_as_written(kept: str) -> None:
    assert clean_policy(f"Section 1. {{Title}}\n{kept}\n") == f"Section 1. {{Title}}\n{kept}\n"


def test_a_text_with_no_section_clause_or_provision_is_refused() -> None:
    for notice in ("(See Amendments Here)\n", "The Lamp Act is now called the Wick Act.\n"):
        with pytest.raises(ModelTextError, match="no section, clause or provision"):
            clean_policy(notice)


@pytest.mark.parametrize(
    ("given", "cleaned"),
    [
        ("The [insert department} shall act.", "The shall act."),
        ("under {insert code reference]; or", "under; or"),
        ("the chamber of [INSERT STATE/COMMONWEALTH) urges", "the chamber of urges"),
        ("posted on the {insert appropriate board website and", "posted on the"),
        (
            "kept in the [Insert state} Treasury account [60] days",
            "kept in the Treasury account [60] days",
        ),
    ],
)
def test_an_instruction_whose_brackets_do_not_pair_goes_and_no_more(
    given: str, cleaned: str
) -> None:
    assert clean_policy(f"Section 1. {{Title}}\n{given}\nSection 2. {{Effective date.}}\n") == (
        f"Section 1. {{Title}}\n{cleaned}\nSection 2. {{Effective date.}}\n"
    )


@pytest.mark.parametrize(
    "line",
    [
        "Renewed by the Task Forces on Invented Affairs on July 16-17, 2020",
        "Renewal approved by the Board of Directors on August 9, 2020",
        "Amended by the ALEC Board of Directors on August 30, 2024.",
        "Source Book of American State Legislation 1987-88",
        "{Title, enacting clause, etc.}",
        "Model Legislation",
        "Model Resolution:",
    ],
)
def test_the_older_policies_marks_go(line: str) -> None:
    assert clean_policy(f"Sec 1. {{Title}} A rule shall stand.\n{line}\n") == (
        "Sec 1. {Title} A rule shall stand.\n"
    )


@pytest.mark.parametrize(
    ("given", "cleaned"),
    [
        # A summary, ended by its label; numbered points in it are not provisions.
        (
            "An Invented Act\nSummary\nThis Act would light lamps.\n1. It lights them.\n"
            "Model Legislation\n(A) A keeper shall light each lamp.\n",
            "(A) A keeper shall light each lamp.\n",
        ),
        # No label: the summary runs to the first section or clause.
        (
            "Summary: This Act would light lamps.\nBe it enacted by the Legislature:\n"
            "A keeper shall light each lamp.\n",
            "Be it enacted by the Legislature:\nA keeper shall light each lamp.\n",
        ),
        # A one-paragraph summary after the enacting clause.
        (
            "BE IT ENACTED BY THE LEGISLATURE:\nSummary: This bill would light lamps.\n"
            "100.01 Definitions.\n",
            "BE IT ENACTED BY THE LEGISLATURE:\n100.01 Definitions.\n",
        ),
        # A label first, then the summary.
        (
            "Model Legislation\nSummary\nAn invented summary.\nSection 1. {Title}\nA rule.\n",
            "Section 1. {Title}\nA rule.\n",
        ),
        # No summary: everything from the top is the policy, sections or not.
        (
            "All fees collected by the State shall fund roads.\n(A) Maintenance;\n"
            "Section 2 {Severability clause.}\n",
            "All fees collected by the State shall fund roads.\n(A) Maintenance;\n"
            "Section 2 {Severability clause.}\n",
        ),
        (
            "An Act concerning lamps.\nBe it enacted by the Legislature of this state:\n"
            "Sec 1. (a) A keeper shall light each lamp.\nSec. 2. This act takes effect.\n",
            "An Act concerning lamps.\nBe it enacted by the Legislature of this state:\n"
            "Sec 1. (a) A keeper shall light each lamp.\nSec. 2. This act takes effect.\n",
        ),
        (
            "WHEREAS, lamps are invented; and\nNOW, THEREFORE, the State amends its code.\n"
            "Section 1. A keeper shall light each lamp.\n",
            "WHEREAS, lamps are invented; and\nNOW, THEREFORE, the State amends its code.\n"
            "Section 1. A keeper shall light each lamp.\n",
        ),
    ],
)
def test_a_policy_begins_after_its_summary_or_at_its_top(given: str, cleaned: str) -> None:
    assert clean_policy(given) == cleaned


@pytest.mark.parametrize(
    ("title", "body", "kind"),
    [
        ("Statement of Principles on Wicks", "Wicks shall be trimmed.", "Statement of Principle"),
        ("Lamp Lighting Ordinance", "Section 1. Lamps shall be lit.", "Model Ordinance"),
        ("Lamp Keepers Recital", "Summary\nModel Resolution\nWHEREAS, lamps", "Model Resolution"),
        ("Older Lamp Act", "Summary\nModel Legislation\nSection 1.", "Model Policy"),
        ("Resolution on Wicks", "WHEREAS, wicks burn;", "Model Resolution"),
        ("Headed Lamp Act", "Definitions\n(1) Lamp means a light.", "Model Policy"),
        ("Lamp Policies Annual Meeting", "Members met.\nLantern Registry Act", None),
    ],
)
def test_an_older_page_without_a_type_is_told_by_its_body(
    title: str, body: str, kind: str | None
) -> None:
    assert body_kind(title, body) == kind


def test_a_list_of_other_policies_is_not_one_though_its_summaries_say_shall() -> None:
    content = (
        '<p><a href="https://alec.org/model-policy/lantern-registry-act/">Lantern Act</a></p>'
        "<p>Keepers shall register lanterns.</p>"
        "<p><a href='/model-policy/wick-act/'>Wick Act</a></p><p>Wicks shall be trimmed.</p>"
    )
    body = html_text(content.encode(), "utf-8")
    assert has_law(body)
    assert body_kind("Lamp Policies Annual Meeting", body, content) is None


@pytest.mark.parametrize(
    ("title", "body", "local"),
    [
        ("Lamp Lighting Ordinance", "Section 1. Lamps shall be lit.", True),
        ("Proclamation Recognizing Lamp Day", "I do hereby proclaim it.", True),
        ("Wick Act", "NOW, THEREFORE, BE IT ORDAINED BY THE COUNCIL:", True),
        ("Wick Resolution", "THE [Insert Governing Body] OF THE [Insert Jurisdiction]", True),
        (
            "Prohibition on Local Ordinances Restricting Lamps Act",
            "No city shall ban lamps.",
            False,
        ),
        ("Wick Act", "The county commission shall report to the state.", False),
    ],
)
def test_a_local_measure_is_told_by_its_title_or_text(title: str, body: str, local: bool) -> None:
    assert local_measure(title, body) is local


def test_an_amendment_is_read_from_the_file_it_links() -> None:
    assert attachment('<p><a href="/wp-content/uploads/a.docx">(See Amendments Here)</a></p>') == (
        f"{SITE}/wp-content/uploads/a.docx"
    )
    assert attachment("<p><a href='/model-policy/other/'>other</a></p>") is None
    response = Fetched(api("x"), "application/json", b'[{"slug": "y"}]')
    with pytest.raises(Exception, match="lists no policy x"):
        api_policy(response, "x")
    got = api_policy(
        Fetched(api("x"), "", b'[{"slug": "x", "title": {"rendered": "T"}, "content": {}}]'), "x"
    )
    assert got == ApiPolicy(title="T", content="")


# ---- real samples, local only ---------------------------------------------------------------
#
# Each reads the collected policies' stored raw files (the API's answer, or an attached file)
# and asserts counts only, so a failure never prints a policy's text. The first compares the
# stored texts with a fresh cleaning, so it fails until integration collects again after the
# cleaning changes; the others check a fresh cleaning of each raw file, so they hold either way.

# Measured independently of the collector's own rules.
_INSTRUCTION_LEFT = re.compile(
    r"[\[{]\s*(?:cite|citation|insert|specify|reference)\b|\{\s*title,\s*enacting", re.I
)
_NOTE_LEFT = re.compile(r"^\s*[\[{(]?\s*(?:(?:drafting|legislative)\s+notes?\b|notes?\s*:)", re.I)
_APPROVAL_LEFT = re.compile(
    r"^\s*\(?\s*(?:re-?)?(?:approved|adopted|renewed|renewal\s+approved)\s+by\b.{0,80}"
    r"\b(?:board\s+of\s+directors|task\s+forces?)\b",
    re.IGNORECASE,
)
# A summary's heading, and a Utah bill's descriptions that stand where a summary would.
_SUMMARY_LEFT = re.compile(
    r"^\s*(?:(?:model\s+policy\s+)?summary\b|general\s+description\s*:|highlighted\s+provisions"
    r"\s*:?\s*$|the\s+problem\s*$)",
    re.IGNORECASE,
)
_LABEL_LEFT = re.compile(r"^\s*model\s+(?:legislation|policy|resolution)\s*:?\s*$", re.I)
_SECTION_OR_CLAUSE = re.compile(
    r"^\s*[\[“\"]?(?:new\s+section\.?\s*)?(?:section|sec\.?)\s+(?:\d|[ivx]+\b|x+\b|_)"
    r"|^\s*(?:where?as\b|(?:now,?\s+)?(?:therefore,?\s+)?be\s+it\s+(?:enacted|resolved))",
    re.IGNORECASE,
)


def _collected() -> list[ModelDoc]:
    try:
        docs = read_docs(SOURCE, MODELDOCS_DIR)
    except (OSError, ValueError):
        return []
    return [d for d in docs if raw_path(d).exists()]


def _body(doc: ModelDoc) -> str:
    """A collected policy's stored raw file, read as the collector reads it."""
    raw = raw_path(doc).read_bytes()
    if doc.media_type == "application/json":
        return html_text(json.loads(raw)[0]["content"]["rendered"].encode(), "utf-8")
    return docx_text(raw) if doc.media_type == DOCX else pdf_text(raw)


@pytest.mark.skipif(not _collected(), reason="no ALEC policies under the data directory")
def test_every_collected_policys_stored_text_is_its_cleaning_and_in_scope() -> None:
    docs = _collected()
    rereads = sum(clean_policy(_body(d)) == load_text(d) for d in docs)
    local = sum(local_measure(d.title, _body(d)) for d in docs)
    assert (rereads, local) == (len(docs), 0)


@pytest.mark.skipif(not _collected(), reason="no ALEC policies under the data directory")
def test_every_collected_policy_cleans_to_its_law_and_nothing_else() -> None:
    texts = [clean_policy(_body(d)) for d in _collected()]
    lines = [line for text in texts for line in text.split("\n") if line]

    assert sum(has_law(text) for text in texts) == len(texts)
    assert sum(clean_policy(text) == text for text in texts) == len(texts)
    assert len([m for text in texts for m in _INSTRUCTION_LEFT.finditer(text)]) == 0
    assert sum(1 for line in lines if _NOTE_LEFT.match(line)) == 0
    assert sum(1 for line in lines if _APPROVAL_LEFT.match(line)) == 0
    assert sum(1 for line in lines if _SUMMARY_LEFT.match(line) or _LABEL_LEFT.match(line)) == 0


@pytest.mark.skipif(not _collected(), reason="no ALEC policies under the data directory")
def test_only_a_summary_is_cut_from_the_front_of_a_policy() -> None:
    cut = with_law = without_summary = 0
    for doc in _collected():
        lines = [line.strip() for line in _body(doc).split("\n") if line.strip()]
        front = lines[: policy_start(lines)]
        if front:
            cut += 1
            with_law += any(_SECTION_OR_CLAUSE.match(line) for line in front)
            without_summary += not any(_SUMMARY_LEFT.match(line) for line in front)
    # The front cut is an older body's summary, never a section or clause.
    assert (with_law, without_summary) == (0, 0)
    assert cut <= 0.1 * len(_collected())


@pytest.mark.skipif(not _collected(), reason="no ALEC policies under the data directory")
def test_cleaning_keeps_most_of_each_collected_policys_words() -> None:
    docs = _collected()
    low_of_body = low_of_policy = 0
    for doc in docs:
        lines = [line.strip() for line in _body(doc).split("\n") if line.strip()]
        kept = len(clean_policy(_body(doc)).split())
        # Labels, notes, approval lines and instructions are a few words; losing more is
        # losing law. Measured against the whole body, and again from where the policy
        # begins, where a lost word of law weighs more: an older body's summary, up to 1,400
        # words, is not law, and the test above holds that only a summary is cut from the front.
        low_of_body += kept < 0.8 * sum(len(line.split()) for line in lines)
        low_of_policy += kept < 0.8 * sum(len(x.split()) for x in lines[policy_start(lines) :])
    assert low_of_body <= 0.05 * len(docs)
    assert low_of_policy <= 0.05 * len(docs)

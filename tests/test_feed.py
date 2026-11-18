"""The Atom feed (DESIGN.md#updates, Publishing): its events from the change log, the group of
bills from three or more states first seen within 14 days, entries that state dates and shared
text only, and the check against RFC 4287's required elements."""

import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pytest

from lineage import feed
from lineage.export.atom import ATOM, violations
from lineage.feed import (
    BillInfo,
    DocInfo,
    Event,
    FeedError,
    check,
    entry,
    feed_events,
    render,
    write_feed,
)
from lineage.publish import FEED_FILE
from lineage.updates import (
    Change,
    OriginGained,
    RecordAdded,
    RelativeGained,
    RelativeLost,
)
from tests.days import DAYS
from tests.live import Recorded, export_inputs, full_export, recorded

SITE = "https://example.org/lineage"
D0 = date(2026, 1, 5)
# Words that would say who wrote a bill or where its text came from.
ATTRIBUTION = re.compile(
    r"\b(wrote|written|writes|authored|author of|drafted|copied|copies|based on|modeled on|"
    r"sponsor\w*|lobby\w*|behind|originat\w*|ghost\w*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Sources:
    root: Path
    package: str


def d(n: int) -> date:
    return D0 + timedelta(days=n)


def added(n: int, bill: str, state: str) -> RecordAdded:
    return RecordAdded(as_of=d(n), bill_id=bill, state=state, session="s", text_sha256=None)


def linked(n: int, a: str, b: str) -> list[Change]:
    return [
        RelativeGained(as_of=d(n), bill_id=a, relative_id=b, stages=["near_duplicate"]),
        RelativeGained(as_of=d(n), bill_id=b, relative_id=a, stages=["near_duplicate"]),
    ]


def groups(changes: list[Change], through: date) -> list[tuple[date, tuple[str, ...]]]:
    events = feed_events(changes, hosted_state="GA", through=through, listed_origin=lambda _: True)
    return sorted(
        (e.day, tuple(b for b, _ in e.related)) for e in events if e.kind == "states_group"
    )


def test_a_group_is_reported_when_its_third_state_arrives_and_again_at_a_new_state() -> None:
    log: list[Change] = [
        added(0, "ga1", "GA"),
        added(2, "ut1", "UT"),
        *linked(2, "ga1", "ut1"),
        added(4, "tx1", "TX"),
        *linked(4, "ut1", "tx1"),  # three states, linked through Utah's
        added(6, "tx2", "TX"),
        *linked(6, "tx2", "ga1"),  # a state it already held: nothing new
        added(8, "wa1", "WA"),
        *linked(8, "wa1", "tx2"),  # a fourth
    ]
    assert groups(log, d(10)) == [
        (d(4), ("ga1", "ut1", "tx1")),
        (d(8), ("ga1", "ut1", "tx1", "tx2", "wa1")),
    ]


def test_a_group_needs_every_bill_first_seen_within_14_days() -> None:
    log: list[Change] = [
        added(0, "ga1", "GA"),
        added(13, "ut1", "UT"),
        *linked(13, "ga1", "ut1"),
        added(14, "tx1", "TX"),  # day 14: Georgia's, first seen on day 0, is out of the window
        *linked(14, "ut1", "tx1"),
        *linked(14, "ga1", "tx1"),
    ]
    assert groups(log, d(20)) == []
    log[3] = added(13, "tx1", "TX")  # within it
    log[4:] = [*linked(13, "ut1", "tx1")]
    assert groups(log, d(20)) == [(d(13), ("ga1", "tx1", "ut1"))]


def test_a_bill_from_before_the_live_track_or_a_lost_link_joins_no_group() -> None:
    old = "ocd-bill/from-the-snapshot"  # never added by the live track: no first-seen day
    log: list[Change] = [
        added(1, "ga1", "GA"),
        added(1, "ut1", "UT"),
        *linked(1, "ga1", old),
        *linked(1, "ut1", old),
        added(2, "tx1", "TX"),
        *linked(2, "tx1", "ga1"),
        *linked(2, "ga1", "ut1"),
        RelativeLost(as_of=d(2), bill_id="ga1", relative_id="ut1"),
        RelativeLost(as_of=d(2), bill_id="ut1", relative_id="ga1"),
    ]
    assert groups(log, d(5)) == []


def test_events_are_kept_14_days_and_name_new_and_existing_bills_apart() -> None:
    log: list[Change] = [
        added(0, "ga-new", "GA"),
        *linked(0, "ga-new", "tx-old"),
        added(20, "ut-new", "UT"),
        *linked(20, "ut-new", "tx-old"),
        OriginGained(
            as_of=d(20),
            bill_id="tx-old",
            doc_id="lib/a",
            stages=["semantic"],
            bill_share=0.6,
            model_share=0.2,
        ),
        OriginGained(
            as_of=d(20),
            bill_id="tx-old",
            doc_id="lib/b",
            stages=["semantic"],
            bill_share=0.1,
            model_share=0.05,
        ),
    ]

    def listed(c: OriginGained) -> bool:
        return max(c.bill_share, c.model_share) >= 0.25

    early = feed_events(log, hosted_state="GA", through=d(5), listed_origin=listed)
    assert [(e.kind, e.bill_id) for e in early] == [
        ("new_bill", "ga-new"),  # Georgia's is hosted and new
        ("relative_gained", "tx-old"),
    ]
    late = feed_events(log, hosted_state="GA", through=d(20), listed_origin=listed)
    assert [(e.kind, e.bill_id) for e in late] == [
        ("origin_gained", "tx-old"),
        ("relative_gained", "tx-old"),  # Utah's is new but not hosted: only the existing bill's
    ]
    assert late[0].related == (("lib/a", "0.6000 0.2000"),)  # under the listing floor is no event
    again = feed_events(log, hosted_state="GA", through=d(20), listed_origin=listed)
    assert [e.id for e in again] == [e.id for e in late]  # ids are the event's, every time


def _bills() -> dict[str, BillInfo]:
    return {
        "ga": BillInfo("GA", "SB 1", "Public expression", f"{SITE}/bills/ga-sb-1/", hosted=True),
        "ut": BillInfo("UT", "HB 2", "Public expression", "https://le.utah.gov/hb2", hosted=False),
        "tx": BillInfo("TX", "HB 3", "Public expression", None, hosted=False),
    }


def test_entries_state_dates_and_shared_text_and_never_a_model_text() -> None:
    model_text = "Section 1. A model's own words, which no entry may carry."
    docs = {
        "lib/a": DocInfo(
            title="Model Public Expression Act",
            organization="A Library",
            leaked=True,
            drafted_by="Example Council",
        )
    }
    events = [
        Event("new_bill", d(1), "ga", (("ut", "near_duplicate+semantic"), ("tx", "semantic"))),
        Event("relative_gained", d(1), "ut", (("ga", "near_duplicate"),)),
        Event("origin_gained", d(1), "tx", (("lib/a", "0.6000 0.2500"),)),
        Event(
            "states_group",
            d(1),
            None,
            (("ut", d(0).isoformat()), ("ga", d(1).isoformat()), ("tx", d(1).isoformat())),
        ),
    ]
    entries = [entry(e, _bills(), docs) for e in events]

    for e in entries:
        assert not ATTRIBUTION.search(e.title + e.content), e
        assert model_text not in e.content
        assert e.updated == f"{d(1).isoformat()}T00:00:00Z"
        assert e.id.startswith("urn:uuid:")
    new, gained, origin, group = entries
    assert new.link == f"{SITE}/bills/ga-sb-1/"  # the hosted page
    assert gained.link == "https://le.utah.gov/hb2"  # another state's: its text's source
    assert origin.link is None
    assert "60% of the bill's wording appears in it, covering 25% of the model" in origin.content
    assert (
        "“Model Public Expression Act”, an Example Council model from a leaked document published "
        "by A Library" in origin.content
    )
    assert "(shares wording)" in new.content
    assert "(similar in meaning)" in new.content
    assert group.title == "Bills in 3 states first seen within 14 days share text"
    assert group.link == f"{SITE}/bills/ga-sb-1/"  # its first hosted bill, not its first bill
    check(render(entries, feed_id="urn:uuid:x", site_url=SITE, updated="2026-01-06T00:00:00Z"))


NCOIL = "National Council of Insurance Legislators (NCOIL)"
ALEC = "American Legislative Exchange Council (ALEC)"


def test_a_model_is_named_with_who_published_it_and_a_leaked_one_with_who_drafted_it() -> None:
    assert feed.model_named("Model Act", "Org", None, quoted=False) == (
        "Model Act, a model published by Org"
    )
    archive = "Example Archive Project"
    assert feed.model_named("ALEC Model Act", archive, ALEC, quoted=True) == (
        f"“Model Act”, an {ALEC} model from a leaked document published by {archive}"
    )
    assert feed.model_named("Act", archive, NCOIL, quoted=False).startswith(f"Act, a {NCOIL} model")
    assert feed.model_named("Act", archive, "Uniform Law Commission", quoted=False).startswith(
        "Act, a Uniform"
    )
    assert feed.model_named("Act", archive, "Institute for Justice", quoted=False).startswith(
        "Act, an Institute"
    )


@pytest.mark.parametrize(
    ("title", "organization", "shown"),
    [
        ("NCOIL Insurance Transfer Model Act", NCOIL, "Insurance Transfer Model Act"),
        ("Public Citizen Model State Law: A Right", "Public Citizen", "Model State Law: A Right"),
        ("ALEC's Model Resolution", ALEC, "Model Resolution"),
        ("States Project: Read Act", "The States Project (policy work)", "Read Act"),
        ("Renaming the ALEC Act", ALEC, "Renaming the ALEC Act"),
        ("Model Act \u2013 NCOIL", NCOIL, "Model Act \u2013 NCOIL"),
        ("NCOIL \u2013 Model Act", NCOIL, "Model Act"),
        ("NCOILS Act", NCOIL, "NCOILS Act"),
        ("NCOIL", NCOIL, "NCOIL"),
    ],
)
def test_a_title_opening_with_its_organization_does_not_repeat_it(
    title: str, organization: str, shown: str
) -> None:
    """As the site's origin line does (site/src/lib/origins.test.ts), ruled at the deploy review."""
    assert feed.shown_title(title, organization) == shown


def test_a_withheld_documents_match_is_not_an_event() -> None:
    def gained(doc_id: str, bill: float, model: float) -> OriginGained:
        return OriginGained(
            as_of=d(1),
            bill_id="tx",
            doc_id=doc_id,
            stages=["near_duplicate"],
            bill_share=bill,
            model_share=model,
        )

    assert feed.listed_origin(gained("lib/a", 0.1, 0.6))
    assert not feed.listed_origin(gained("lib/a", 0.1, 0.2))  # under the floor
    assert not feed.listed_origin(gained("lib/a", 0.1, 0.6), frozenset({"lib/a"}))


ID, TITLE, UPDATED = "<id>urn:x</id>", "<title>t</title>", "<updated>2026-01-01T00:00:00Z</updated>"
AUTHOR = "<author><name>a</name></author>"
HEAD = ID + TITLE + UPDATED + AUTHOR
GOOD_ENTRY = f"<entry><id>urn:e</id>{TITLE}{UPDATED}<content>c</content></entry>"


def _feed(entries: str, head: str = HEAD) -> bytes:
    return f'<feed xmlns="{ATOM}">{head}{entries}</feed>'.encode()


@pytest.mark.parametrize(
    ("doc", "problem"),
    [
        (_feed("", TITLE + UPDATED + AUTHOR), "0 atom:id"),
        (_feed("", ID + TITLE + TITLE + UPDATED + AUTHOR), "2 atom:title"),
        (_feed("", ID + TITLE + "<updated>2026-01-01</updated>" + AUTHOR), "not an RFC 3339"),
        (_feed("", ID + TITLE + UPDATED.replace("-01-", "-13-") + AUTHOR), "not an RFC 3339"),
        (_feed("", "<id>x</id>" + TITLE + UPDATED + AUTHOR), "not an absolute IRI"),
        (_feed(GOOD_ENTRY, ID + TITLE + UPDATED), "no atom:author"),
        (_feed(GOOD_ENTRY, HEAD + "<author/>"), "without exactly one atom:name"),
        (_feed('<link href="a"/><link rel="alternate" href="b"/>'), "two alternate links"),
        (_feed("<link/>"), "no href"),
        (_feed(f"<entry>{TITLE}{UPDATED}<content>c</content></entry>"), "entry 1: 0 atom:id"),
        (_feed(f"<entry><id>urn:e</id>{TITLE}{UPDATED}</entry>"), "neither atom:content"),
        (_feed(GOOD_ENTRY.replace("</entry>", "<summary/><summary/></entry>")), "one atom:summary"),
        (_feed(GOOD_ENTRY + GOOD_ENTRY), "another entry's"),
        (b"<rss/>", "not atom:feed"),
        (b"<feed", "not well-formed"),
    ],
)
def test_the_check_fails_on_each_rule_the_rfc_sets(doc: bytes, problem: str) -> None:
    assert any(problem in v for v in violations(doc)), violations(doc)
    with pytest.raises(FeedError, match="RFC 4287"):
        check(doc)


def test_a_feed_with_its_required_elements_passes() -> None:
    assert violations(_feed(GOOD_ENTRY)) == []
    offset = UPDATED.replace("Z", "+02:00")
    linked_only = f"<entry><id>urn:e</id>{TITLE}{offset}<link href='h'/></entry>"
    assert violations(_feed(linked_only)) == []


@pytest.fixture(scope="module")
def rec(tmp_path_factory: pytest.TempPathFactory) -> Recorded:
    return recorded(tmp_path_factory)


@pytest.mark.parametrize("state", ["GA", "UT"])
def test_the_recorded_days_give_a_national_feed_that_passes_the_check(
    rec: Recorded, tmp_path: Path, state: str
) -> None:
    inputs = export_inputs(tmp_path / "inputs")
    export = full_export(rec.tables[3], tmp_path / "export" / state, inputs, state)
    sources = Sources(inputs["modeldocs_root"], inputs["collector_package"])

    report = write_feed(
        rec.tables[3],
        state=state,
        export=export,
        site_url=SITE + "/",
        changes_dir=rec.log,
        modeldocs_root=sources.root,
        collector_package=sources.package,
    )
    data = (export / FEED_FILE).read_bytes()
    check(data)
    assert report["through"] == DAYS[2]  # a full export: the last logged day
    assert len(report["states"]) >= 10  # events from every state the days touched
    root = ET.fromstring(data)
    ns = {"a": ATOM}
    entries = root.findall("a:entry", ns)
    assert len(entries) == report["entries"] > 0
    kinds = {e.find("a:category", ns).get("term") for e in entries}  # type: ignore[union-attr]
    assert {"relative_gained", "states_group"} <= kinds
    if state == "UT":
        assert "new_bill" in kinds  # Utah's bills arrive on day 2 with relatives
    for e in entries:
        text = "".join(e.itertext())
        assert not ATTRIBUTION.search(text), text
        link = e.find("a:link", ns)
        if link is not None and link.get("href", "").startswith(SITE):
            assert f"/bills/{state.lower()}-" in link.get("href", "")  # only hosted bills' pages
    georgia = [e for e in entries if "GA SB 9001" in (e.findtext("a:title", "", ns))]
    assert georgia  # it gains Utah SB 18 on day 2 and the deposits acts' bills on day 3
    assert root.findtext("a:updated", "", ns) == f"{DAYS[2]}T00:00:00Z"

    again = tmp_path / "again"
    shutil.copytree(export, again)
    write_feed(
        rec.tables[3],
        state=state,
        export=again,
        site_url=SITE,
        changes_dir=rec.log,
        modeldocs_root=sources.root,
        collector_package=sources.package,
    )
    assert (again / FEED_FILE).read_bytes() == data  # the same days give the same bytes


def test_a_feed_that_breaks_the_rfc_or_has_no_site_is_not_written(
    rec: Recorded, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = export_inputs(tmp_path / "inputs")
    export = full_export(rec.tables[3], tmp_path / "export" / "GA", inputs)
    sources = Sources(inputs["modeldocs_root"], inputs["collector_package"])
    with pytest.raises(FeedError, match="absolute"):
        write_feed(
            rec.tables[3], state="GA", export=export, site_url="example.org", changes_dir=rec.log,
            modeldocs_root=sources.root, collector_package=sources.package,
        )  # fmt: skip

    def broken(*_: object, **__: object) -> bytes:
        return _feed("", "")

    monkeypatch.setattr(feed, "render", broken)
    with pytest.raises(FeedError, match="RFC 4287"):
        write_feed(
            rec.tables[3],
            state="GA",
            export=export,
            site_url=SITE,
            changes_dir=rec.log,
            modeldocs_root=sources.root,
            collector_package=sources.package,
        )
    assert not (export / FEED_FILE).exists()

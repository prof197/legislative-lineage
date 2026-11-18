"""The origin story's rule (DESIGN.md#export): among the hosted state's passed bills, the bill
whose best listed origin found by both stages has the largest share of the bill's words; ties go
to the larger share of the model's words, then the lower bill slug. No organization is named. A
match is featured only when a labeler judged it correct (ruled at the step 48 review): the
first candidate in the rule's order judged correct is shown, and none when none was."""

import random

from lineage.export_schema import BillPage, BillRef, ModelDocRef, Origin, Stage, Status
from lineage.origin_story import Chosen, HostedBill, ListedOrigin, choose, hosted_bill

BOTH: tuple[Stage, ...] = ("near_duplicate", "semantic")


def origin(doc: str, bill: float, model: float, stages: tuple[Stage, ...] = BOTH) -> ListedOrigin:
    return ListedOrigin(doc_id=doc, bill_share=bill, model_share=model, stages=frozenset(stages))


def bill(slug: str, *origins: ListedOrigin, status: Status | None = "passed") -> HostedBill:
    return HostedBill(slug=slug, status=status, origins=origins)


class _Every:
    """Every match judged correct: the rule's order alone decides."""

    def __contains__(self, key: object) -> bool:
        return True


EVERY = _Every()


def test_the_largest_share_of_the_bill_wins() -> None:
    a = bill("ga-a", origin("lib/x", 0.30, 0.9))
    b = bill("ga-b", origin("lib/y", 0.62, 0.1))
    c = bill("ga-c", origin("lib/z", 0.45, 1.0))
    assert choose([a, b, c], EVERY) == Chosen(
        bill=b, origin=b.origins[0], eligible=3, passed_over=0
    )


def test_only_passed_bills_are_eligible() -> None:
    passed = bill("ga-p", origin("lib/x", 0.30, 0.5))
    others = [
        bill(f"ga-{s}", origin("lib/y", 0.95, 0.95), status=s)
        for s in ("introduced", "died", "passed_legislature", None)
    ]
    chosen = choose([*others, passed], EVERY)
    assert chosen is not None
    assert chosen.bill == passed
    assert chosen.eligible == 1


def test_an_origin_found_by_one_stage_is_not_eligible() -> None:
    near_only = bill("ga-n", origin("lib/x", 0.90, 0.9, ("near_duplicate",)))
    meaning_only = bill("ga-m", origin("lib/y", 0.80, 0.8, ("semantic",)))
    documented = bill("ga-d", origin("lib/z", 0.30, 0.3))
    chosen = choose([near_only, meaning_only, documented], EVERY)
    assert chosen is not None
    assert chosen.bill == documented
    assert chosen.eligible == 1


def test_a_bill_is_read_by_its_best_origin_found_by_both_stages() -> None:
    # Its larger origin was found by one stage only, so the one both found is its best.
    mixed = bill("ga-a", origin("lib/one", 0.90, 0.9, ("semantic",)), origin("lib/both", 0.40, 0.2))
    other = bill("ga-b", origin("lib/x", 0.35, 0.9))
    chosen = choose([other, mixed], EVERY)
    assert chosen is not None
    assert (chosen.bill.slug, chosen.origin.doc_id) == ("ga-a", "lib/both")


def test_stages_in_any_order_or_repeated_count_as_both() -> None:
    o = ListedOrigin(
        doc_id="lib/x",
        bill_share=0.5,
        model_share=0.5,
        stages=frozenset(["semantic", "near_duplicate", "semantic"]),
    )
    assert choose([bill("ga-a", o)], EVERY) == Chosen(
        bill=bill("ga-a", o), origin=o, eligible=1, passed_over=0
    )


def test_a_tie_on_the_bill_share_goes_to_the_larger_model_share() -> None:
    a = bill("ga-a", origin("lib/x", 0.50, 0.40))
    b = bill("ga-b", origin("lib/y", 0.50, 0.75))
    chosen = choose([a, b], EVERY)
    assert chosen is not None
    assert chosen.bill == b


def test_a_tie_on_both_shares_goes_to_the_lower_slug_whatever_the_order() -> None:
    bills = [bill(f"ga-2025_26-hb-{n}", origin(f"lib/{n}", 0.5, 0.5)) for n in (30, 4, 112, 7)]
    for seed in range(5):
        shuffled = bills[:]
        random.Random(seed).shuffle(shuffled)
        chosen = choose(shuffled, EVERY)
        assert chosen is not None
        # Slugs compare as strings, as the export writes them: "hb-112" sorts before "hb-30".
        assert chosen.bill.slug == "ga-2025_26-hb-112"


def test_within_a_bill_ties_go_to_the_model_share_then_the_document_id() -> None:
    b = bill(
        "ga-a",
        origin("lib/c", 0.5, 0.5),
        origin("lib/b", 0.5, 0.9),
        origin("lib/a", 0.5, 0.9),
        origin("lib/d", 0.2, 1.0),
    )
    chosen = choose([b], EVERY)
    assert chosen is not None
    assert chosen.origin.doc_id == "lib/a"


def test_no_eligible_bill_gives_none() -> None:
    assert choose([], EVERY) is None
    assert choose([bill("ga-a")], EVERY) is None
    assert choose([bill("ga-a", origin("lib/x", 0.9, 0.9, ("near_duplicate",)))], EVERY) is None
    assert choose([bill("ga-a", origin("lib/x", 0.9, 0.9), status="introduced")], EVERY) is None


def test_only_a_match_a_labeler_judged_correct_is_featured() -> None:
    """The rule's first pick was never judged correct, so the next in its order that was is
    shown, and the bills passed over are counted."""
    unjudged = bill("ga-a", origin("lib/x", 0.90, 0.3))
    wrong = bill("ga-b", origin("lib/y", 0.80, 0.4))
    right = bill("ga-c", origin("lib/z", 0.40, 0.5))
    chosen = choose([unjudged, wrong, right], {("ga-c", "lib/z")})
    assert chosen == Chosen(bill=right, origin=right.origins[0], eligible=3, passed_over=2)


def test_a_bills_next_origin_judged_correct_can_be_featured() -> None:
    b = bill("ga-a", origin("lib/unread", 0.90, 0.3), origin("lib/read", 0.60, 0.5))
    other = bill("ga-b", origin("lib/z", 0.50, 0.9))
    chosen = choose([b, other], {("ga-a", "lib/read"), ("ga-b", "lib/z")})
    assert chosen is not None
    assert (chosen.bill.slug, chosen.origin.doc_id, chosen.passed_over) == ("ga-a", "lib/read", 0)


def test_no_story_when_no_candidate_was_judged_correct() -> None:
    bills = [bill("ga-a", origin("lib/x", 0.9, 0.9)), bill("ga-b", origin("lib/y", 0.5, 0.5))]
    assert choose(bills, set[tuple[str, str]]()) is None
    # A judgment on an origin the rule cannot feature changes nothing.
    one_stage = bill("ga-c", origin("lib/z", 0.9, 0.9, ("near_duplicate",)))
    assert choose([*bills, one_stage], {("ga-c", "lib/z")}) is None


def _ref(slug: str, status: Status | None) -> BillRef:
    return BillRef(
        slug=slug,
        state="GA",
        session="2025_26",
        identifier="HB 1",
        title="A bill",
        status=status,
        status_reason=None,
        status_date=None,
        first_action=None,
    )


def test_a_bill_page_reads_as_a_hosted_bill() -> None:
    doc = ModelDocRef(
        id="lib/x",
        source="lib",
        organization="Org",
        title="Model",
        url="https://example.org/x",
        published=None,
        leaked=False,
    )
    page = BillPage(
        bill=_ref("ga-2025_26-hb-1", "passed"),
        text=None,
        sections=[],
        words_searched=100,
        families=0,
        relatives=[],
        origins=[Origin(doc=doc, stages=list(BOTH), bill_share=0.4, model_share=0.6, sections=[1])],
    )
    assert hosted_bill(page) == HostedBill(
        slug="ga-2025_26-hb-1",
        status="passed",
        origins=(origin("lib/x", 0.4, 0.6),),
    )

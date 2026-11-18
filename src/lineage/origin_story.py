"""The origin story's rule: which hosted bill the site's origin story shows (DESIGN.md#export).

One story shows a documented origin, chosen by a rule that names no organization: among the
hosted state's passed bills, the bill whose best listed origin found by both stages has the
largest share of the bill's words found in the model. Ties go to the larger share of the
model's words, then the lower bill slug. Whatever organization that origin's document belongs
to is the one the story names, so the choice cannot favour or avoid any library.

A featured story features only a match a labeler has judged correct (ruled at the step 48
review): the rule's candidates, each passed bill's origins found by both stages, are read in
the rule's order, and the first a labeler judged correct is shown; when none was, there is no
story. The first pick the rule made for Georgia was a compact matched to an unrelated compact
on the articles every compact shares, which no labeler had read.

A pure function over the export's listed origins: what a page lists, after the floor, so the
story never shows an origin a bill page does not. "Found by both stages" means both the
near-duplicate and the semantic stage found the match, the evidence least likely to be wrong.
"""

from collections.abc import Container, Iterable, Sequence
from dataclasses import dataclass

from lineage.export_schema import BillPage, OriginStory, Stage, Status

BOTH: frozenset[Stage] = frozenset({"near_duplicate", "semantic"})


@dataclass(frozen=True, slots=True)
class ListedOrigin:
    """An origin as a bill page lists it, with what the rule reads."""

    doc_id: str
    bill_share: float  # of the bill's words, the share found in the model
    model_share: float  # of the model's words, the share found in the bill
    stages: frozenset[Stage]


@dataclass(frozen=True, slots=True)
class HostedBill:
    slug: str
    status: Status | None
    origins: Sequence[ListedOrigin]


@dataclass(frozen=True, slots=True)
class Chosen:
    bill: HostedBill
    origin: ListedOrigin  # its first origin in the rule's order that a labeler judged correct
    eligible: int  # passed bills with a listed origin found by both stages
    passed_over: int  # of them, bills the rule ranks first whose origins none judged correct


def choose(bills: Iterable[HostedBill], judged: Container[tuple[str, str]]) -> Chosen | None:
    """The story's bill and origin: of the passed bills' origins both stages found, in the
    rule's order (the largest share of the bill, then of the model, then the lower bill slug and
    document id, so the export's order never decides), the first whose (bill slug, document id)
    is in `judged`, the matches a labeler judged correct. None when none is."""
    candidates = sorted(
        ((b, o) for b in bills if b.status == "passed" for o in b.origins if o.stages >= BOTH),
        key=lambda c: (-c[1].bill_share, -c[1].model_share, c[0].slug, c[1].doc_id),
    )
    eligible = len({b.slug for b, _ in candidates})
    ahead: set[str] = set()
    for b, o in candidates:
        if (b.slug, o.doc_id) in judged:
            return Chosen(bill=b, origin=o, eligible=eligible, passed_over=len(ahead - {b.slug}))
        ahead.add(b.slug)
    return None


def hosted_bill(page: BillPage) -> HostedBill:
    """A bill page as the rule reads it."""
    return HostedBill(
        slug=page.bill.slug,
        status=page.bill.status,
        origins=tuple(
            ListedOrigin(
                doc_id=o.doc.id,
                bill_share=o.bill_share,
                model_share=o.model_share,
                stages=frozenset(o.stages),
            )
            for o in page.origins
        ),
    )


FILE = "origin_story.json"
STORY = "origin"  # its key in rules/stories.toml, which places it on a state's site


def story_file(pages: Iterable[BillPage], judged: Container[tuple[str, str]]) -> OriginStory | None:
    """The origin story's file from the hosted state's bill pages and the matches a labeler
    judged correct, by (bill slug, document id), or None when the rule chooses no bill: what
    the export writes, and the live track's incremental export too."""
    by_slug = {p.bill.slug: p for p in pages}
    chosen = choose((hosted_bill(p) for p in by_slug.values()), judged)
    if chosen is None:
        return None
    page = by_slug[chosen.bill.slug]
    origin = next(o for o in page.origins if o.doc.id == chosen.origin.doc_id)
    return OriginStory(
        bill=page.bill, origin=origin, eligible=chosen.eligible, passed_over=chosen.passed_over
    )

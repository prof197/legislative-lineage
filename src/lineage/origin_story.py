"""The origin story's rule: which hosted bill the site's origin story shows (DESIGN.md#export).

One story shows a documented origin, chosen by a rule that names no organization: among the
hosted state's passed bills, the bill whose best listed origin found by both stages has the
largest share of the bill's words found in the model. Ties go to the larger share of the
model's words, then the lower bill slug. Whatever organization that origin's document belongs
to is the one the story names, so the choice cannot favour or avoid any library.

A pure function over the export's listed origins: what a page lists, after the floor, so the
story never shows an origin a bill page does not. "Found by both stages" means both the
near-duplicate and the semantic stage found the match, the evidence least likely to be wrong.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from lineage.export_schema import BillPage, Stage, Status

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
    origin: ListedOrigin  # the bill's best listed origin found by both stages
    eligible: int  # passed bills with a listed origin found by both stages


def _documented(origins: Iterable[ListedOrigin]) -> ListedOrigin | None:
    """A bill's best origin found by both stages: the largest share of the bill, then of the
    model, then the lower document id, so the export's order never decides."""
    found = [o for o in origins if o.stages >= BOTH]
    return min(found, key=lambda o: (-o.bill_share, -o.model_share, o.doc_id), default=None)


def choose(bills: Iterable[HostedBill]) -> Chosen | None:
    """The story's bill and origin, or None when no passed bill has an origin both stages found."""
    candidates = [
        (b, o) for b in bills if b.status == "passed" and (o := _documented(b.origins)) is not None
    ]
    if not candidates:
        return None
    b, o = min(candidates, key=lambda c: (-c[1].bill_share, -c[1].model_share, c[0].slug))
    return Chosen(bill=b, origin=o, eligible=len(candidates))


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

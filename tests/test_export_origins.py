"""A bill page's origins (DESIGN.md#origin-matches): each names its document's library and any
organization the document was published with, so a joint bill credits both."""

from datetime import date

import pytest

from lineage.export.pages import ExportError, origins_by_bill
from lineage.origins.matches import FLOOR, BillSpan, OriginMatch
from lineage.origins.modeldocs import ModelDoc, Source

SOURCE = Source(
    key="example-library",
    organization="Example Council",
    descriptor="nonprofit research organization, 501(c)(3)",
    library_url="https://example.org/models",
    terms_note="viewing and copying with attribution",
    terms_read_on=date(2026, 11, 15),
)
DOC = ModelDoc(
    id="example-library/joint-act",
    source=SOURCE.key,
    title="Model Joint Act",
    url="https://example.org/models/joint-act",
    collected_on=date(2026, 11, 15),
    robots="allowed",
    leaked=False,
    raw_sha256="0" * 64,
    media_type="application/pdf",
    text_sha256="1" * 64,
    co_publishers=("Second Institute",),
)
MATCH = OriginMatch(
    bill_id="ocd-bill/1",
    doc_id=DOC.id,
    stages=["near_duplicate"],
    model_share=0.5,
    bill_share=0.25,
    spans=[BillSpan(ordinal=2, start=0, end=40, kind="exact")],
)


def test_a_joint_documents_page_line_names_both_publishers() -> None:
    (origin,) = origins_by_bill([MATCH], {DOC.id: (DOC, SOURCE)}, {"ocd-bill/1"})["ocd-bill/1"]

    assert origin.doc.organization == "Example Council"
    assert origin.doc.co_publishers == ["Second Institute"]
    assert origin.sections == [2]


def test_a_match_no_source_holds_stops_the_export() -> None:
    with pytest.raises(ExportError, match="no collected source"):
        origins_by_bill([MATCH], {}, {"ocd-bill/1"})


def test_a_page_lists_an_origin_only_at_the_floor() -> None:
    """The floor reads the larger share: a bill taking a quarter of a long model is listed, one
    whose shares both fall short is not."""
    at = MATCH.model_copy(update={"model_share": FLOOR, "bill_share": 0.01})
    below = MATCH.model_copy(
        update={"bill_id": "ocd-bill/2", "model_share": 0.2, "bill_share": 0.2}
    )
    listed = origins_by_bill([at, below], {DOC.id: (DOC, SOURCE)}, {"ocd-bill/1", "ocd-bill/2"})

    assert set(listed) == {"ocd-bill/1"}

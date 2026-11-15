"""A collector that finds one document and reports it blocked, without touching the network."""

from datetime import date

from lineage.collect import CollectContext, CollectReport
from lineage.modeldocs import Source

SOURCE = Source(
    key="example-library",
    organization="Example Council",
    descriptor="nonprofit membership organization, 501(c)(3)",
    members="Member organizations in several states (synthetic fixture)",
    members_source="https://example.org/about",
    library_url="https://example.org/models",
    terms_note="viewing and copying with attribution",
    terms_read_on=date(2026, 11, 15),
)


def collect(context: CollectContext) -> CollectReport:
    return CollectReport(
        source=SOURCE.key,
        run_on=context.on,
        found=1,
        collected=0,
        unchanged=0,
        blocked=["https://example.org/models/private.pdf"],
        failed=[],
    )

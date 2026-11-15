"""A leaked archive for the registry's and the fixture export's tests: no network, no storing."""

from datetime import date

from lineage.collect import CollectContext, CollectReport
from lineage.modeldocs import Source

SOURCE = Source(
    key="example-archive",
    organization="Example Archive Project",
    descriptor="nonprofit research organization, 501(c)(3)",
    library_url="https://example.net/archive",
    terms_note="wiki text under CC BY-SA; the documents are the drafting organization's",
    terms_read_on=date(2026, 11, 15),
    leaked=True,
    leak_note="published by the archive after a leak",
)


def collect(context: CollectContext) -> CollectReport:
    return CollectReport(
        source=SOURCE.key,
        run_on=context.on,
        found=0,
        collected=0,
        unchanged=0,
        blocked=[],
        failed=[],
    )

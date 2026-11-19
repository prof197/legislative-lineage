"""The Institute for Family Studies' model state legislation.

What is collected: the model bills IFS publishes on its own, listed in `DOCUMENTS` as read on
2026-11-15: the App Store Accountability Act, linked from its Family First Technology
Initiative's Stop Digital Harm page. The 2023 social media bill IFS wrote with the Ethics and
Public Policy Center stands among IFS's briefs too, but its file is linked from EPPC's page, so
it is collected once, by `lineage.origins.collectors.eppc`, which names IFS as its co-publisher.
Briefs, reports and fact sheets are not model legislation and are left out.

A document is read as EPPC's are (`lineage.origins.collectors.eppc`): its page's link that names the
model bill, never the fact sheet, on IFS's own host, and the same cleaning, since the two
organizations' bills share a layout.

Fetching is on (`SITE_FETCH`): read on 2026-11-15, IFS publishes no terms of use and nothing
asks a visitor to accept terms, and its robots.txt lets this project read the documents' paths.
See DESIGN.md#model-documents.
"""

from collections.abc import Sequence
from datetime import date

from lineage.origins.collect import CollectContext, CollectReport
from lineage.origins.collect_base import Fetch, ReadRobots, fetch_url, read_robots
from lineage.origins.collectors.eppc import Listed, collect_documents
from lineage.origins.modeldocs import Source

IFS = "https://ifstudies.org"
LIBRARY = f"{IFS}/family-first-technology-initiative/stop-digital-harm"
# On, from the terms as read (see the module docstring); tests pass `site=` either way.
SITE_FETCH = True
HOSTS = frozenset({"ifstudies.org", "www.ifstudies.org"})

# The descriptor's sources, read 2026-11-15:
# - IRS: ProPublica's Nonprofit Explorer, The Institute For Family Studies, EIN 27-0950140,
#   501(c)(3), exempt since 2010, latest Form 990 for the year ending December 2025:
#   https://projects.propublica.org/nonprofits/organizations/270950140
# - kind: https://ifstudies.org/about-us (research and public engagement; no membership).
# Terms, read 2026-11-15: ifstudies.org's footer links https://ifstudies.org/privacy-policy and
# https://ifstudies.org/ai-submission-policy, neither about using its content, and no terms of
# use. robots.txt: https://ifstudies.org/robots.txt disallows one report directory to all
# crawlers and allows the rest.
SOURCE = Source(
    key="ifs",
    organization="Institute for Family Studies",
    # Type: ifstudies.org/about-us. IRS: EIN 27-0950140 in ProPublica's Nonprofit Explorer.
    descriptor="nonprofit corporation exempt under IRS section 501(c)(3); a research and "
    "public engagement organization in Charlottesville, Virginia; no members",
    library_url=LIBRARY,
    terms_url=None,
    terms_note="IFS publishes no terms of use: its site links only a privacy policy and a "
    "policy for submissions, neither about using its content. Pages carry a copyright notice "
    "and nothing asks a visitor to accept terms. Its robots.txt bars one report directory to "
    "every crawler and lets any crawler read the rest. The texts are read for matching, kept "
    "local and never redistributed.",
    terms_read_on=date(2026, 11, 15),
)

# IFS's own model state legislation, read 2026-11-15. The page and the PDF give no date.
DOCUMENTS: tuple[Listed, ...] = (
    Listed(
        slug="app-store-accountability-act",
        title="App Store Accountability Act",
        page=LIBRARY,
        published=None,
    ),
)


def collect_from(
    context: CollectContext,
    *,
    documents: Sequence[Listed] = DOCUMENTS,
    fetch: Fetch = fetch_url,
    robots_reader: ReadRobots = read_robots,
    site: bool = SITE_FETCH,
) -> CollectReport:
    return collect_documents(
        SOURCE,
        context,
        documents,
        hosts=HOSTS,
        fetch=fetch,
        robots_reader=robots_reader,
        site=site,
    )


def collect(context: CollectContext) -> CollectReport:
    # Module names are looked up per call, so a test can stand in for the network.
    return collect_from(
        context,
        documents=DOCUMENTS,
        fetch=fetch_url,
        robots_reader=read_robots,
        site=SITE_FETCH,
    )

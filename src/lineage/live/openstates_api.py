"""Open States API v3: the bills search the daily intake sends, and its bills in the bulk shape.

The intake asks `GET /bills` for one state's bills updated since a cursor, oldest update first,
twenty to a page, with the actions, sponsorships and versions included. Each `Bill` it returns
is mapped to the record Open States' session archives hold, which `lineage.data.ingest` and
`lineage.live.updates` read, so a day of the API's bills folds and rebuilds like an archive.
Parameter names, the include options, the key's header, the page size and the paging fields are
those of the API's OpenAPI document (https://v3.openstates.org/docs). See DESIGN.md#updates.
"""

import urllib.parse
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

ROOT = "https://v3.openstates.org/"
KEY_ENV = "OPENSTATES_API_KEY"
KEY_HEADER = "X-API-KEY"
# The API's largest page of bills (its BillPagination.max_per_page); a larger one answers 400.
PER_PAGE = 20
# Oldest update first, so the last bill on a page is a cursor the next request can start from.
SORT = "updated_asc"
INCLUDE = ("sponsorships", "abstracts", "other_titles", "actions", "sources", "documents",
           "versions")  # fmt: skip
STATES = (
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN",
    "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH",
    "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA",
    "VT", "WA", "WI", "WV", "WY",
)  # fmt: skip


def jurisdiction(state: str) -> str:
    return f"ocd-jurisdiction/country:us/state:{state.lower()}/government"


def bills_url(state: str, since: datetime, page: int, *, root: str = ROOT) -> str:
    """One page of a state's bills updated at or after `since`. The key is never in the URL:
    it goes in the `X-API-KEY` header, so a logged URL cannot leak it."""
    params: list[tuple[str, str]] = [
        ("jurisdiction", jurisdiction(state)),
        ("updated_since", since.isoformat()),
        ("sort", SORT),
        ("per_page", str(PER_PAGE)),
        ("page", str(page)),
        *(("include", name) for name in INCLUDE),
    ]
    # urlencode escapes the offset's "+", which a query string would read as a space.
    return f"{root}bills?{urllib.parse.urlencode(params)}"


# The response, as far as the bulk record needs it. Fields the archives do not carry (ids of
# actions and versions, openstates_url, extras) are ignored.


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class Organization(_Model):
    name: str
    classification: str | None = None


class Jurisdiction(_Model):
    id: str
    name: str


class DocumentLink(_Model):
    url: str
    media_type: str | None = None


class Document(_Model):
    note: str = ""
    date: str = ""
    links: list[DocumentLink] = []


class Action(_Model):
    organization: Organization
    description: str
    date: str
    classification: list[str] = []
    order: int


class Sponsorship(_Model):
    name: str
    primary: bool
    classification: str


class Abstract(_Model):
    abstract: str
    note: str = ""


class Title(_Model):
    title: str
    note: str = ""


class Source(_Model):
    url: str


class Bill(_Model):
    id: str
    session: str
    jurisdiction: Jurisdiction
    from_organization: Organization
    identifier: str
    title: str
    classification: list[str] = []
    subject: list[str] = []
    updated_at: datetime
    abstracts: list[Abstract] = []
    other_titles: list[Title] = []
    sponsorships: list[Sponsorship] = []
    actions: list[Action] = []
    sources: list[Source] = []
    documents: list[Document] = []
    versions: list[Document] = []


class Pagination(_Model):
    per_page: int
    page: int
    max_page: int
    total_items: int


class BillList(_Model):
    results: list[Bill]
    pagination: Pagination


def _documents(docs: Sequence[Document]) -> list[dict[str, Any]]:
    return [
        {
            "note": d.note,
            "date": d.date,
            "links": [{"url": lk.url, "media_type": lk.media_type} for lk in d.links],
        }
        for d in docs
    ]


def bulk_record(bill: Bill) -> dict[str, Any]:
    """`bill` as a session archive's record, its keys in the archives' order, without text:
    the API carries none, so the intake fetches it and sets `raw_text` and `raw_text_url`.

    The chamber is the originating organization's classification, as in the archives ("upper",
    "lower", Nebraska's "legislature"). Votes are not asked for and related bills name no
    bill id in v3, so both stay empty; nothing downstream reads them."""
    return {
        "id": bill.id,
        "legislative_session": bill.session,
        "jurisdiction_name": bill.jurisdiction.name,
        "identifier": bill.identifier,
        "title": bill.title,
        "chamber": bill.from_organization.classification,
        "classification": list(bill.classification),
        "subject": list(bill.subject),
        "abstracts": [{"abstract": a.abstract, "note": a.note} for a in bill.abstracts],
        "other_titles": [{"title": t.title, "note": t.note} for t in bill.other_titles],
        "actions": [
            {
                "organization__name": a.organization.name,
                "description": a.description,
                "date": a.date,
                "classification": list(a.classification),
                "order": a.order,
            }
            for a in bill.actions
        ],
        "related_bills": [],
        "sponsors": [
            {"name": s.name, "primary": s.primary, "classification": s.classification}
            for s in bill.sponsorships
        ],
        "documents": _documents(bill.documents),
        "versions": _documents(bill.versions),
        "sources": [{"url": s.url} for s in bill.sources],
        "votes": [],
        "raw_text": "",
        "raw_text_url": None,
    }

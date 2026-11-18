"""Atom (RFC 4287): the namespace, and each way a document breaks the rules for a feed's and an
entry's elements. lineage.feed writes the live track's feed and checks it here; the data
release's packing checks the feed it ships the same way (lineage.release).
"""

import re
import xml.etree.ElementTree as ET
from datetime import datetime

ATOM = "http://www.w3.org/2005/Atom"


# RFC 4287's required elements and cardinalities.

_RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
_IRI = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:\S+$")
# Elements a feed or an entry may hold at most once (RFC 4287 4.1.1, 4.1.2).
_FEED_ONCE = ("generator", "icon", "logo", "rights", "subtitle")
_ENTRY_ONCE = ("content", "published", "rights", "source", "summary")


def _q(tag: str) -> str:
    return f"{{{ATOM}}}{tag}"


def _check_common(node: ET.Element, where: str, problems: list[str]) -> None:
    for tag in ("id", "title", "updated"):
        found = node.findall(_q(tag))
        if len(found) != 1:
            problems.append(f"{where}: {len(found)} atom:{tag}, not exactly one")
    for found in node.findall(_q("id")):
        if not _IRI.match((found.text or "").strip()):
            problems.append(f"{where}: atom:id {found.text!r} is not an absolute IRI")
    for found in node.findall(_q("updated")):
        text = (found.text or "").strip()
        try:
            ok = bool(_RFC3339.match(text))
            datetime.fromisoformat(text)  # a valid month, day and time, too
        except ValueError:
            ok = False
        if not ok:
            problems.append(f"{where}: atom:updated {text!r} is not an RFC 3339 date-time")
    alternates: set[tuple[str, str]] = set()
    for link in node.findall(_q("link")):
        if not link.get("href"):
            problems.append(f"{where}: an atom:link has no href")
        if link.get("rel", "alternate") == "alternate":
            key = (link.get("type", ""), link.get("hreflang", ""))
            if key in alternates:
                problems.append(f"{where}: two alternate links of one type and hreflang")
            alternates.add(key)
    for author in node.findall(_q("author")):
        if len(author.findall(_q("name"))) != 1:
            problems.append(f"{where}: an atom:author without exactly one atom:name")


def violations(data: bytes) -> list[str]:
    """Each way the document breaks RFC 4287's rules for a feed's and an entry's elements."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        return [f"not well-formed XML: {e}"]
    if root.tag != _q("feed"):
        return [f"the root is {root.tag}, not atom:feed"]
    problems: list[str] = []
    _check_common(root, "feed", problems)
    for tag in _FEED_ONCE:
        if len(root.findall(_q(tag))) > 1:
            problems.append(f"feed: more than one atom:{tag}")
    feed_author = bool(root.findall(_q("author")))
    ids: set[str] = set()
    for n, node in enumerate(root.findall(_q("entry")), start=1):
        where = f"entry {n}"
        _check_common(node, where, problems)
        for tag in _ENTRY_ONCE:
            if len(node.findall(_q(tag))) > 1:
                problems.append(f"{where}: more than one atom:{tag}")
        if not feed_author and not node.findall(_q("author")):
            problems.append(f"{where}: no atom:author, and the feed has none")
        has_content = bool(node.findall(_q("content")))
        alternate = [
            ln for ln in node.findall(_q("link")) if ln.get("rel", "alternate") == "alternate"
        ]
        if not has_content and not alternate:
            problems.append(f"{where}: neither atom:content nor an alternate atom:link")
        for content in node.findall(_q("content")):
            if content.get("src") and not node.findall(_q("summary")):
                problems.append(f"{where}: atom:content has src but the entry has no summary")
        for found in node.findall(_q("id")):
            text = (found.text or "").strip()
            if text in ids:
                problems.append(f"{where}: atom:id {text} is another entry's")
            ids.add(text)
    return problems

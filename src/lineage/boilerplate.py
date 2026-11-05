"""Flag sections that are nothing but boilerplate: enacting clauses, effective dates,
severability and repealers.

These clauses are near-identical across thousands of bills and would swamp real matches. A
short pattern list catches them; the document-frequency stoplist catches the long tail. A
section is flagged only when almost nothing is left once its boilerplate is removed, so a
provision that ends with an effective date stays matchable. Patterns run on normalized text
(lowercase words, no punctuation). See DESIGN.md#boilerplate.
"""

import re

# Bounded word runs, `(?: \w+){0,n}`, keep every pattern linear and stop a match from
# swallowing a long stretch of substance.
PATTERNS: dict[str, re.Pattern[str]] = {
    "enacting clause": re.compile(
        r"\b(?:be it enacted|(?:the )?people of the state of \w+(?: \w+)? do enact)"
        r"(?: \w+){0,30}?(?: as follows|$)"
    ),
    "effective date": re.compile(
        r"\b(?:this|the|said) (?:act|section|law|bill|chapter|article)s?(?: \w+){0,6}? "
        r"(?:shall |will |does )?(?:(?:take|takes) effect|(?:become|becomes|be|is) "
        r"(?:effective|operative|in full force(?: and effect)?|in force))(?: \w+){0,30}"
    ),
    "severability": re.compile(
        r"\bif any (?:provision|section|part|clause|sentence|paragraph|subsection|portion|item)s?"
        r"(?: \w+){0,90}? (?:invalid|unconstitutional|unenforceable|void)(?: \w+){0,60}"
        r"|\b(?:the )?provisions of this (?:act|section|chapter|article) are (?:hereby )?"
        r"(?:declared )?(?:to be )?severable\b"
    ),
    "repealer": re.compile(
        r"\ball (?:laws|acts) (?:and|or) parts of (?:laws|acts) in conflict "
        r"(?:herewith|with this (?:act|section|chapter))(?: are| is)?(?: hereby)? repealed\b"
    ),
}
# The section's own heading ("section 4", "sec 12") is not content.
_HEADING = re.compile(r"^(?:new section )?(?:section|sec) \w+")
# Room for a catchline such as "effective date" or "severability clause".
_MAX_RESIDUAL_WORDS = 4


def boilerplate_kind(normalized: str) -> str | None:
    """The kind of boilerplate a section consists of, or None if it has real content."""
    removed: dict[str, int] = {}
    residual = _HEADING.sub("", normalized, count=1)
    for kind, pattern in PATTERNS.items():
        before = len(residual.split())
        residual = pattern.sub(" ", residual)
        removed[kind] = before - len(residual.split())
    if not any(removed.values()) or len(residual.split()) > _MAX_RESIDUAL_WORDS:
        return None
    return max(removed, key=lambda k: removed[k])

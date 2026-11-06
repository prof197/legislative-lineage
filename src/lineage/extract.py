"""Bill text from a legislature's HTML or PDF, as plain lines for the splitter.

Extraction keeps lines and drops presentation. It does not clean: lineage.sections.clean strips
line-number gutters, running headers and invisible characters from fetched and exported text
alike, so both are split the same way. Struck text is dropped. In a bill it is current law the
bill deletes, which is not the bill's own language, and matching it would link bills that
merely amend the same statute. See DESIGN.md#dr-6.
"""

import io
import logging
import re
from html.parser import HTMLParser

from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError, PyPdfError

_SKIP = frozenset({"script", "style", "head", "title", "noscript", "template"})
_STRUCK = frozenset({"s", "strike", "del"})
_BLOCK = frozenset(
    {
        "p", "div", "br", "tr", "li", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6",
        "pre", "blockquote", "section", "article", "dd", "dt", "dl", "hr", "center", "body",
    }
)  # fmt: skip
_VOID = frozenset({"br", "hr", "img", "meta", "link", "input", "col", "area", "base", "wbr"})
_LINE_THROUGH = re.compile(r"text-decoration[^;]*line-through", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_SPACES = re.compile(r"[ \t\f\v\u00a0]+")


class ExtractError(Exception):
    pass


class _TextParser(HTMLParser):
    """Collects text, a newline at each block boundary; skips non-content and struck elements."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden: list[str] = []  # open tags whose content is dropped, innermost last
        self._pre = 0  # open <pre> elements, inside which line breaks are the text's own

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK:
            self.parts.append("\n")
        if tag in _VOID:
            return
        if tag == "pre":
            self._pre += 1
        style = next((v for k, v in attrs if k == "style" and v), "")
        if self._hidden or tag in _SKIP or tag in _STRUCK or _LINE_THROUGH.search(style):
            self._hidden.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre" and self._pre:
            self._pre -= 1
        if self._hidden and self._hidden[-1] == tag:
            self._hidden.pop()
        elif tag in self._hidden:
            # Unclosed children inside a hidden element: close back to its tag.
            del self._hidden[len(self._hidden) - 1 - self._hidden[::-1].index(tag) :]
        if tag in _BLOCK or tag in {"td", "th"}:
            self.parts.append("\n" if tag in _BLOCK else " ")

    def handle_data(self, data: str) -> None:
        if not self._hidden:
            self.parts.append(data if self._pre else _WHITESPACE.sub(" ", data))


def _decode(data: bytes, charset: str | None) -> str:
    for encoding in (charset, "utf-8"):
        if encoding:
            try:
                return data.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue
    # Word's HTML exports, common on legislature sites, are Windows-1252.
    return data.decode("cp1252", errors="replace")


def html_text(data: bytes, charset: str | None) -> str:
    parser = _TextParser()
    parser.feed(_decode(data, charset))
    parser.close()
    lines = (_SPACES.sub(" ", line).strip() for line in "".join(parser.parts).split("\n"))
    return "\n".join(line for line in lines if line)


def pdf_text(data: bytes, *, layout: bool = False) -> str:
    """The PDF's text layer, page by page. `layout` places words by their position on the page,
    which recovers lines from PDFs that position each word separately (Iowa, Idaho)."""
    # pypdf logs every malformed object it recovers from; the outcome is judged on the text.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        reader = PdfReader(io.BytesIO(data))
        mode = "layout" if layout else "plain"
        text = "\n".join(page.extract_text(extraction_mode=mode) or "" for page in reader.pages)
    except DependencyError as e:
        # AES-encrypted PDFs, even with an empty password, need the optional cryptography package.
        raise ExtractError(f"encrypted PDF: {e}") from e
    except (PdfReadError, PyPdfError, ValueError, KeyError, TypeError, IndexError) as e:
        raise ExtractError(f"unreadable PDF: {type(e).__name__}: {e}") from e
    if not text.strip():
        raise ExtractError("no text layer (a scanned PDF)")
    return text


def extract(data: bytes, content_type: str, *, layout: bool = False) -> str:
    """Text from a response body, read as its Content-Type says or, failing that, its bytes.

    A page that yields no text, such as one built by scripts in the browser, is an error.
    """
    media, _, params = content_type.partition(";")
    media = media.strip().lower()
    charset = next(
        (p.split("=", 1)[1].strip(" \"'") for p in params.split(";") if "charset=" in p.lower()),
        None,
    )
    if media == "application/pdf" or data.startswith(b"%PDF"):
        return pdf_text(data, layout=layout)
    if media in {"text/html", "application/xhtml+xml"} or data.lstrip()[:1] == b"<":
        text = html_text(data, charset)
        if not text:
            raise ExtractError("no text in the page (built by scripts?)")
        return text
    raise ExtractError(f"unsupported format {media or 'unknown'}")

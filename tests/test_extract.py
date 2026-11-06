"""Bill text from a legislature's HTML or PDF, before the splitter's own cleaning."""

import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import ContentStream, DictionaryObject, NameObject

from lineage.extract import ExtractError, extract, html_text, pdf_text

BILL_HTML = b"""<html><head><title>HB 12</title><style>p {margin:0}</style>
<script>var x = 1;</script></head><body>
<p>AN ACT relating to widgets.</p>
<p>Section&nbsp;1.  Short title.</p><p>This act may be cited as the Widget Act.</p>
<p>Section 2. A person shall <s>not</s> <u>register</u> a widget.<br>Each registration
expires.</p>
<p><span style="text-decoration: line-through">Repealed text.</span></p>
<table><tr><td>Fee</td><td>$5</td></tr></table>
</body></html>"""


def test_html_keeps_the_text_and_its_lines_and_drops_markup_and_struck_text() -> None:
    text = html_text(BILL_HTML, "utf-8")

    assert text.splitlines() == [
        "AN ACT relating to widgets.",
        "Section 1. Short title.",
        "This act may be cited as the Widget Act.",
        "Section 2. A person shall register a widget.",
        "Each registration expires.",
        "Fee $5",
    ]


def test_html_decodes_with_the_declared_charset_or_falls_back() -> None:
    page = "<p>Fiancée § 3</p>".encode("cp1252")

    assert html_text(page, "windows-1252") == "Fiancée § 3"
    assert html_text(page, None) == "Fiancée § 3"


def _pdf(*pages: str) -> bytes:
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(612, 792)
        lines = " ".join(
            f"BT /F1 12 Tf 72 {720 - 16 * i} Td ({line}) Tj ET"
            for i, line in enumerate(text.splitlines())
        )
        stream = ContentStream(None, writer)
        stream.set_data(lines.encode("latin-1"))
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        page.replace_contents(stream)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_pdf_text_is_read_page_by_page_in_order() -> None:
    text = pdf_text(_pdf("Section 1. Short title.\nThis act", "Section 2. Fees."))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert lines == ["Section 1. Short title.", "This act", "Section 2. Fees."]


def test_a_pdf_without_a_text_layer_is_an_error() -> None:
    with pytest.raises(ExtractError, match="no text layer"):
        pdf_text(_pdf(""))


def test_a_broken_pdf_is_an_error() -> None:
    with pytest.raises(ExtractError):
        pdf_text(b"%PDF-1.4 truncated")


def test_extract_picks_the_reader_from_the_content_type_then_the_bytes() -> None:
    assert extract(BILL_HTML, "text/html; charset=utf-8").startswith("AN ACT")
    assert extract(_pdf("Section 1. A."), "application/octet-stream").strip() == "Section 1. A."
    with pytest.raises(ExtractError, match="unsupported"):
        extract(b"PK\x03\x04", "application/msword")


def test_a_page_with_no_text_is_an_error() -> None:
    page = b"<html><body><script>render()</script></body></html>"

    with pytest.raises(ExtractError, match="no text"):
        extract(page, "text/html")


def test_a_pdf_can_be_read_by_word_position() -> None:
    text = pdf_text(_pdf("Section 1. Short title."), layout=True)

    assert "Section 1. Short title." in " ".join(text.split())

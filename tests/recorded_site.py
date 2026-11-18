"""Recorded responses for the commission collector's tests, from tests/fixtures/collect/: a fetch
and a robots.txt reader that never touch the network, and Word and PDF files built from text."""

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

from lineage.collect_base import CollectFetchError, Fetched
from lineage.core.robots import Robots, parse_robots

COLLECT = Path(__file__).parent / "fixtures" / "collect"
SYNTHETIC_ACT = COLLECT / "synthetic_act.txt"

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def docx_bytes(text: str) -> bytes:
    """A minimal Word document, one paragraph per line, tabs as w:tab."""
    paragraphs = "".join(
        "<w:p><w:r>"
        + "<w:tab/>".join(
            f'<w:t xml:space="preserve">{escape(part)}</w:t>' for part in line.split("\t")
        )
        + "</w:r></w:p>"
        for line in text.split("\n")
    )
    document = (
        f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{_W}">'
        f"<w:body>{paragraphs}</w:body></w:document>"
    )
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
        z.writestr("word/document.xml", document)
    return out.getvalue()


def pdf_bytes(text: str) -> bytes:
    """A one-page PDF with a text layer, one text line per line of `text`."""

    def literal(line: str) -> str:
        return (
            line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("\t", "    ")
        )

    lines = text.split("\n")
    height = 40 + 12 * len(lines)
    stream = "".join(
        f"BT /F1 9 Tf 20 {height - 20 - 12 * i} Td ({literal(line)}) Tj ET\n"
        for i, line in enumerate(lines)
    ).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 900 {height}] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>".encode(),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


@dataclass
class RecordedSite:
    """The site as tests/fixtures/collect/site.json records it; every request is logged."""

    responses: dict[str, dict[str, str]]
    robots_body: bytes
    requests: list[tuple[str, float]] = field(default_factory=list[tuple[str, float]])
    robots_reads: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    overrides: dict[str, bytes] = field(default_factory=dict[str, bytes])

    def fetch(self, url: str, interval: float) -> Fetched:
        self.requests.append((url, interval))
        if url in self.overrides:
            return Fetched(url, "application/octet-stream", self.overrides[url])
        entry = self.responses.get(url)
        if entry is None:
            raise CollectFetchError("HTTP 404")
        if "body" in entry:
            body = entry["body"].encode()
        else:
            source = (COLLECT / entry["file"]).read_text()
            body = {"docx": docx_bytes, "pdf": pdf_bytes}[entry["build"]](source)
        return Fetched(url, entry["content_type"], body)

    def read_robots(self, scheme: str, host: str) -> Robots:
        self.robots_reads.append((scheme, host))
        url = f"{scheme}://{host}/robots.txt"
        return parse_robots(url, 200, self.robots_body, "2026-11-15T00:00:00+00:00")


def recorded_site() -> RecordedSite:
    responses = json.loads((COLLECT / "site.json").read_text())
    responses.pop("_note")
    return RecordedSite(responses, (COLLECT / "robots.txt").read_bytes())

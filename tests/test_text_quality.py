"""The text layer's quality test: a seeded sample per state and format, against Open States."""

from typing import Any

from lineage.text_fetch import Job
from lineage.text_quality import draw_sample, evaluate, overlap


def _plan(state: str, kind: str, n: int, *, html: bool = True) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{state}-{kind}-{i:03}", "state": state, "kind": kind, "outcome": "planned",
            "note": "Introduced", "date": "",
            "html_url": f"https://{state}.example/{kind}/{i}.htm" if html else None,
            "pdf_url": f"https://{state}.example/{kind}/{i}.pdf",
        }
        for i in range(n)
    ]  # fmt: skip


def test_each_state_and_format_gets_fifty_with_twenty_that_have_text() -> None:
    plan = _plan("NJ", "blank", 100) + _plan("NJ", "usable", 100)

    sample = draw_sample(plan)

    for fmt in ("html", "pdf"):
        groups = [g for g, job in sample if job.format == fmt]
        assert (groups.count("without_text"), groups.count("with_text")) == (30, 20)


def test_a_short_side_is_made_up_from_the_other() -> None:
    plan = _plan("MA", "blank", 100, html=False) + _plan("MA", "usable", 5, html=False)

    groups = [g for g, _ in draw_sample(plan)]

    assert (groups.count("without_text"), groups.count("with_text")) == (45, 5)


def test_the_sample_is_fixed_by_its_seed() -> None:
    plan = _plan("NJ", "blank", 100) + _plan("NJ", "usable", 100)

    assert draw_sample(plan) == draw_sample(plan)
    assert draw_sample(plan, seed=1) != draw_sample(plan)


def test_overlap_is_word_five_gram_jaccard() -> None:
    a = "Section 1. The board shall meet each month in the capital city."
    assert overlap(a, a) == 1.0
    assert overlap(a, "An unrelated sentence about fees and their schedule for permits.") == 0.0


def test_evaluate_compares_our_text_with_theirs_on_the_same_record() -> None:
    bill = "Section 1. Short title.\nThis act is the Widget Act.\nSection 2. Fees.\nFive dollars."
    url = "https://nj.example/usable/0.htm"
    sample = [
        ("without_text", Job("a", "https://nj.example/blank/0.htm", "html", "Introduced", "")),
        ("with_text", Job("b", url, "html", "Introduced", "")),
    ]
    fetched = {
        (i, "html"): {"outcome": "ok", "text": bill, "body_bytes": 100, "seconds": 0.5}
        for i in ("a", "b")
    }
    versions: list[dict[str, Any]] = [
        {"note": "Introduced", "links": [{"url": url, "media_type": "text/html"}]}
    ]
    theirs: dict[str, tuple[str | None, str | None, list[dict[str, Any]]]] = {
        "b": ("one long paragraph " * 50, url, versions)
    }

    cell = evaluate(sample, fetched, theirs, {"a": "NJ", "b": "NJ"})["NJ"]["html"]

    assert cell["clean_rate"] == {
        "without_text": 1.0, "with_text_ours": 1.0, "with_text_theirs": 0.0,
        "without_text_pdf_layout": None,
    }  # fmt: skip
    assert cell["same_version_rate"] == 1.0
    assert cell["sampled"] == {"without_text": 1, "with_text": 1}

"""The private-material scan: names matched in any case, home paths only as written."""

from tools.private_scan import findings, names_pattern

NAMES = names_pattern(["Alexis", "Quillfeather"])


def test_names_match_in_any_case() -> None:
    text = "intro\nby alexis\nQUILLFEATHER wrote\n"

    assert list(findings(text, NAMES)) == [(2, "private name"), (3, "private name")]


def test_home_directory_paths_are_found() -> None:
    # Built at run time, or the scan would flag this file.
    text = "cd /" + "Users/someone/code\nls /" + "home/someone\n"

    assert list(findings(text, None)) == [(1, "home-directory path"), (2, "home-directory path")]


def test_a_url_with_a_home_segment_is_not_a_home_directory() -> None:
    text = "https://legislature.mi.gov/Home/GetObject?objectName=2025-HB-4045.pdf\n"

    assert list(findings(text, NAMES)) == []


def test_no_names_set_checks_home_paths_only() -> None:
    assert names_pattern(["", " "]) is None

"""The leak check: deny-listed terms matched in any case, local paths only as written."""

from tools.leak_check import findings, terms_pattern

TERMS = terms_pattern(["staging-bucket", "internal-codename"])


def test_terms_match_in_any_case() -> None:
    text = "intro\nsee staging-bucket\nINTERNAL-CODENAME build\n"

    assert list(findings(text, TERMS)) == [(2, "deny-listed term"), (3, "deny-listed term")]


def test_local_paths_are_found() -> None:
    # Built at run time, or the check would flag this file.
    text = "\n".join(
        [
            "cd /" + "Users/someone/code",
            "ls /" + "home/someone",
            '"plan": "/private/' + 'tmp/run-1/plan.parquet"',
            "/" + "var/folders/xy/T/out.json",
        ]
    )

    assert list(findings(text, None)) == [(n, "local path") for n in range(1, 5)]


def test_a_url_with_a_home_segment_is_not_a_local_path() -> None:
    text = "https://legislature.mi.gov/Home/GetObject?objectName=2025-HB-4045.pdf\n"

    assert list(findings(text, TERMS)) == []


def test_no_terms_set_checks_local_paths_only() -> None:
    assert terms_pattern(["", " "]) is None

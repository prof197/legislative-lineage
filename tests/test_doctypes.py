import json
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.doctypes import (
    RULES,
    DocTypesError,
    build_eligibility,
    decide,
    load_rules,
    rule_samples,
)
from lineage.ingest import ingest
from lineage.sectioning import build_sections
from lineage.snapshot import scan, write_manifest
from tests.conftest import SESSIONS

HEADER = "state,type,title,unless,decision,reason\n"


def _rules(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "rules.csv"
    path.write_text("# comment lines are skipped\n" + HEADER + body)
    return path


def test_the_first_rule_that_fits_decides(tmp_path: Path) -> None:
    rules = load_rules(
        _rules(
            tmp_path,
            "*,resolution,^commend,,exclude,ceremonial\n"
            "GA,resolution,-\\s*CA$,,match,amendment\n"
            "*,resolution,,,match,everything else\n",
        )
    )

    def reason(state: str, title: str) -> str | None:
        rule = decide(rules, state, "resolution", title)
        return rule.reason if rule else None

    assert reason("GA", "Commending a team -CA") == "ceremonial"
    assert reason("GA", "Taxation; provide -CA") == "amendment"
    assert reason("TX", "Taxation; provide -CA") == "everything else"


def test_a_record_no_rule_fits_is_excluded(tmp_path: Path) -> None:
    rules = load_rules(_rules(tmp_path, "*,bill,,,match,bill\n"))

    assert decide(rules, "GA", "proclamation", "Proclaiming a day") is None


def test_patterns_ignore_case_and_unless_vetoes_a_title(tmp_path: Path) -> None:
    rules = load_rules(
        _rules(tmp_path, "*,joint resolution,^recogniz,constitution,exclude,commemorative\n")
    )

    assert decide(rules, "UT", "joint resolution", "RECOGNIZING teachers") is not None
    assert decide(rules, "MO", "joint resolution", "Recognizes the Constitution") is None


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("*,bill,,,allow,bill\n", "decision"),
        ("*,bill,(,,match,bill\n", "pattern"),
        ("*,bill,,,match,\n", "reason"),
    ],
)
def test_rejects_a_malformed_rule_naming_its_line(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(DocTypesError, match=rf"line 3: .*{message}"):
        load_rules(_rules(tmp_path, body))


# The committed table, checked on titles from the snapshot. Each pair is a real record.
@pytest.mark.parametrize(
    ("state", "kind", "title", "matched"),
    [
        ("VA", "joint resolution", "Commending Mobile Hope.", False),
        ("WI", "joint resolution", "Relating to: congratulating the Grafton Black Hawks.", False),
        (
            "UT",
            "joint resolution",
            "H.J.R. 21 Joint Resolution Honoring the Hispanic Community",
            False,
        ),
        ("VA", "joint resolution", "Commemorating the life and legacy of Secretariat.", False),
        (
            "NJ",
            "joint resolution",
            "Commemorates 75th anniversary of Princeton Plasma Physics Laboratory.",
            False,
        ),
        (
            "WI",
            "joint resolution",
            "Relating to: commemorating the 80th anniversary of D-Day.",
            False,
        ),
        (
            "WI",
            "joint resolution",
            "Relating to: paying tribute to the entire Polish community that has lived in "
            "Wisconsin for 180 years and contributed to its strength.",
            False,
        ),
        # Tennessee titles every memorial "Memorials, <occasion> - <honoree> -", and an honoree's
        # name can carry a word the guard reads as law, so its rule has no guard.
        (
            "TN",
            "joint resolution",
            "Memorials, Recognition - Fort Loudoun Electric Cooperative, 85th anniversary -",
            False,
        ),
        (
            "TN",
            "joint resolution",
            "Memorials, Recognition - Fallen law enforcement officers of Hawkins County  -",
            False,
        ),
        (
            "TN",
            "joint resolution",
            "Memorials, Congress - Urges Congress to bring home the remains of the USS Intrepid "
            "sailors entombed in Libya. -",
            True,
        ),
        # Commemorative and procedural joint resolutions, whatever their wording.
        (
            "KY",
            "joint resolution",
            "A JOINT RESOLUTION designating the Nelson County Veterans Bridge on United States "
            "Highway 31E.",
            False,
        ),
        (
            "TN",
            "joint resolution",
            "Naming and Designating - Designates Pumpkin Pie as a symbol of the State of "
            "Tennessee. -",
            False,
        ),
        (
            "NJ",
            "joint resolution",
            'Designates June 25 of each year as "Delaware and Raritan Canal Day."',
            False,
        ),
        (
            "WI",
            "joint resolution",
            "Relating to: proclaiming the months of October 2025 and October 2026 as Cybersecurity "
            "Awareness Month.",
            False,
        ),
        ("VA", "joint resolution", "Governor; confirming appointments.", False),
        ("IL", "joint resolution", "ADJOURN HOUSE & SENATE", False),
        ("UT", "joint resolution", "Joint Rules Resolution - Amendments to Joint Rules", False),
        (
            "TN",
            "joint resolution",
            "General Assembly, Joint Conventions - Calls joint convention to elect comptroller and "
            "state treasurer on January 11, 2023. -",
            False,
        ),
        (
            "CT",
            "joint resolution",
            "RESOLUTION GRANTING THE CLAIMS COMMISSIONER AN EXTENSION OF TIME TO DISPOSE OF THE "
            "CLAIM OF THE CITY OF HARTFORD.",
            False,
        ),
        # Positions, calls on Congress, and changes to the process itself are kept.
        (
            "AK",
            "joint resolution",
            "Encouraging the United States Congress to designate 2025 as the Year of the Soldier.",
            True,
        ),
        (
            "TN",
            "joint resolution",
            "General Assembly, Statement of Intent or Position - Supports "
            "Tennessee\u2019s response to the United States Department of Energy Request for "
            "Information on Nuclear Lifecycle Innovation Campuses. -",
            True,
        ),
        (
            "WA",
            "joint resolution",
            "Concerning senate confirmation of gubernatorial appointments to fill vacancies on the "
            "supreme court.",
            True,
        ),
        (
            "FL",
            "joint resolution",
            "Senate Confirmation of Gubernatorial Appointments and Legislative Approval of "
            "Extended States of Emergency",
            True,
        ),
        (
            "VT",
            "joint resolution",
            "Joint resolution relating to compensation of Members of the General Assembly during "
            "the remainder of the 2024 Adjourned Session.",
            True,
        ),
        (
            "TN",
            "joint resolution",
            "General Assembly, Review Or Ratification of Rules - Ratifies the increase of the 911 "
            "surcharge rate to $1.86 as approved by the Tennessee Emergency Communications "
            "Board. -",
            True,
        ),
        (
            "TX",
            "joint resolution",
            "Proposing a constitutional amendment requiring this state to observe daylight saving "
            "time year-round.",
            True,
        ),
        # Substantive joint resolutions that a looser pattern would have excluded.
        (
            "TN",
            "joint resolution",
            "Constitutional Amendments - Proposes additional language in Article II, Section 28 to "
            "prohibit the general assembly from levying, authorizing, or otherwise permitting a "
            "state tax on property.  -",
            True,
        ),
        (
            "TX",
            "joint resolution",
            "Proposing a constitutional amendment to increase the mandatory age of retirement for "
            "state justices and judges.",
            True,
        ),
        ("NJ", "joint resolution", "Recognizes right to adequate food.", True),
        (
            "WV",
            "joint resolution",
            "Honorably Discharged Veteran's Property Tax Exemption Amendment",
            True,
        ),
        (
            "AK",
            "joint resolution",
            "Urging the United States Congress to recognize public safety telecommunicators as "
            "first responders",
            True,
        ),
        (
            "GA",
            "resolution",
            "Bona Fide Conservation Use Property; maximum acreage to qualify for assessment and "
            "taxation; increase -CA",
            True,
        ),
        ("GA", "resolution", "Dr. Jane Doe; commend", False),
        (
            "NJ",
            "concurrent resolution",
            "Proposes constitutional amendment to allow lower property tax rate on improvements "
            "than on land.",
            True,
        ),
        (
            "NJ",
            "concurrent resolution",
            "Urges New Jersey State Park Service to establish visitor's center at Princeton "
            "Battlefield State Park.",
            False,
        ),
        (
            "MS",
            "concurrent resolution",
            "Constitution; amend to restore authority to tax Grand Gulf Nuclear Power Plant to "
            "Claiborne County.",
            True,
        ),
        ("MS", "concurrent resolution", "Commend the Jackson State football team.", False),
        ("CA", "bill+appropriation", "Budget Act of 2024.", True),
        ("HI", "proclamation", "Proclaiming a day", False),
    ],
)
def test_the_committed_rules(state: str, kind: str, title: str, matched: bool) -> None:
    rule = decide(load_rules(RULES), state, kind, title)

    assert (rule is not None and rule.match) is matched


@pytest.fixture
def tables(raw_dir: Path, tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.json"
    write_manifest(scan(raw_dir), manifest)
    out = tmp_path / "tables"
    ingest(raw_dir, manifest, SESSIONS, out, floor_bytes=0)
    build_sections(out, workers=1, floor_bytes=0)
    return out


def test_writes_eligibility_and_a_report_counting_each_rule(tables: Path, tmp_path: Path) -> None:
    report = tmp_path / "document_types.json"

    doc = build_eligibility(tables, rules_path=RULES, report_path=report)

    con = duckdb.connect()
    rows = con.execute(
        f"SELECT b.state, b.identifier, e.eligible, e.rule "
        f"FROM read_parquet('{tables / 'eligible.parquet'}') e "
        f"JOIN read_parquet('{tables / 'bills' / '*.parquet'}') b USING (id) ORDER BY ALL"
    ).fetchall()
    numbers = {(r.state, r.type, r.title is None): r.number for r in load_rules(RULES)}
    bill, resolution = numbers["*", "bill", True], numbers["*", "resolution", True]
    # SR 1065 recognizes a week: a simple resolution, excluded by the default rule.
    assert ("GA", "SR 1065", False, resolution) in rows
    assert ("GA", "SB 238", True, bill) in rows
    assert all(rule is not None for *_, rule in rows)

    assert json.loads(report.read_text()) == doc
    assert doc["command"] == "uv run lineage eligible"
    by_rule = {r["rule"]: r for r in doc["rules"]}
    assert by_rule[bill]["records"] == 4
    assert by_rule[resolution]["records"] == 1
    assert by_rule[resolution]["by_state"] == {"GA": 1}
    assert doc["totals"] == {"records": 5, "eligible": 4, "eligible_with_sections": 2}
    assert doc["unlisted_types"] == {}


def test_reports_states_whose_eligible_records_all_lack_text(tables: Path, tmp_path: Path) -> None:
    doc = build_eligibility(tables, rules_path=RULES, report_path=tmp_path / "r.json")

    # Alaska's only record has blank text and Hawaii's none at all; Georgia's have text.
    assert doc["states_without_text"] == ["AK", "HI"]


SAMPLED_RULES = (
    "*,resolution,recogni,crime,exclude,ceremonial\n"
    "*,resolution,week,,exclude,observance\n"
    "*,bill,,,match,bill\n"
)


def test_the_committed_report_counts_rules_without_naming_titles(
    tables: Path, tmp_path: Path
) -> None:
    rules = _rules(tmp_path, SAMPLED_RULES)

    doc = build_eligibility(tables, rules_path=rules, report_path=tmp_path / "r.json")

    assert "Crime Victim" not in json.dumps(doc)
    assert doc["rules"][0]["spared_by_state"] == {"GA": 1}
    assert doc["rules"][1]["by_state"] == {"GA": 1}


def test_samples_name_the_titles_each_rule_caught_or_spared(tables: Path, tmp_path: Path) -> None:
    rules = _rules(tmp_path, SAMPLED_RULES)

    doc = rule_samples(tables, rules_path=rules)

    assert doc["command"] == "uv run lineage stats --samples"
    spared, caught = doc["rules"]
    assert spared["rule"] == 1
    assert spared["sample_titles"] == []
    assert [t[:40] for t in spared["spared_titles"]] == ["GA: National Crime Victim's Rights Week;"]
    assert caught["rule"] == 2
    assert [t[:40] for t in caught["sample_titles"]] == ["GA: National Crime Victim's Rights Week;"]


def test_stats_writes_the_samples_only_on_request(tables: Path, tmp_path: Path) -> None:
    local = tmp_path / "local" / "samples.json"
    args = [
        "stats",
        "--rules",
        str(_rules(tmp_path, SAMPLED_RULES)),
        "--tables",
        str(tables),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--sessions",
        str(SESSIONS),
        "--report",
        str(tmp_path / "coverage.json"),
        "--samples-report",
        str(local),
    ]

    assert CliRunner().invoke(app, args).exit_code == 0
    assert not local.exists()
    result = CliRunner().invoke(app, [*args, "--samples"])
    assert result.exit_code == 0, result.output
    assert json.loads(local.read_text())["rules"]

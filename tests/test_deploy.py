"""The live track's deploy (DESIGN.md#updates, DR-13): the release packed from the day's export,
the gates, the circuit breaker, a held day and its release, and the review files, with every
command that would leave this machine (git, gh, npm, the leak check) run by a fake."""

import inspect
import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage import deploy
from lineage.cli import app
from lineage.deploy import (
    JOB_SHELL,
    TOKEN_SERVICE,
    Breaker,
    Call,
    DeployError,
    DeployPaths,
    DeployRules,
    Limits,
    Result,
    breaker,
    deploy_day,
    github_shell,
    in_session,
    listed_origin_claims,
    load_rules,
    local_paths,
    origin_claims_review,
    prepare_candidate,
    read_record,
    reject_day,
    release_day,
    tag_for,
    weekly_review,
    write_review,
)
from lineage.export_schema import Release
from lineage.feed import write_feed
from lineage.origins.matches import FLOOR
from lineage.origins.withheld import Reviewed
from lineage.origins.withheld import write_report as write_withheld
from lineage.publish import export_changes, promote, staging_dir
from lineage.release import RELEASE_MANIFEST, Checksums, changed_files, read_checksums
from lineage.updates import OriginGained, OriginLost, RecordAdded, write_changes
from tests.days import DAYS
from tests.live import Recorded, export_inputs, full_export, recorded
from tools.leak_check import LOCAL_PATH

SITE = "https://example.org/lineage"
D2, D3 = (date.fromisoformat(d) for d in DAYS[1:])


@pytest.fixture(scope="module")
def rec(tmp_path_factory: pytest.TempPathFactory) -> Recorded:
    return recorded(tmp_path_factory)


@pytest.fixture(scope="module")
def live(rec: Recorded, tmp_path_factory: pytest.TempPathFactory) -> dict[int, Path]:
    """The live export after day 2 and after day 3, each as the daily job leaves it: the pages
    a day touches rewritten, the feed written, the staging copy promoted."""
    root = tmp_path_factory.mktemp("deploy-live")
    inputs = export_inputs(root / "inputs")
    export = full_export(rec.tables[1], root / "GA", inputs)
    out: dict[int, Path] = {}
    for n in (2, 3):
        log = rec.log_through(n, root / f"log{n}")
        export_changes(
            rec.tables[n], state="GA", export=export, changes_dir=log, models_dir=None, floor=0,
            **inputs,
        )  # fmt: skip
        write_feed(
            rec.tables[n],
            state="GA",
            export=staging_dir(export),
            site_url=SITE,
            changes_dir=log,
            modeldocs_root=inputs["modeldocs_root"],
            collector_package=inputs["collector_package"],
        )
        promote(export)
        out[n] = shutil.copytree(export, root / f"after{n}")
    return out


class Fake:
    """The commands the deploy runs, recorded; each answers as it would, or fails when told."""

    def __init__(self, checkout: Path, *, site_bytes: int = 150_000_000) -> None:
        self.checkout = checkout
        self.site_bytes = site_bytes
        self.calls: list[Call] = []
        self.fail: dict[str, int] = {}  # a command's start -> how many more times it fails
        self.download: Path | None = None

    def __call__(self, call: Call) -> Result:
        self.calls.append(call)
        line = " ".join(call.argv)
        for start, times in list(self.fail.items()):
            if line.startswith(start) and times > 0:
                self.fail[start] = times - 1
                return Result(1, f"error: {start} failed (test)")
        if line == "npm run size":
            report = self.checkout / "reports" / "site.json"
            report.write_text(json.dumps({"total_bytes": self.site_bytes, "files": 1}))
        if line.startswith("gh release download") and self.download is not None:
            where = Path(call.argv[call.argv.index("--dir") + 1])
            shutil.copy(self.download, where / self.download.name)
        return Result(0, "")

    def lines(self) -> list[str]:
        return [" ".join(c.argv) for c in self.calls]

    def published(self) -> list[str]:
        return [
            line
            for line in self.lines()
            if line.startswith(("git commit", "gh release create", "git push", "gh workflow"))
        ]


def job_checkout(root: Path) -> Path:
    """The job's clone, as far as the deploy reads it: a repository with the pin and the
    committed size report."""
    checkout = root / "checkout"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "site").mkdir()
    (checkout / "reports").mkdir()
    (checkout / "reports" / "site.json").write_text('{"total_bytes": 1}\n')
    pin = Release(
        tag="data-ga-2026-11-12", asset="export-GA-data-ga-2026-11-12.tar.gz", sha256="0" * 64
    )
    (checkout / RELEASE_MANIFEST).write_text(json.dumps(pin.model_dump()) + "\n")
    return checkout


def _rules(*, ceiling: int = 2500, floor: int = 1000, min_history: int = 5) -> DeployRules:
    rules = load_rules()
    limits = {
        "origin_claims": Limits(floor=30, ceiling=50),
        "changed_files": Limits(floor=floor, ceiling=ceiling),
    }
    return replace(rules, breaker=replace(rules.breaker, limits=limits, min_history=min_history))


DENY_TERM = "zzdenyterm"


@pytest.fixture(autouse=True)
def deny_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deny list the deploy reads from the environment, as the daily job sets it; a term
    no fixture file holds."""
    monkeypatch.setenv("LEAK_CHECK_TERMS", f" ,{DENY_TERM}")


@pytest.fixture
def setup(live: dict[int, Path], tmp_path: Path) -> tuple[DeployPaths, Fake, Path]:
    paths = DeployPaths(root=tmp_path / "deploy", checkout=job_checkout(tmp_path))
    export = tmp_path / "data" / "GA"
    shutil.copytree(live[2], export)
    return paths, Fake(paths.checkout), export


def _to(export: Path, live: Path) -> None:
    """The live export moved on to another day, as the daily job's promote leaves it."""
    shutil.rmtree(export)
    shutil.copytree(live, export)


PUBLISH_ORDER = [
    "git pull --ff-only origin main",
    "git status --porcelain",
    "npm ci",
    "npm run build",
    "npm run test:copy",
    "npm run size",
    "git checkout -- reports/site.json",
    "git add site/data-release.json",
]


def test_a_day_is_packed_gated_and_published_in_order(
    setup: tuple[DeployPaths, Fake, Path], rec: Recorded
) -> None:
    paths, fake, export = setup
    record = deploy_day(
        state="GA",
        export=export,
        changes_dir=rec.log,
        paths=paths,
        shell=fake,
        rules=_rules(),
        floor=0,
    )

    assert record.state == "deployed"
    assert record.day == D2
    tag = tag_for("GA", D2)
    assert record.tag == tag == "data-ga-2026-01-13"
    lines = fake.lines()
    assert lines[: len(PUBLISH_ORDER)] == PUBLISH_ORDER
    assert lines[len(PUBLISH_ORDER)].startswith("git commit -m build(data): data-ga-2026-01-13")
    assert lines[len(PUBLISH_ORDER) + 1] == (
        "uv run --frozen python -m tools.leak_check --history --require-terms"
    )
    asset = paths.candidates / tag / f"export-GA-{tag}.tar.gz"
    assert lines[len(PUBLISH_ORDER) + 2].startswith(f"gh release create {tag} {asset} ")
    assert lines[-2:] == ["git push origin HEAD:main", "gh workflow run pages.yml --ref main"]
    # Every command ran in the job's own checkout, the site's in its site directory.
    assert {c.cwd for c in fake.calls} == {paths.checkout, paths.checkout / "site"}
    # The pin names the asset by its digest, and the site was built from the unpacked release.
    pin = Release.model_validate_json((paths.checkout / RELEASE_MANIFEST).read_text())
    assert (pin.tag, pin.asset) == (tag, asset.name)
    data = paths.checkout / "site" / "public" / "data"
    assert Release.model_validate_json((data / "release.json").read_text()) == pin
    assert (data / "checksums.json").exists()
    assert (data / "feed.xml").exists()
    assert record.gates["contract"]["ok"]
    assert record.gates["copy"]["ok"]
    assert record.gates["size"]["share_of_pages_limit"] == 0.15
    # No release before it: every file counts as changed.
    assert record.counts == {
        "origin_claims": 0,
        "changed_files": len(read_checksums(asset).files),
    }
    assert record.done == ["commit", "leak", "release", "push", "dispatch"]
    assert read_record(paths, D2) == record

    # Run again on the same day: nothing more is published.
    fake.calls.clear()
    assert (
        deploy_day(
            state="GA",
            export=export,
            changes_dir=rec.log,
            paths=paths,
            shell=fake,
            rules=_rules(),
            floor=0,
        ).state
        == "deployed"
    )
    assert fake.calls == []


def test_the_next_day_counts_only_what_changed_since(
    setup: tuple[DeployPaths, Fake, Path], live: dict[int, Path], rec: Recorded
) -> None:
    paths, fake, export = setup
    first = deploy_day(
        state="GA",
        export=export,
        changes_dir=rec.log,
        paths=paths,
        shell=fake,
        rules=_rules(),
        floor=0,
    )
    _to(export, live[3])
    second = deploy_day(
        state="GA",
        export=export,
        changes_dir=rec.log,
        paths=paths,
        shell=fake,
        rules=_rules(),
        floor=0,
    )
    assert second.state == "deployed"
    expected = changed_files(
        read_checksums(paths.candidates / (first.tag or "") / (first.asset or "")),
        read_checksums(paths.candidates / (second.tag or "") / (second.asset or "")),
    )
    assert expected["changed"]  # counted against day 2's release, not as every file
    assert second.counts["changed_files"] == sum(len(v) for v in expected.values())
    assert second.baseline["origin_claims"]["rule"] == "ceiling"  # one day is too little history


def test_a_jump_holds_the_day_and_a_later_day_waits_behind_it(
    setup: tuple[DeployPaths, Fake, Path], live: dict[int, Path], rec: Recorded
) -> None:
    paths, fake, export = setup
    # With nothing before it, day 2 changes every one of the fixture's few files; a ceiling of
    # 10 holds it.
    rules = _rules(ceiling=10)

    held = deploy_day(
        state="GA",
        export=export,
        changes_dir=rec.log,
        paths=paths,
        shell=fake,
        rules=rules,
        floor=0,
    )
    assert held.state == "held"
    assert held.held_on_counts
    assert held.reasons
    assert "changed files" in held.reasons[0]
    assert "ceiling of 10" in held.reasons[0]
    assert held.baseline["changed_files"] == {"median": None, "limit": 10.0, "rule": "ceiling"}
    assert fake.published() == []
    assert (held.tag or "") not in (paths.checkout / RELEASE_MANIFEST).read_text()

    _to(export, live[3])
    behind = deploy_day(
        state="GA",
        export=export,
        changes_dir=rec.log,
        paths=paths,
        shell=fake,
        rules=_rules(),  # its own counts pass
        floor=0,
    )
    assert behind.state == "held"
    assert not behind.held_on_counts
    assert behind.held_behind == D2
    assert behind.reasons == ["held behind 2026-01-13, which awaits review"]
    assert fake.published() == []

    with pytest.raises(DeployError, match="2026-01-13 is held before it"):
        release_day(D3, paths=paths, rules=rules, shell=fake)
    released = release_day(D2, paths=paths, rules=rules, shell=fake)
    assert released.state == "released"
    assert [line.split()[:4] for line in fake.published()][1] == [
        "gh",
        "release",
        "create",
        "data-ga-2026-01-13",
    ]
    pin = Release.model_validate_json((paths.checkout / RELEASE_MANIFEST).read_text())
    assert pin.tag == "data-ga-2026-01-13"
    assert "released after the maintainer's review" in " ".join(fake.calls[-5].argv)

    # With day 2 released, the next run publishes day 3, which no longer waits.
    again = deploy_day(
        state="GA",
        export=export,
        changes_dir=rec.log,
        paths=paths,
        shell=fake,
        rules=rules,
        floor=0,
    )
    assert again.state == "deployed"
    assert Release.model_validate_json((paths.checkout / RELEASE_MANIFEST).read_text()).tag == (
        "data-ga-2026-01-14"
    )
    with pytest.raises(DeployError, match="not held"):
        release_day(D3, paths=paths, rules=rules, shell=fake)


def _name_in_a_bill(export: Path, word: str) -> str:
    """A bill page whose title carries `word`, as a public record might; returns its path."""
    page = sorted((export / "bills").glob("*.json"))[0]
    doc = json.loads(page.read_text())
    doc["bill"]["title"] = f"{doc['bill']['title']} {word}"
    page.write_text(json.dumps(doc))
    return page.relative_to(export).as_posix()


def test_a_changed_file_naming_a_deny_listed_term_holds_the_day(
    setup: tuple[DeployPaths, Fake, Path], rec: Recorded
) -> None:
    """Ruled at the deploy review: the deny list is read over the release's files too. A hit
    holds the day for review rather than failing it, since public records name people, and the
    record names the file, never the term."""
    paths, fake, export = setup
    named = _name_in_a_bill(export, DENY_TERM.upper())

    held = deploy_day(
        state="GA", export=export, changes_dir=rec.log, paths=paths, shell=fake, rules=_rules(),
        floor=0,
    )  # fmt: skip
    assert held.state == "held"
    assert held.held_on_counts
    assert held.gates["deny_list"]["files"] == [named]
    assert held.gates["deny_list"]["ok"] is False
    (reason,) = held.reasons
    assert f"1 files added or changed name a deny-listed term ({named})" in reason
    assert DENY_TERM not in reason.lower()
    assert fake.published() == []


def test_a_term_inside_a_longer_word_is_not_the_term(
    setup: tuple[DeployPaths, Fake, Path], rec: Recorded
) -> None:
    paths, fake, export = setup
    _name_in_a_bill(export, f"Fitz{DENY_TERM}")

    done = deploy_day(
        state="GA", export=export, changes_dir=rec.log, paths=paths, shell=fake, rules=_rules(),
        floor=0,
    )  # fmt: skip
    assert done.state == "deployed"
    assert done.gates["deny_list"]["ok"] is True


def test_without_the_deny_list_nothing_is_deployed(
    setup: tuple[DeployPaths, Fake, Path], rec: Recorded, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, fake, export = setup
    monkeypatch.delenv("LEAK_CHECK_TERMS")
    with pytest.raises(DeployError, match="LEAK_CHECK_TERMS is not set"):
        deploy_day(
            state="GA", export=export, changes_dir=rec.log, paths=paths, shell=fake,
            rules=_rules(), floor=0,
        )  # fmt: skip
    assert fake.published() == []


def test_a_rejected_day_is_never_published_and_the_next_day_publishes_as_usual(
    setup: tuple[DeployPaths, Fake, Path], live: dict[int, Path], rec: Recorded
) -> None:
    """Ruled at the deploy review: a held day can be rejected. Its pages are not deployed, the
    live track keeps its data, and the next good day publishes normally, counted against the
    rejected day's release."""
    paths, fake, export = setup
    held = deploy_day(
        state="GA", export=export, changes_dir=rec.log, paths=paths, shell=fake,
        rules=_rules(ceiling=10), floor=0,
    )  # fmt: skip
    assert held.state == "held"

    rejected = reject_day(D2, paths=paths)
    assert rejected.state == "rejected"
    assert rejected.reasons[-1].startswith("rejected after review on ")
    assert fake.published() == []
    with pytest.raises(DeployError, match="not held \\(rejected\\)"):
        release_day(D2, paths=paths, rules=_rules(), shell=fake)
    with pytest.raises(DeployError, match="only a held day is rejected"):
        reject_day(D2, paths=paths)

    _to(export, live[3])
    nxt = deploy_day(
        state="GA", export=export, changes_dir=rec.log, paths=paths, shell=fake, rules=_rules(),
        floor=0,
    )  # fmt: skip
    # Not held behind the rejected day, and counted against its release, which stays packed as
    # the reference, so day 2's changes are not counted again.
    assert nxt.state == "deployed", nxt.reasons
    sums = {
        d: Checksums.model_validate_json(
            (paths.candidates / tag_for("GA", d) / "checksums.json").read_text()
        )
        for d in (D2, D3)
    }
    diff = changed_files(sums[D2], sums[D3])
    assert nxt.counts["changed_files"] == sum(len(v) for v in diff.values())
    assert Release.model_validate_json((paths.checkout / RELEASE_MANIFEST).read_text()).tag == (
        "data-ga-2026-01-14"
    )
    assert read_record(paths, D2).state == "rejected"  # type: ignore[union-attr]


def test_bad_claims_are_withheld_then_the_day_is_approved_repacked(
    setup: tuple[DeployPaths, Fake, Path], live: dict[int, Path], rec: Recorded, tmp_path: Path
) -> None:
    """Ruled at the deploy review: a held day with bad claims is not fixed in the tables. The
    documents behind them are withheld in the committed report, the live export written again,
    and the day approved: repacked from the export as it now stands, since its own release still
    lists the withheld documents."""
    paths, fake, export = setup
    report = tmp_path / "withheld.json"
    write_withheld([], report, [])
    held = deploy_day(
        state="GA", export=export, changes_dir=rec.log, paths=paths, shell=fake,
        rules=_rules(ceiling=10), floor=0, withheld_report=report,
    )  # fmt: skip
    assert held.state == "held"
    _to(export, live[3])
    behind = deploy_day(
        state="GA", export=export, changes_dir=rec.log, paths=paths, shell=fake, rules=_rules(),
        floor=0, withheld_report=report,
    )  # fmt: skip
    assert behind.held_behind == D2

    # The maintainer withholds a document: the day's own release is now out of date.
    gone = Reviewed(doc_id="lib/bad", title="Bad", listed=1, note="wrong", on="2027-01-12")
    write_withheld([], report, [gone])
    with pytest.raises(DeployError, match="has changed since 2026-01-13 was packed"):
        release_day(D2, paths=paths, rules=_rules(), shell=fake, withheld_report=report)
    # Repacked before the export is written under the new report: refused.
    with pytest.raises(DeployError, match="not written under the withhold report"):
        release_day(
            D2, paths=paths, rules=_rules(), shell=fake, withheld_report=report, repack=True,
            export=export, state="GA",
        )  # fmt: skip
    assert fake.published() == []

    # Written again (here: a report whose list the export's sources page matches), approved.
    write_withheld([], report, [])
    report.write_text(
        report.read_text().replace('"documents_with_labels": 0', '"documents_with_labels": 1')
    )
    approved = release_day(
        D2, paths=paths, rules=_rules(), shell=fake, withheld_report=report, repack=True,
        export=export, state="GA",
    )  # fmt: skip
    assert approved.state == "released"
    assert approved.tag == "data-ga-2026-01-14"  # the export as it stands, through day 3
    assert approved.reasons[-1] == "approved after review; repacked as of 2026-01-14"
    pin = Release.model_validate_json((paths.checkout / RELEASE_MANIFEST).read_text())
    assert pin.tag == "data-ga-2026-01-14"
    assert read_record(paths, D3).state == "superseded"  # type: ignore[union-attr]


def test_the_reject_command_refuses_a_day_not_held(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["publish", "reject", "--day", "2027-01-12", "--deploy-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "2027-01-12 is not held (no record); only a held day is rejected" in result.output


def test_every_git_and_gh_call_carries_the_keychain_token_and_no_other_login(
    tmp_path: Path,
) -> None:
    """Ruled at the deploy review: the job exports its own token before every git or gh call,
    read afresh, so it can never fall back to the maintainer's gh login."""
    seen: list[Call] = []
    reads: list[int] = []

    def inner(call: Call) -> Result:
        seen.append(call)
        return Result(0, "")

    def token() -> str:
        reads.append(1)
        return "job-token"

    shell = github_shell(inner, token, tmp_path / "gh-config")
    shell(Call(("git", "push", "origin", "HEAD:main"), tmp_path))
    shell(Call(("gh", "release", "create", "t"), tmp_path))
    shell(Call(("npm", "ci"), tmp_path))

    push, release, npm = seen
    assert push.argv == (
        "git", "-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential",
        "push", "origin", "HEAD:main",
    )  # fmt: skip
    for call in (push, release):
        assert call.env["GH_TOKEN"] == "job-token"
        assert call.env["GH_CONFIG_DIR"] == str(tmp_path / "gh-config")  # no stored login there
        assert call.env["GIT_TERMINAL_PROMPT"] == "0"
        assert call.env["GH_PROMPT_DISABLED"] == "1"
    assert release.argv == ("gh", "release", "create", "t")
    assert npm == Call(("npm", "ci"), tmp_path)  # nothing else gets the token
    assert len(reads) == 2  # read before each git or gh call


def test_without_the_token_no_git_or_gh_call_runs(tmp_path: Path) -> None:
    ran: list[Call] = []
    shell = github_shell(lambda c: (ran.append(c), Result(0, ""))[1], lambda: None, tmp_path)
    with pytest.raises(DeployError, match=TOKEN_SERVICE):
        shell(Call(("gh", "workflow", "run", "pages.yml"), tmp_path))
    with pytest.raises(DeployError, match="uses no other login"):
        shell(Call(("git", "pull", "--ff-only", "origin", "main"), tmp_path))
    assert ran == []


def test_the_deploy_stops_before_anything_when_the_keychain_holds_no_token(
    setup: tuple[DeployPaths, Fake, Path], rec: Recorded
) -> None:
    paths, fake, export = setup
    with pytest.raises(DeployError, match=TOKEN_SERVICE):
        deploy_day(
            state="GA", export=export, changes_dir=rec.log, paths=paths, rules=_rules(), floor=0,
            shell=github_shell(fake, lambda: None, paths.root / "gh-config"),
        )  # fmt: skip
    assert fake.lines() == []  # not even the pull
    assert read_record(paths, D2).state == "failed"  # type: ignore[union-attr]


def test_the_commands_publish_through_the_token_shell_by_default() -> None:
    for fn in (deploy_day, release_day, prepare_candidate):
        assert inspect.signature(fn).parameters["shell"].default is JOB_SHELL


def test_a_failed_leak_check_undoes_the_commit_and_publishes_nothing(
    setup: tuple[DeployPaths, Fake, Path], rec: Recorded
) -> None:
    paths, fake, export = setup
    fake.fail["uv run --frozen python -m tools.leak_check"] = 1
    with pytest.raises(DeployError, match="leak check failed"):
        deploy_day(
            state="GA",
            export=export,
            changes_dir=rec.log,
            paths=paths,
            shell=fake,
            rules=_rules(),
            floor=0,
        )
    assert "git reset --hard HEAD~1" in fake.lines()
    assert not any(line.startswith(("gh release create", "git push")) for line in fake.lines())
    record = read_record(paths, D2)
    assert record is not None
    assert record.state == "failed"
    assert record.done == []
    assert "leak check" in (record.error or "")


def test_a_stopped_publish_resumes_without_a_second_release(
    setup: tuple[DeployPaths, Fake, Path], rec: Recorded
) -> None:
    paths, fake, export = setup
    fake.fail["git push"] = 1
    with pytest.raises(DeployError, match="git push failed"):
        deploy_day(
            state="GA",
            export=export,
            changes_dir=rec.log,
            paths=paths,
            shell=fake,
            rules=_rules(),
            floor=0,
        )
    stopped = read_record(paths, D2)
    assert stopped is not None
    assert stopped.done == ["commit", "leak", "release"]
    fake.calls.clear()
    record = deploy_day(
        state="GA",
        export=export,
        changes_dir=rec.log,
        paths=paths,
        shell=fake,
        rules=_rules(),
        floor=0,
    )
    assert record.state == "deployed"
    assert fake.lines() == [
        "git status --porcelain",
        "git push origin HEAD:main",
        "gh workflow run pages.yml --ref main",
    ]


@pytest.mark.parametrize(
    ("fails", "match"),
    [("npm run test:copy", "copy check failed"), ("npm run build", "build failed")],
)
def test_a_failed_site_gate_publishes_nothing(
    setup: tuple[DeployPaths, Fake, Path], rec: Recorded, fails: str, match: str
) -> None:
    paths, fake, export = setup
    fake.fail[fails] = 1
    with pytest.raises(DeployError, match=match):
        deploy_day(
            state="GA",
            export=export,
            changes_dir=rec.log,
            paths=paths,
            shell=fake,
            rules=_rules(),
            floor=0,
        )
    assert fake.published() == []
    assert not any(line.startswith("git add") for line in fake.lines())


def test_the_size_gate_stops_past_half_the_pages_limit(
    setup: tuple[DeployPaths, Fake, Path], rec: Recorded
) -> None:
    paths, _, export = setup
    fake = Fake(paths.checkout, site_bytes=500_000_001)
    with pytest.raises(DeployError, match="50% the size gate allows"):
        deploy_day(
            state="GA",
            export=export,
            changes_dir=rec.log,
            paths=paths,
            shell=fake,
            rules=_rules(),
            floor=0,
        )
    assert fake.published() == []
    assert "git checkout -- reports/site.json" in fake.lines()  # the committed report restored


def test_an_export_naming_a_local_path_is_not_packed(
    setup: tuple[DeployPaths, Fake, Path], rec: Recorded
) -> None:
    paths, fake, export = setup
    page = next((export / "bills").glob("*.json"))
    doc = page.read_text()
    # Spelled in parts, so this file names no local path itself (tools/leak_check.py).
    page.write_text(doc.replace('"title":"', '"title":"/' + "Users/someone/notes ", 1))
    with pytest.raises(DeployError, match="names a local path"):
        deploy_day(
            state="GA",
            export=export,
            changes_dir=rec.log,
            paths=paths,
            shell=fake,
            rules=_rules(),
            floor=0,
        )
    assert fake.calls == []


def test_a_web_address_under_home_is_not_a_local_path(tmp_path: Path) -> None:
    home = "/" + "home"  # in parts, so this file names no local path itself
    (tmp_path / "a.json").write_text(f'{{"url": "https://legis.example.gov{home}/bill.pdf"}}\n')
    (tmp_path / "b.json").write_text(f'{{"note": "see {home}/someone/bill.txt"}}\n')
    assert local_paths(tmp_path) == ["b.json:1"]


def test_the_local_path_pattern_is_the_leak_checks() -> None:
    assert deploy.LOCAL_PATH.pattern == LOCAL_PATH.pattern


# The circuit breaker's rule.


def _b(**kw: int | float) -> Breaker:
    limits = {"origin_claims": Limits(10, 25), "changed_files": Limits(100, 400)}
    base = Breaker(window=4, min_history=3, multiple=3.0, limits=limits)
    return replace(base, **kw)  # pyright: ignore[reportArgumentType]


def _days(*pairs: tuple[int, int]) -> list[Mapping[str, int]]:
    return [{"origin_claims": o, "changed_files": f} for o, f in pairs]


def test_with_too_little_history_a_count_is_held_above_its_ceiling() -> None:
    history = _days((50, 900), (50, 900))  # two days: fewer than three
    assert not breaker({"origin_claims": 25, "changed_files": 400}, history, _b()).hold
    decision = breaker({"origin_claims": 26, "changed_files": 400}, history, _b())
    assert decision.hold
    assert decision.reasons == [
        "26 origin claims, past its ceiling of 25 while fewer than 3 days are deployed"
    ]


def test_with_history_a_count_is_held_above_both_the_multiple_and_the_floor() -> None:
    history = _days((1, 50), (2, 60), (100, 70), (3, 80), (4, 90))  # the window keeps the last 4
    # Median origin claims 3.5 -> 3 x 3.5 = 10.5 over a floor of 10; files 75 -> 225 over 100.
    ok = breaker({"origin_claims": 10, "changed_files": 225}, history, _b())
    assert not ok.hold
    assert ok.baseline["origin_claims"] == {"median": 3.5, "limit": 10.5, "rule": "median"}
    assert ok.baseline["days"] == 4
    held = breaker({"origin_claims": 11, "changed_files": 226}, history, _b())
    assert held.hold
    assert len(held.reasons) == 2
    # A quiet stretch does not let the limit fall under the floor.
    quiet = _days((0, 1), (0, 1), (0, 1))
    assert not breaker({"origin_claims": 10, "changed_files": 100}, quiet, _b()).hold
    assert breaker({"origin_claims": 11, "changed_files": 100}, quiet, _b()).hold


def test_the_rules_file_records_georgias_first_day_confirmed() -> None:
    rules = load_rules()
    first = rules.first_day["GA"]
    assert first == date(2027, 1, 11)
    # The General Assembly meets on the second Monday in January.
    assert first.weekday() == 0
    assert 8 <= first.day <= 14
    assert first.month == 1
    # Confirmed at the deploy review: the daily job deploys from that day, not before.
    assert rules.confirmed["GA"] is True
    assert not in_session(rules, "GA", first - timedelta(days=1))
    assert in_session(rules, "GA", first)
    unconfirmed = replace(rules, confirmed={"GA": False})
    assert not in_session(unconfirmed, "GA", first)  # unconfirmed, the date turns nothing on
    assert not in_session(rules, "UT", date(2030, 1, 1))  # no first day recorded: never
    assert rules.max_share == 0.5
    b = rules.breaker
    assert (b.window, b.min_history, b.multiple) == (10, 5, 3.0)
    assert b.limits["origin_claims"] == Limits(floor=30, ceiling=50)
    assert b.limits["changed_files"] == Limits(floor=1000, ceiling=2500)


# Origin claims: what the breaker counts and the weekly review lists.


def _gained(day: date, bill: str, doc: str, share: float) -> OriginGained:
    return OriginGained(
        as_of=day, bill_id=bill, doc_id=doc, stages=["near_duplicate"], bill_share=share,
        model_share=share / 2,
    )  # fmt: skip


def test_only_the_origin_claims_a_page_lists_are_counted() -> None:
    changes = [
        _gained(D2, "b1", "d1", FLOOR),
        _gained(D2, "b2", "d1", FLOOR - 0.01),
        _gained(D3, "b3", "d2", 0.9),
        RecordAdded(as_of=D2, bill_id="b1", state="GA", session="2025_26", text_sha256=None),
    ]
    assert [c.bill_id for c in listed_origin_claims(changes, [D2])] == ["b1"]
    assert [c.bill_id for c in listed_origin_claims(changes, [D2, D3])] == ["b1", "b3"]
    # A withheld document's matches are listed nowhere, so they are not claims.
    assert [c.bill_id for c in listed_origin_claims(changes, [D2, D3], frozenset({"d2"}))] == ["b1"]


def test_the_weekly_review_lists_the_weeks_new_origin_claims(tmp_path: Path) -> None:
    week_end = date(2027, 1, 17)
    log = tmp_path / "changes"
    write_changes(log, date(2027, 1, 10), [_gained(date(2027, 1, 10), "b0", "d1", 0.9)])
    write_changes(log, date(2027, 1, 12), [_gained(date(2027, 1, 12), "b1", "d1", 0.8)])
    write_changes(
        log,
        date(2027, 1, 14),
        [
            _gained(date(2027, 1, 14), "b2", "d2", 0.7),
            OriginLost(as_of=date(2027, 1, 14), bill_id="b1", doc_id="d1"),
        ],
    )
    paths = DeployPaths(root=tmp_path / "deploy", checkout=tmp_path / "checkout")
    out = weekly_review(
        week_end,
        changes_dir=log,
        paths=paths,
        describe=lambda ids: {i: {"state": "GA", "identifier": f"HB {i[1:]}"} for i in ids},
        documents={"d1": {"title": "Model Act", "organization": "An Organization"}},
    )
    doc = json.loads(out.read_text())
    assert doc["through"] == "2027-01-17"
    claims = doc["origin_claims"]
    assert [(c["bill_id"], c["first_listed"], c["still_listed"]) for c in claims] == [
        ("b1", "2027-01-12", False),
        ("b2", "2027-01-14", True),
    ]
    assert claims[0]["bill_share"] == 0.8
    assert claims[0]["model_share"] == 0.4
    assert claims[0]["stages"] == ["near_duplicate"]
    text = out.with_suffix(".txt").read_text()
    assert "New origin claims listed in the 7 days through 2027-01-17: 2" in text
    assert "GA HB 1 and “Model Act” (An Organization): 80% of the bill's wording" in text
    assert "no longer listed" in text
    assert "- 2027-01-14: GA HB 2 and d2:" in text  # a document not collected shows its id


def test_a_review_with_no_claims_says_so(tmp_path: Path) -> None:
    out = write_review([], through=date(2027, 1, 17), out=tmp_path / "r.json")
    assert json.loads(out.read_text())["origin_claims"] == []
    assert out.with_suffix(".txt").read_text().endswith("2027-01-17: 0\n")
    assert origin_claims_review([], through=date(2027, 1, 17)) == []


# Before session: a candidate for review.


def test_a_candidate_is_packed_gated_and_summarized_and_nothing_is_published(
    setup: tuple[DeployPaths, Fake, Path], live: dict[int, Path], rec: Recorded, tmp_path: Path
) -> None:
    paths, fake, export = setup
    # The pinned release: day 2's export, packed as the reviewed flow packs it.
    earlier = DeployPaths(root=tmp_path / "elsewhere", checkout=paths.checkout)
    pinned = deploy.pack(export, state="GA", tag="data-ga-2026-01-13", paths=earlier, floor=0)
    (paths.checkout / RELEASE_MANIFEST).write_text(json.dumps(pinned.pin.model_dump()) + "\n")
    fake.download = pinned.asset
    _to(export, live[3])

    doc = prepare_candidate(
        D3, state="GA", export=export, changes_dir=rec.log, paths=paths, shell=fake, floor=0
    )

    assert fake.published() == []
    assert not any(line.startswith("git add") for line in fake.lines())
    assert "gh release download data-ga-2026-01-13 --pattern " in " ".join(fake.lines())
    assert doc["tag"] == "data-ga-2026-01-14"
    assert doc["pinned"] == "data-ga-2026-01-13"
    assert doc["files"]["changed"]
    assert "meta.json" in doc["files"]["changed"]
    assert doc["gates"]["copy"]["ok"]
    assert doc["gates"]["size"]["ok"]
    review = paths.candidates / "data-ga-2026-01-14" / "review.txt"
    text = review.read_text()
    assert text.startswith("Release candidate data-ga-2026-01-14 for review")
    assert "nothing is published" in text
    assert re.search(r"Against the pinned release data-ga-2026-01-13: \d+ files added", text)
    assert "New origin claims listed in the 7 days through 2026-01-14: 0" in text
    # The pin is the reviewed one still.
    pin = Release.model_validate_json((paths.checkout / RELEASE_MANIFEST).read_text())
    assert pin.tag == "data-ga-2026-01-13"


def test_the_release_command_refuses_a_day_not_held(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["publish", "release", "--day", "2027-01-12", "--deploy-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "2027-01-12 is not held (no record)" in result.output

"""Publishing a day's release from the job's checkout: the pin committed and leak-checked,
then the GitHub release, the push and the Pages dispatch, each step recorded as done
(lineage.live.deploy).

See DESIGN.md#updates, DR-13.
"""

from collections.abc import Callable, Mapping

from lineage.export.release import RELEASE_MANIFEST, write_release_manifest
from lineage.live.deploy_gates import Candidate
from lineage.live.deploy_ledger import DayRecord, DeployError, DeployPaths, write_record
from lineage.live.deploy_shell import Call, Shell, run_checked

WORKFLOW = "pages.yml"
BRANCH = "main"


# Publishing.


def _commit_message(tag: str, counts: Mapping[str, int], *, reviewed: bool) -> tuple[str, str]:
    subject = f"build(data): {tag} from the live track"
    how = (
        "released after the maintainer's review of a held day"
        if reviewed
        else ("deployed by the daily job behind its gates and circuit breaker")
    )
    body = (
        f"Pins {tag}, {how} (DESIGN.md#updates). "
        f"{counts.get('changed_files', 0)} files changed against the release before it; "
        f"{counts.get('origin_claims', 0)} new origin claims listed."
    )
    return subject, body


def publish(
    record: DayRecord,
    candidate: Candidate,
    *,
    paths: DeployPaths,
    shell: Shell,
    reviewed: bool = False,
) -> DayRecord:
    """Commit the pin, check the commit for leaks, then create the release, push and dispatch
    the Pages workflow; each step is recorded as done, so a stopped run resumes after it."""
    checkout = paths.checkout

    def step(name: str, action: Callable[[], object]) -> None:
        if name in record.done:
            return
        action()
        record.done.append(name)
        write_record(paths, record)

    def commit() -> None:
        write_release_manifest(candidate.asset, tag=candidate.tag, path=checkout / RELEASE_MANIFEST)
        subject, body = _commit_message(candidate.tag, record.counts, reviewed=reviewed)
        run_checked(shell, ["git", "add", str(RELEASE_MANIFEST)], checkout, "git add")
        run_checked(shell, ["git", "commit", "-m", subject, "-m", body], checkout, "git commit")

    def leak() -> None:
        result = shell(
            Call(
                ("uv", "run", "--frozen", "python", "-m", "tools.leak_check", "--history",
                 "--require-terms"),
                checkout,
            )
        )  # fmt: skip
        if result.code != 0:
            # The commit never leaves this machine: it is undone, and the day fails.
            run_checked(shell, ["git", "reset", "--hard", "HEAD~1"], checkout, "undoing the commit")
            record.done.remove("commit")
            tail = result.output.strip().splitlines()[-1:] or [f"exit code {result.code}"]
            raise DeployError(f"the leak check failed on the commit: {tail[0][:300]}")

    notes = (
        "The live track's data for the site, packed by `lineage publish deploy` "
        "(DESIGN.md#updates). checksums.json inside lists every file's SHA-256."
    )
    step("commit", commit)
    step("leak", leak)
    step(
        "release",
        lambda: run_checked(
            shell,
            ["gh", "release", "create", candidate.tag, str(candidate.asset), "--title",
             candidate.tag, "--notes", notes],
            checkout,
            "gh release create",
        ),
    )  # fmt: skip
    step(
        "push",
        lambda: run_checked(
            shell, ["git", "push", "origin", f"HEAD:{BRANCH}"], checkout, "git push"
        ),
    )
    step(
        "dispatch",
        lambda: run_checked(
            shell, ["gh", "workflow", "run", WORKFLOW, "--ref", BRANCH], checkout, "dispatch"
        ),
    )
    return record


def ready_checkout(paths: DeployPaths, shell: Shell, *, resuming: bool) -> None:
    """The checkout brought up to main, and clean but for a commit a stopped run left."""
    if not (paths.checkout / ".git").exists():
        raise DeployError(
            f"no checkout of the repository at {paths.checkout}; clone it there once, on {BRANCH}"
        )
    if not resuming:
        run_checked(
            shell, ["git", "pull", "--ff-only", "origin", BRANCH], paths.checkout, "git pull"
        )
    dirty = run_checked(shell, ["git", "status", "--porcelain"], paths.checkout, "git status")
    if dirty.strip():
        raise DeployError(f"the checkout at {paths.checkout} has uncommitted changes")

"""The live track's deploy: each day's export packed as a release, gated, held when it jumps
past a normal day, and published (DESIGN.md#updates, DR-13).

Before a hosted state's first day of session (rules/deploy.toml) the site deploys only from a
reviewed release: the daily job stops at the local export, and the weekly job prepares a release
candidate, packed and gated with a short review summary, and publishes nothing
(`prepare_candidate`). From that day the daily job deploys after promoting the export
(`deploy_day`):

1. **Pack** the export as release `data-<st>-<as of>` (lineage.export.release), which refuses any
file
   or field the export contract does not name, and check every file for a local path.
2. **Build the site** from the unpacked release in the job's own checkout of the repository, as
   the Pages workflow builds it, and run the copy check; then measure it with
   `site/scripts/size.mjs` against the Pages limit.
3. **The circuit breaker.** The day's new origin claims (the change log's, as a page or the feed
   lists them) and its changed files (against the release before it) are set against the days
   deployed without a hold; a count past the limit holds the day (`breaker`). The files the day
   adds or changes are read for the leak check's deny list, as whole words; a file naming a
   term holds the day too (`deny_listed`), since public records name many people and whether
   one should be shown is the maintainer's to rule.
4. **Commit the pin** to `site/data-release.json` and run the leak check on that commit, the
   check CI runs, with the deny list from the environment.
5. **Publish:** create the GitHub release with the asset, push the commit, and dispatch the
   Pages workflow, which verifies the asset against the pin before it builds.

A held day publishes nothing and is recorded with its counts and baseline; the daily job alerts
on it as on a failure. Only `release_day`, which the maintainer runs after review, publishes it.
A later day is held behind it rather than published past it, since its release would carry the
held day's changes; once the held day is released, the next day's run publishes as before. Or
the maintainer rejects it (`reject_day`): its release is never published, the live track keeps
its data, and the next day's run publishes as usual, counted against the rejected day's release
so its jump is not counted twice.
Every step that leaves this machine (git, gh, npm, the leak check) goes through an injectable
`Shell`, so the tests run the whole flow with fakes. Each day's record, under `<data>/deploy/`,
lists the steps done, so a run stopped part way resumes rather than publishing twice.

This module runs a day, a held day's release or rejection, and the reviews. The steps are
modules of their own: the ledger of days (lineage.live.deploy_ledger), the shell and the job's
token (deploy_shell), packing and the gates (deploy_gates), the rules and the circuit breaker
(deploy_rules), and publishing (deploy_publish).

See DESIGN.md#updates, DR-13.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from lineage.core.hashing import file_sha256
from lineage.export.release import RELEASE_MANIFEST, ReleaseError, changed_files
from lineage.live.deploy_gates import (
    candidate_of,
    changed_paths,
    deny_gate,
    deny_listed,
    deny_pattern,
    deny_reason,
    export_day,
    pack,
    pinned_checksums,
    previous_candidate,
    site_gates,
    tag_day,
    tag_for,
)
from lineage.live.deploy_ledger import (
    DayRecord,
    DeployError,
    DeployPaths,
    blocking_day,
    read_record,
    records,
    write_record,
)
from lineage.live.deploy_publish import publish, ready_checkout
from lineage.live.deploy_rules import (
    DeployRules,
    breaker,
    day_counts,
    listed_origin_claims,
    load_rules,
    normal_history,
)
from lineage.live.deploy_shell import JOB_SHELL, Shell
from lineage.live.updates import Change, OriginLost, read_changes
from lineage.origins.withheld import REPORT as WITHHELD_REPORT
from lineage.origins.withheld import load_withheld

HELD = 3  # the exit code of a held day: neither a success nor a failed step
REVIEW_DAYS = 7


def deploy_day(
    *,
    state: str,
    export: Path,
    changes_dir: Path,
    paths: DeployPaths | None = None,
    rules: DeployRules | None = None,
    shell: Shell = JOB_SHELL,
    floor: int,
    withheld_report: Path = WITHHELD_REPORT,
) -> DayRecord:
    """Deploy the live export, as of the last day applied to it: pack, gate, check the breaker,
    publish. Returns the day's record: deployed (or already), held, publishing nothing, or
    failed, raised after it is recorded."""
    paths = paths if paths is not None else DeployPaths()
    rules = rules if rules is not None else load_rules()
    day = export_day(export)
    if day is None:
        raise DeployError(f"{export} is a full export; the live track deploys a day applied to it")
    record = read_record(paths, day)
    if record is not None and record.state in ("deployed", "released", "superseded"):
        return record
    if record is not None and record.state == "held":
        # Held on its counts, it waits for `release_day`; held behind a day since released, it
        # goes out as it would have, its gates run again on the checkout as it stands.
        if record.held_on_counts or blocking_day(paths, day) is not None:
            return record
        try:
            return _publish_held(record, paths=paths, rules=rules, shell=shell, reviewed=False)
        except (DeployError, ReleaseError, OSError) as e:
            record.state, record.error = "failed", str(e)
            write_record(paths, record)
            raise DeployError(str(e)) from e
    resuming = record is not None and bool(record.done)
    record = record if resuming and record is not None else DayRecord(day=day, state="failed")
    try:
        pattern = deny_pattern()
        tag = tag_for(state, day)
        candidate = candidate_of(paths, tag) if resuming else None
        if candidate is None:
            candidate = pack(export, state=state, tag=tag, paths=paths, floor=floor)
            record.gates["contract"] = {"ok": True, "files": len(candidate.checksums.files)}
            record.gates["local_paths"] = {"ok": True}
            record.gates["withheld"] = {"report": withheld_digest(withheld_report)}
        record.tag, record.asset = tag, candidate.asset.name
        if not resuming:
            ready_checkout(paths, shell, resuming=False)
            record.gates |= site_gates(
                candidate, paths=paths, shell=shell, max_share=rules.max_share
            )
            reference = previous_candidate(paths, state, day)
            since = tag_day(reference.tag) if reference is not None else None
            logged = sorted({c.as_of for c in read_changes(changes_dir)})
            days = [d for d in logged if d <= day and (since is None or d > since)]
            record.counts = day_counts(
                candidate,
                None if reference is None else reference.checksums,
                read_changes(changes_dir),
                days,
                load_withheld(withheld_report).ids,
            )
            checked = changed_paths(candidate, None if reference is None else reference.checksums)
            named = deny_listed(candidate.asset, checked, pattern)
            record.gates["deny_list"] = deny_gate(named, len(checked))
            decision = breaker(record.counts, normal_history(paths, day), rules.breaker)
            record.baseline = decision.baseline
            reasons = decision.reasons + ([deny_reason(named)] if named else [])
            if reasons:
                record.state, record.held_on_counts, record.reasons = "held", True, reasons
                write_record(paths, record)
                return record
            blocker = blocking_day(paths, day)
            if blocker is not None:
                record.state, record.held_behind = "held", blocker.day
                record.reasons = [f"held behind {blocker.day}, which awaits review"]
                write_record(paths, record)
                return record
        else:
            ready_checkout(paths, shell, resuming=True)
        record.state = "failed"
        record.error = None
        write_record(paths, record)
        publish(record, candidate, paths=paths, shell=shell)
        record.state = "deployed"
        write_record(paths, record)
        _supersede(paths, day)
        return record
    except (DeployError, ReleaseError, OSError) as e:
        record.state, record.error = "failed", str(e)
        write_record(paths, record)
        raise DeployError(str(e)) from e


def _supersede(paths: DeployPaths, day: date) -> None:
    """Days held behind another and published now as part of `day`'s release."""
    for r in records(paths):
        if r.day < day and r.state == "held" and not r.held_on_counts:
            r.state = "superseded"
            write_record(paths, r)


def withheld_digest(path: Path = WITHHELD_REPORT) -> str:
    """The withhold report as it stood when a release was packed, so a release made after the
    maintainer withholds documents can tell it is out of date."""
    return file_sha256(path) if path.exists() else "none"


def exported_withheld(export: Path) -> frozenset[str]:
    """The documents the export's sources page lists as withheld: what its pages were written
    under."""
    doc = json.loads((export / "sources.json").read_text())
    return frozenset(w["id"] for s in doc["sources"] for w in s.get("withheld", []))


def release_day(
    day: date,
    *,
    paths: DeployPaths | None = None,
    rules: DeployRules | None = None,
    shell: Shell = JOB_SHELL,
    repack: bool = False,
    export: Path | None = None,
    state: str = "GA",
    floor: int = 0,
    withheld_report: Path = WITHHELD_REPORT,
) -> DayRecord:
    """Approve a held day after the maintainer's review and publish it, its site gates run
    again. Refused while an earlier day is held on its counts, so no day is published past
    another.

    As packed, its own release goes out, unless the withhold report has changed since it was
    packed: its pages would still list the documents just withheld. With `repack`, after
    withholding (`lineage withhold --doc`, then `publish pages`, `feed` and `promote`), the live
    `export` as it now stands is packed instead, as of its own day, and refused unless its
    sources page lists exactly the report's withheld documents; days held behind the reviewed
    one up to then go out with it."""
    paths = paths if paths is not None else DeployPaths()
    rules = rules if rules is not None else load_rules()
    record = read_record(paths, day)
    if record is None or record.state != "held":
        state_now = "no record" if record is None else record.state
        raise DeployError(f"{day} is not held ({state_now}); only a held day is released")
    blocker = blocking_day(paths, day)
    if blocker is not None:
        raise DeployError(f"{blocker.day} is held before it; release that day first")
    digest = withheld_digest(withheld_report)
    if not repack:
        packed = record.gates.get("withheld", {}).get("report")
        if packed is not None and packed != digest:
            raise DeployError(
                f"the withhold report has changed since {day} was packed; publish pages, feed "
                "and promote, then release it with --repack"
            )
        return _publish_held(record, paths=paths, rules=rules, shell=shell, reviewed=True)
    if export is None:
        raise DeployError("--repack packs the live export; name it")
    through = export_day(export)
    if through is None or through < day:
        raise DeployError(f"the live export is not as of {day} or later")
    later = [r for r in records(paths) if day < r.day <= through and r.held_on_counts
             and r.state == "held"]  # fmt: skip
    if later:
        raise DeployError(f"{later[0].day} is held on its own counts too; review it first")
    if exported_withheld(export) != load_withheld(withheld_report).ids:
        raise DeployError(
            "the live export was not written under the withhold report as it stands; run "
            "`lineage publish pages`, `publish feed` and `publish promote` first"
        )
    candidate = pack(export, state=state, tag=tag_for(state, through), paths=paths, floor=floor)
    record.tag, record.asset = candidate.tag, candidate.asset.name
    record.gates["contract"] = {"ok": True, "files": len(candidate.checksums.files)}
    record.gates["local_paths"] = {"ok": True}
    record.gates["withheld"] = {"report": digest}
    record.reasons = [*record.reasons, f"approved after review; repacked as of {through}"]
    write_record(paths, record)
    out = _publish_held(record, paths=paths, rules=rules, shell=shell, reviewed=True)
    for r in records(paths):
        if day < r.day <= through and r.state == "held":
            r.state = "superseded"
            write_record(paths, r)
    return out


def reject_day(day: date, *, paths: DeployPaths | None = None) -> DayRecord:
    """Reject a held day after the maintainer's review: its release is never published and
    nothing leaves this machine. The live track keeps its data, so the next day's release
    carries it, counted against the rejected day's release, which stays packed as the
    reference (`previous_candidate`); a day held behind it waits no longer, and is published
    with the next day's release. A rejected day does not set what a normal day is."""
    paths = paths if paths is not None else DeployPaths()
    record = read_record(paths, day)
    if record is None or record.state != "held":
        state = "no record" if record is None else record.state
        raise DeployError(f"{day} is not held ({state}); only a held day is rejected")
    record.state = "rejected"
    record.reasons = [*record.reasons, f"rejected after review on {datetime.now(UTC).date()}"]
    write_record(paths, record)
    return record


def _publish_held(
    record: DayRecord, *, paths: DeployPaths, rules: DeployRules, shell: Shell, reviewed: bool
) -> DayRecord:
    """A held day's own release published: released after review, or deployed once the day it
    waited behind is out."""
    candidate = candidate_of(paths, record.tag or "")
    if candidate is None:
        raise DeployError(f"{record.day}'s release is not under {paths.candidates}")
    resuming = bool(record.done)
    ready_checkout(paths, shell, resuming=resuming)
    if not resuming:
        record.gates |= site_gates(candidate, paths=paths, shell=shell, max_share=rules.max_share)
    publish(record, candidate, paths=paths, shell=shell, reviewed=reviewed)
    record.state = "released" if reviewed else "deployed"
    record.error = None
    write_record(paths, record)
    _supersede(paths, record.day)
    return record


# Before session: a candidate for review. During session: the week's origin claims.


def origin_claims_review(
    changes: Sequence[Change],
    *,
    through: date,
    days: int = REVIEW_DAYS,
    describe: Callable[[list[str]], Mapping[str, Mapping[str, Any]]] = lambda _: {},
    documents: Mapping[str, Mapping[str, Any]] | None = None,
    withheld: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """The listed origin claims first listed in the `days` days ending `through`, each with its
    bill, document, both shares, stages, the day first listed, and whether it still stands; a
    withheld document's are not listed, so not reviewed."""
    start = through - timedelta(days=days - 1)
    window = [start + timedelta(days=n) for n in range(days)]
    claims = listed_origin_claims(changes, window, withheld)
    lost = {(c.bill_id, c.doc_id): c.as_of for c in changes if isinstance(c, OriginLost)}
    bills = describe(sorted({c.bill_id for c in claims}))
    docs = documents or {}
    out: list[dict[str, Any]] = []
    for c in sorted(claims, key=lambda c: (c.as_of, c.bill_id, c.doc_id)):
        gone = lost.get((c.bill_id, c.doc_id))
        out.append(
            {
                "bill_id": c.bill_id,
                "bill": dict(bills.get(c.bill_id, {})),
                "doc_id": c.doc_id,
                "document": dict(docs.get(c.doc_id, {})),
                "bill_share": c.bill_share,
                "model_share": c.model_share,
                "stages": list(c.stages),
                "first_listed": c.as_of.isoformat(),
                "still_listed": gone is None or gone < c.as_of,
            }
        )
    return out


def _claim_line(c: Mapping[str, Any]) -> str:
    bill = c["bill"]
    name = f"{bill['state']} {bill['identifier']}" if bill else c["bill_id"]
    doc = c["document"]
    title = f"“{doc['title']}” ({doc['organization']})" if doc else c["doc_id"]
    standing = "" if c["still_listed"] else "; no longer listed"
    return (
        f"- {c['first_listed']}: {name} and {title}: {c['bill_share']:.0%} of the bill's wording, "
        f"{c['model_share']:.0%} of the model's; {' and '.join(c['stages'])}{standing}"
    )


def write_review(
    claims: Sequence[Mapping[str, Any]], *, through: date, out: Path, extra: Sequence[str] = ()
) -> Path:
    """The review as JSON and as text beside it, for the maintainer to read."""
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"through": through.isoformat(), "days": REVIEW_DAYS, "origin_claims": list(claims)}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    lines = [
        *extra,
        f"New origin claims listed in the {REVIEW_DAYS} days through {through}: {len(claims)}",
        *(_claim_line(c) for c in claims),
    ]
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    return out


def weekly_review(
    through: date,
    *,
    changes_dir: Path,
    paths: DeployPaths | None = None,
    describe: Callable[[list[str]], Mapping[str, Mapping[str, Any]]] = lambda _: {},
    documents: Mapping[str, Mapping[str, Any]] | None = None,
    withheld_report: Path = WITHHELD_REPORT,
) -> Path:
    """The week's new origin claims, for the maintainer's weekly review during session ->
    `<deploy>/reviews/<through>.json` and `.txt`."""
    paths = paths if paths is not None else DeployPaths()
    claims = origin_claims_review(
        read_changes(changes_dir),
        through=through,
        describe=describe,
        documents=documents,
        withheld=load_withheld(withheld_report).ids,
    )
    return write_review(claims, through=through, out=paths.reviews / f"{through.isoformat()}.json")


def prepare_candidate(
    day: date,
    *,
    state: str,
    export: Path,
    changes_dir: Path,
    paths: DeployPaths | None = None,
    rules: DeployRules | None = None,
    shell: Shell = JOB_SHELL,
    describe: Callable[[list[str]], Mapping[str, Mapping[str, Any]]] = lambda _: {},
    documents: Mapping[str, Mapping[str, Any]] | None = None,
    floor: int,
    withheld_report: Path = WITHHELD_REPORT,
) -> dict[str, Any]:
    """Before session: the live export packed and gated as a release candidate, with a review
    summary of the files it changes against the pinned release and the week's new origin
    claims -> `<deploy>/candidates/<tag>/review.json` and `.txt`. Publishes nothing."""
    paths = paths if paths is not None else DeployPaths()
    rules = rules if rules is not None else load_rules()
    pattern = deny_pattern()
    as_of = export_day(export) or day
    tag = tag_for(state, as_of)
    candidate = pack(export, state=state, tag=tag, paths=paths, floor=floor)
    ready_checkout(paths, shell, resuming=False)
    gates: dict[str, Any] = {
        "contract": {"ok": True, "files": len(candidate.checksums.files)},
        "local_paths": {"ok": True},
    }
    gates |= site_gates(candidate, paths=paths, shell=shell, max_share=rules.max_share)
    pin, pinned = pinned_checksums(paths, shell)
    diff = changed_files(pinned, candidate.checksums)
    checked = changed_paths(candidate, pinned)
    named = deny_listed(candidate.asset, checked, pattern)
    gates["deny_list"] = deny_gate(named, len(checked))
    claims = origin_claims_review(
        read_changes(changes_dir),
        through=as_of,
        describe=describe,
        documents=documents,
        withheld=load_withheld(withheld_report).ids,
    )
    by_part: dict[str, int] = {}
    for paths_changed in diff.values():
        for p in paths_changed:
            part = p.split("/")[0] if "/" in p else p
            by_part[part] = by_part.get(part, 0) + 1
    summary = [
        f"Release candidate {tag} for review, packed from the live export as of {as_of}; "
        "nothing is published.",
        f"Against the pinned release {pin.tag}: {len(diff['added'])} files added, "
        f"{len(diff['changed'])} changed, {len(diff['removed'])} removed "
        f"({', '.join(f'{k} {v}' for k, v in sorted(by_part.items())) or 'none'}).",
        f"Gates passed: contract ({gates['contract']['files']} files), local paths, copy, size "
        f"({gates['size']['share_of_pages_limit']:.1%} of the Pages limit). The leak check runs "
        "on the commit that pins it.",
        (
            f"Deny list: {deny_reason(named)}."
            if named
            else f"Deny list: none of the {len(checked)} files added or changed names a term."
        ),
        f"To publish it after review: gh release create {tag} {candidate.asset}, then commit its "
        f"pin to {RELEASE_MANIFEST} and run the Pages workflow (DESIGN.md#export).",
        "",
    ]
    out = paths.candidates / tag / "review.json"
    write_review(claims, through=as_of, out=out, extra=summary)
    doc = json.loads(out.read_text()) | {
        "tag": tag,
        "asset": candidate.asset.name,
        "pinned": pin.tag,
        "files": diff,
        "gates": gates,
    }
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def describe_bills(
    tables: Path, *, state: str, site_url: str
) -> Callable[[list[str]], dict[str, dict[str, Any]]]:
    """Each bill a review names, as the feed names it: state, identifier, title and link."""
    from lineage.live.feed import bill_infos  # the feed's naming pulls in the export's stack

    def describe(ids: list[str]) -> dict[str, dict[str, Any]]:
        infos = bill_infos(tables, state=state, site_url=site_url.rstrip("/"), ids=ids)
        return {
            b: {"state": i.state, "identifier": i.identifier, "title": i.title, "link": i.link}
            for b, i in infos.items()
        }

    return describe


def document_names(
    modeldocs_root: Path | None = None, collector_package: str | None = None
) -> dict[str, dict[str, Any]]:
    """Each collected model document's title and the organization whose library holds it."""
    from lineage.origins.collect import PACKAGE, collectors
    from lineage.origins.modeldocs import MODELDOCS_DIR, read_docs

    root = modeldocs_root if modeldocs_root is not None else MODELDOCS_DIR
    return {
        d.id: {"title": d.title, "organization": c.source.organization}
        for c in collectors(collector_package or PACKAGE).values()
        for d in read_docs(c.source, root)
    }

""".github/workflows/data.yml, CI's download of the pinned data release: read-only, run on pull
requests to main and by hand, its actions pinned to the commit SHAs the deploy uses, and the
same fetch, build and copy check as pages.yml, without the deploy."""

import re
from pathlib import Path

WORKFLOWS = Path(__file__).parent.parent / ".github" / "workflows"
DATA = (WORKFLOWS / "data.yml").read_text()
PAGES = (WORKFLOWS / "pages.yml").read_text()
_USES = re.compile(r"^\s*-? *uses: *(\S+?)@(\S+)(.*)$", re.MULTILINE)


def _uses(text: str) -> dict[str, tuple[str, str]]:
    return {m[1]: (m[2], m[3].strip()) for m in _USES.finditer(text)}


def test_it_runs_on_pull_requests_to_main_and_by_hand_with_read_only_permissions() -> None:
    assert re.search(
        r"^on:\n  pull_request:\n    branches: \[main\]\n  workflow_dispatch:\n", DATA, re.M
    )
    assert re.search(r"^permissions:\n  contents: read\n\n", DATA, re.M)
    # No job widens them, and nothing deploys.
    assert DATA.count("permissions:") == 1
    assert "pages: write" not in DATA
    assert "deploy-pages" not in DATA
    assert "upload-pages-artifact" not in DATA


def test_every_action_is_pinned_to_the_sha_pages_uses() -> None:
    uses = _uses(DATA)
    pages = _uses(PAGES)
    assert uses
    for action, (ref, comment) in uses.items():
        assert re.fullmatch(r"[0-9a-f]{40}", ref), action
        assert re.fullmatch(r"# v\d+\.\d+\.\d+", comment), action
        assert pages[action] == (ref, comment), action


def test_it_fetches_builds_and_checks_the_copy_as_the_deploy_does() -> None:
    steps = re.findall(r"^\s*- run: (.+)$", DATA, re.M)
    assert steps == re.findall(r"^\s*- run: (.+)$", PAGES, re.M)
    assert steps.index("scripts/fetch-data.sh") < steps.index("npm run build")
    assert steps[-1] == "npm run test:copy"
    assert re.search(
        r"- run: scripts/fetch-data.sh\n\s+env:\n\s+GH_TOKEN: \$\{\{ github.token \}\}\n", DATA
    )

"""What leaves this machine: every command the deploy runs, through an injectable `Shell`,
and the job's own GitHub token for git and gh (lineage.live.deploy).

See DESIGN.md#updates, DR-13.
"""

import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from lineage.live.deploy_ledger import DEPLOY_DIR, DeployError


@dataclass(frozen=True, slots=True)
class Call:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str] = field(default_factory=dict[str, str])  # added to the job's own


@dataclass(frozen=True, slots=True)
class Result:
    code: int
    output: str = ""


Shell = Callable[[Call], Result]


def subprocess_shell(call: Call) -> Result:
    done = subprocess.run(
        list(call.argv),
        cwd=call.cwd,
        env={**os.environ, **call.env},
        capture_output=True,
        text=True,
        check=False,
    )
    return Result(done.returncode, done.stdout + done.stderr)


# The job's GitHub access (ruled at the deploy review): a fine-grained token scoped to this
# repository, read from the keychain before every git or gh call, never the maintainer's own login.
TOKEN_SERVICE = "legislative-lineage-gh-token"
TokenSource = Callable[[], str | None]


def keychain_token() -> str | None:
    """The job's token from the login keychain, or None when it holds none."""
    found = subprocess.run(
        ["security", "find-generic-password", "-w", "-s", TOKEN_SERVICE],
        capture_output=True,
        text=True,
        check=False,
    )
    token = found.stdout.strip()
    return token if found.returncode == 0 and token else None


def github_shell(shell: Shell, token: TokenSource, config_dir: Path) -> Shell:
    """`shell`, with every git and gh call given the job's token, read afresh, and stopped when
    there is none. gh gets GH_TOKEN and a config directory of the job's own, empty, so it cannot
    read a stored login; git clears every credential helper configured anywhere and uses only
    gh's, which answers with that token. Neither may prompt."""

    def run(call: Call) -> Result:
        if call.argv[:1] not in (("git",), ("gh",)):
            return shell(call)
        value = token()
        if not value:
            raise DeployError(
                f"no GitHub token in the keychain ({TOKEN_SERVICE}); the job uses no other login"
            )
        argv = call.argv
        if argv[0] == "git":
            argv = ("git", "-c", "credential.helper=",
                    "-c", "credential.helper=!gh auth git-credential", *argv[1:])  # fmt: skip
        config_dir.mkdir(parents=True, exist_ok=True)
        env = {
            **call.env,
            "GH_TOKEN": value,
            "GH_CONFIG_DIR": str(config_dir),
            "GH_PROMPT_DISABLED": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
        return shell(Call(argv, call.cwd, env))

    return run


JOB_SHELL = github_shell(subprocess_shell, keychain_token, DEPLOY_DIR / "gh-config")


def run_checked(shell: Shell, argv: Sequence[str], cwd: Path, what: str) -> str:
    result = shell(Call(tuple(argv), cwd))
    if result.code != 0:
        tail = result.output.strip().splitlines()[-1:] or [f"exit code {result.code}"]
        raise DeployError(f"{what} failed: {tail[0][:300]}")
    return result.output

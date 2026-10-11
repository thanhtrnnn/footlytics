"""M2 side: ship a commit to the host, start the suite in tmux, poll, pull the row back.

The run lives in a tmux session on the host, so a dropped ssh session (password
ControlMaster until key login is set up) loses only the polling; `bench resume` picks it up.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .paths import DEFAULT_ROOT
from .results import append_result, write_scoreboard
from .runner import DONE


class SshDown(RuntimeError):
    """ssh itself failed (exit 255): the session dropped or auth is gone."""


class RemoteError(RuntimeError):
    """A command on the host failed."""


_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
#: Fail fast instead of hanging on a dead ControlMaster (laptop sleep, network change).
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3"]


def _safe(*names: str) -> None:
    """Names pasted into the host's shell: suite, run id, sha, dataset. Nothing else gets in."""
    for n in names:
        if not _SAFE.match(n):
            raise ValueError(f"refusing {n!r}: only letters, digits, '.', '_' and '-' are allowed")


def env_fingerprint(pyproject: bytes, host_setup: bytes) -> str:
    """What the host environment was built from; host_setup.sh writes the same hash."""
    return hashlib.sha256(pyproject + host_setup).hexdigest()


def make_run_id(suite: str, sha: str, now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now:%Y%m%d-%H%M%S}-{suite}-{sha[:7]}"


@dataclass
class Remote:
    host: str
    runner: Callable = subprocess.run
    _root: Optional[str] = field(default=None, repr=False)

    def _down(self, why: str) -> SshDown:
        return SshDown(f"ssh to {self.host} failed ({why}). Reconnect with "
                       f"`ssh {self.host} true`, then `footlytics bench resume <run-id> "
                       f"--remote {self.host}`.")

    def sh(self, cmd: str, input: Optional[bytes] = None, check: bool = True,
           timeout: float = 120) -> str:
        try:
            p = self.runner(["ssh", *SSH_OPTS, self.host, cmd], input=input,
                            capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise self._down(f"no answer in {timeout:.0f} s") from None
        out, err = p.stdout.decode(), p.stderr.decode()
        if p.returncode == 255:
            raise self._down(err.strip()[-200:])
        if check and p.returncode != 0:
            raise RemoteError(f"`{cmd}` exited {p.returncode}: {err.strip()[-800:]}")
        return out

    @property
    def root(self) -> str:
        if self._root is None:
            self._root = self.sh(f"echo {DEFAULT_ROOT}").strip()
        return self._root

    def ship(self, sha: str, archive: bytes) -> str:
        _safe(sha)
        d = f"{self.root}/src/{sha}"
        if self.sh(f"test -f {d}/pyproject.toml && echo yes || echo no").strip() == "yes":
            return d
        self.sh(f"rm -rf {d}.tmp && mkdir -p {d}.tmp && tar -x -C {d}.tmp && mv {d}.tmp {d}",
                input=archive, timeout=600)
        self.sh(f"cd {self.root}/src && ls -1t | tail -n +6 | xargs -r rm -rf")
        return d

    def setup(self, sha: str) -> str:
        _safe(sha)
        d = f"{self.root}/src/{sha}"
        return self.sh(f"bash {d}/footlytics/bench/host_setup.sh {self.root} {d}", timeout=1800)

    def ensure_env(self, sha: str, deps_sha: str) -> bool:
        cur = self.sh(f"cat {self.root}/venv/.deps-sha 2>/dev/null || true").strip()
        if cur == deps_sha:
            return False
        self.setup(sha)
        return True

    def _python(self, sha: str, args: str, log: str, session: str) -> None:
        cmd = (f"cd {self.root}/src/{sha} && FOOTLYTICS_BENCH_ROOT={self.root} "
               f"HF_HOME={self.root}/cache/hf HF_HUB_DISABLE_XET=1 "
               f"YOLO_CONFIG_DIR={self.root}/cache/ultralytics "
               f"{self.root}/venv/bin/python -m footlytics.cli "
               f"{args} > {log} 2>&1")
        self.sh(f"mkdir -p {Path(log).parent} && tmux new-session -d -s {session} "
                f"{shlex.quote(cmd)}")

    def start(self, run_id: str, suite: str, sha: str) -> None:
        _safe(run_id, suite, sha)
        self._python(sha, f"bench local --suite {suite} --run-id {run_id} --git-sha {sha} "
                          f"--host {self.host}",
                     f"{self.root}/runs/{run_id}/log.txt", f"footlytics-{run_id}")

    def start_fetch(self, sha: str, what: list[str]) -> str:
        _safe(sha, *what)
        stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        log = f"{self.root}/runs/fetch-{stamp}.log"
        self._python(sha, "bench fetch " + " ".join(what), log, f"footlytics-fetch-{stamp}")
        return log

    def status(self, run_id: str) -> dict:
        _safe(run_id)
        d = f"{self.root}/runs/{run_id}"
        raw = self.sh(f"cat {d}/status.json 2>/dev/null || true").strip()
        st = json.loads(raw) if raw else {"state": "starting"}
        if st["state"] in DONE:
            return st
        alive = self.sh(f"tmux has-session -t footlytics-{run_id} 2>/dev/null "
                        "&& echo yes || echo no").strip() == "yes"
        if not alive:
            return {"state": "crashed",
                    "log_tail": self.sh(f"tail -n 30 {d}/log.txt 2>/dev/null || true")}
        return st

    def result(self, run_id: str) -> dict:
        _safe(run_id)
        return json.loads(self.sh(f"cat {self.root}/runs/{run_id}/result.json"))


def follow(remote: Remote, run_id: str, results_dir: Path, interval: float = 30,
           echo: Callable = print, sleep: Callable = time.sleep) -> dict:
    """Poll until the run ends, then append its row and regenerate the scoreboard."""
    last = None
    while True:
        st = remote.status(run_id)
        if st["state"] == "crashed":
            raise RemoteError(f"run {run_id} died without a result:\n{st['log_tail']}")
        shown = (st["state"], st.get("task"))
        if shown != last:
            echo(f"[{run_id}] {st['state']}" + (f": {st['task']}" if st.get("task") else ""))
            last = shown
        if st["state"] in DONE:
            row = remote.result(run_id)
            results_dir = Path(results_dir)
            if append_result(results_dir / "results.jsonl", row):
                echo(f"appended to {results_dir / 'results.jsonl'}")
            write_scoreboard(results_dir)
            return row
        sleep(interval)

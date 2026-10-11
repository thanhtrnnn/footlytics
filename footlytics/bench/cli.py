"""`footlytics bench`: run suites on the benchmark host and keep the scoreboard."""
from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from typing import Optional

import typer

bench_app = typer.Typer(help="Benchmark: our output scored by the official SoccerNet evaluators.")
REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def _clean_sha() -> str:
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", ".",
                 ":!bench/results").decode().strip()
    if dirty:
        raise typer.BadParameter("uncommitted changes: a row must name the code it scored. "
                                 f"Commit first.\n{dirty}")
    return _git("rev-parse", "HEAD").decode().strip()


def _suite_exists(suite: str) -> None:
    from .runner import SUITES_DIR

    if not (SUITES_DIR / f"{suite}.yaml").exists():
        known = sorted(p.stem for p in SUITES_DIR.glob("*.yaml"))
        raise typer.BadParameter(f"no suite {suite!r}; one of {known}")


def _fetchable(what: list[str]) -> None:
    from .datasets import FETCHABLE

    unknown = [w for w in what if w not in FETCHABLE]
    if unknown:
        raise typer.BadParameter(f"unknown dataset(s) {unknown}; one of {sorted(FETCHABLE)}")


def _remote(host: str):
    from .remote import Remote

    return Remote(host)


def _guard(fn):
    """Turn a dropped ssh session into its recovery instructions and exit code 2."""
    from .remote import RemoteError, SshDown

    try:
        return fn()
    except SshDown as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2)
    except (RemoteError, ValueError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)


@bench_app.command("run")
def run(suite: str = typer.Option("harness"),
        remote: Optional[str] = typer.Option(None, help="ssh alias of the host, e.g. iec4090_2"),
        interval: int = typer.Option(30, help="seconds between polls")) -> None:
    """Run a suite (on the host with --remote) and append its row to bench/results."""
    from .remote import env_fingerprint, follow, make_run_id
    from .results import RESULTS_DIR, append_result, write_scoreboard

    _suite_exists(suite)
    sha = _clean_sha()
    run_id = make_run_id(suite, sha)
    if remote is None:
        from .paths import BenchPaths
        from .runner import run_suite

        row = run_suite(suite, run_id, BenchPaths.from_env(), sha, socket.gethostname())
        append_result(RESULTS_DIR / "results.jsonl", row)
        write_scoreboard(RESULTS_DIR)
        typer.echo(f"{run_id}: {row['status']}")
        return

    def go():
        r = _remote(remote)
        r.ship(sha, _git("archive", "--format=tar", sha))
        deps = env_fingerprint(_git("show", f"{sha}:pyproject.toml"),
                               _git("show", f"{sha}:footlytics/bench/host_setup.sh"))
        if r.ensure_env(sha, deps):
            typer.echo("host environment (re)built")
        r.start(run_id, suite, sha)
        typer.echo(f"started {run_id} on {remote}; if ssh drops: "
                   f"`footlytics bench resume {run_id} --remote {remote}`")
        row = follow(r, run_id, RESULTS_DIR, interval, echo=typer.echo)
        typer.echo(f"{run_id}: {row['status']}")
    _guard(go)


@bench_app.command("resume")
def resume(run_id: str, remote: str = typer.Option(...),
           interval: int = typer.Option(30)) -> None:
    """Re-attach to a run after an ssh drop: poll, then pull its row."""
    from .remote import follow
    from .results import RESULTS_DIR

    _guard(lambda: typer.echo(follow(_remote(remote), run_id, RESULTS_DIR, interval,
                                     echo=typer.echo)["status"]))


@bench_app.command("setup")
def setup(remote: str = typer.Option(...)) -> None:
    """(Re)build the host: uv, pipeline venv on the system torch, evaluator venv, evaluators."""
    sha = _clean_sha()

    def go():
        r = _remote(remote)
        r.ship(sha, _git("archive", "--format=tar", sha))
        typer.echo(r.setup(sha))
    _guard(go)


@bench_app.command("fetch")
def fetch(what: list[str] = typer.Argument(
              ..., help="gsr-valid gsr-test calib-valid calib-test baseline"),
          remote: Optional[str] = typer.Option(
              None, help="start the download on this host in tmux")) -> None:
    """Download pinned datasets (on the host; with --remote, start that in tmux and return)."""
    _fetchable(what)
    if remote:
        sha = _clean_sha()

        def go():
            r = _remote(remote)
            r.ship(sha, _git("archive", "--format=tar", sha))
            typer.echo(f"fetching in tmux on {remote}; log: {r.start_fetch(sha, what)}")
        _guard(go)
        return
    from .datasets import fetch as fetch_one
    from .paths import BenchPaths

    paths = BenchPaths.from_env()
    for w in what:
        typer.echo(f"{w}: {fetch_one(paths, w)}")


@bench_app.command("local", hidden=True)
def local(suite: str = typer.Option(...), run_id: str = typer.Option(...),
          git_sha: str = typer.Option(...), host: str = typer.Option(...)) -> None:
    """Host side of `bench run --remote`: run the suite here and leave result.json."""
    from .paths import BenchPaths
    from .runner import run_suite

    row = run_suite(suite, run_id, BenchPaths.from_env(), git_sha, host)
    typer.echo(f"{run_id}: {row['status']}")


@bench_app.command("scoreboard")
def scoreboard() -> None:
    """Regenerate bench/results/SCOREBOARD.md from results.jsonl."""
    from .results import RESULTS_DIR, write_scoreboard

    typer.echo(str(write_scoreboard(RESULTS_DIR)))

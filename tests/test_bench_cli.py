"""`footlytics bench` CLI: bad input is refused on the M2, ssh drops exit 2."""
from typer.testing import CliRunner

from footlytics.bench import cli


def no_remote(host):
    raise AssertionError("contacted the host")


def test_unknown_suite_is_refused_before_contacting_the_host(monkeypatch):
    monkeypatch.setattr(cli, "_clean_sha", lambda: "abc")
    monkeypatch.setattr(cli, "_remote", no_remote)
    r = CliRunner().invoke(cli.bench_app, ["run", "--suite", "nope", "--remote", "h"])
    assert r.exit_code != 0 and "nope" in r.output


def test_unknown_fetch_target_is_refused(monkeypatch):
    monkeypatch.setattr(cli, "_clean_sha", lambda: "abc")
    monkeypatch.setattr(cli, "_remote", no_remote)
    r = CliRunner().invoke(cli.bench_app, ["fetch", "gsr-vlaid", "--remote", "h"])
    assert r.exit_code != 0 and "gsr-vlaid" in r.output


def test_ssh_drop_exits_2_with_recovery(monkeypatch):
    from footlytics.bench.remote import SshDown

    class Down:
        def __init__(self, host):
            pass

        def status(self, run_id):
            raise SshDown("ssh to h failed. Reconnect with `ssh h true`")
    monkeypatch.setattr(cli, "_remote", Down)
    r = CliRunner().invoke(cli.bench_app, ["resume", "20261012-100000-harness-abc1234",
                                           "--remote", "h"])
    assert r.exit_code == 2 and "ssh h true" in r.output

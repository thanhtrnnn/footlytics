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


def test_setup_ships_the_soccermaster_checkout_without_caches(monkeypatch, tmp_path):
    import io
    import tarfile
    code = tmp_path / "SoccerMaster"
    (code / "models" / "__pycache__").mkdir(parents=True)
    (code / "models" / "x.py").write_text("x = 1")
    (code / "models" / "__pycache__" / "x.pyc").write_bytes(b"0")
    calls = []

    class R:
        def __init__(self, host):
            pass

        def ship(self, sha, archive):
            calls.append("ship")

        def ship_dir(self, name, archive):
            names = tarfile.open(fileobj=io.BytesIO(archive)).getnames()
            calls.append((name, sorted(n for n in names if n.endswith((".py", ".pyc")))))

        def setup(self, sha):
            calls.append("setup")
            return "host ready"
    monkeypatch.setattr(cli, "_clean_sha", lambda: "abc")
    monkeypatch.setattr(cli, "_git", lambda *a: b"tar")
    monkeypatch.setattr(cli, "_remote", R)
    r = CliRunner().invoke(cli.bench_app, ["setup", "--remote", "h", "--soccermaster-code", str(code)])
    assert r.exit_code == 0, r.output
    assert calls == ["ship", ("SoccerMaster", ["./models/x.py"]), "setup"]

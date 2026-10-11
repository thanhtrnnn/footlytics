"""Remote driver: ssh failures are explicit, shipping is idempotent, resume never double-appends."""
import json
import subprocess

import pytest

from footlytics.bench import remote as rm
from footlytics.bench.results import load_results


class FakeSsh:
    """Answers ssh commands from a script of (substring, returncode, stdout)."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def __call__(self, argv, input=None, capture_output=True):
        cmd = argv[-1]
        self.calls.append(cmd)
        for needle, code, out in self.script:
            if needle in cmd:
                return subprocess.CompletedProcess(argv, code, out.encode(), b"err")
        return subprocess.CompletedProcess(argv, 0, b"", b"")


ROOT = [("echo ~/footlytics-bench", 0, "/home/iec/footlytics-bench\n")]


def test_dropped_session_says_how_to_recover():
    r = rm.Remote("iec4090_2", runner=FakeSsh([("echo ~", 255, "")]))
    with pytest.raises(rm.SshDown, match="ssh iec4090_2 true"):
        r.root


def test_ship_skips_a_commit_already_there():
    ssh = FakeSsh(ROOT + [("test -f", 0, "yes\n")])
    rm.Remote("h", runner=ssh).ship("abc", b"tar")
    assert not any("tar -x" in c for c in ssh.calls)


def test_ship_uploads_and_prunes_old_sources():
    ssh = FakeSsh(ROOT + [("test -f", 0, "no\n")])
    d = rm.Remote("h", runner=ssh).ship("abc", b"tar")
    assert d == "/home/iec/footlytics-bench/src/abc"
    assert any("tar -x -C /home/iec/footlytics-bench/src/abc.tmp" in c for c in ssh.calls)
    assert any("tail -n +6" in c for c in ssh.calls)


def test_start_runs_in_a_named_tmux_session():
    ssh = FakeSsh(ROOT)
    rm.Remote("iec4090_2", runner=ssh).start("20261012-100000-harness-abc1234", "harness", "abc")
    cmd = next(c for c in ssh.calls if "tmux new-session" in c)
    assert "-s footlytics-20261012-100000-harness-abc1234" in cmd
    assert "bench local --suite harness" in cmd and "FOOTLYTICS_BENCH_ROOT=" in cmd
    # shared account: ultralytics must not touch ~/.config, HF must not touch ~/.cache
    assert "YOLO_CONFIG_DIR=/home/iec/footlytics-bench/cache/ultralytics" in cmd
    assert "HF_HOME=/home/iec/footlytics-bench/cache/hf" in cmd


def test_status_reports_a_dead_session_with_its_log():
    ssh = FakeSsh(ROOT + [("cat /home/iec/footlytics-bench/runs/r1/status.json", 0,
                           '{"state": "running"}'),
                          ("tmux has-session", 0, "no\n"), ("tail -n 30", 0, "Killed\n")])
    st = rm.Remote("h", runner=ssh).status("r1")
    assert st["state"] == "crashed" and "Killed" in st["log_tail"]


def test_follow_appends_once_even_when_resumed(tmp_path):
    row = {"run_id": "r1", "suite": "harness", "git_sha": "abc", "host": "h",
           "finished_at": "2026-10-12T10:00:00Z", "status": "complete", "tasks": {}}
    ssh = FakeSsh(ROOT + [("status.json", 0, '{"state": "complete"}'),
                          ("result.json", 0, json.dumps(row))])
    r = rm.Remote("h", runner=ssh)
    for _ in range(2):
        rm.follow(r, "r1", tmp_path, interval=0, echo=lambda *a: None, sleep=lambda s: None)
    assert [x["run_id"] for x in load_results(tmp_path / "results.jsonl")] == ["r1"]
    assert (tmp_path / "SCOREBOARD.md").exists()


def test_run_id_is_sortable_and_names_suite_and_commit():
    from datetime import datetime, timezone
    rid = rm.make_run_id("harness", "abcdef123456", datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc))
    assert rid == "20261012-100000-harness-abcdef1"


def test_fetch_runs_in_tmux_without_the_xet_chunk_cache():
    # hf_xet keeps a second copy of every chunk (up to 10 GB) next to the download: on a
    # shared disk with 40 GB free that is not affordable, so downloads go over plain HTTP
    ssh = FakeSsh(ROOT)
    log = rm.Remote("iec4090_2", runner=ssh).start_fetch("abc", ["gsr-valid", "gsr-test"])
    cmd = next(c for c in ssh.calls if "tmux new-session" in c)
    assert "HF_HUB_DISABLE_XET=1" in cmd and "bench fetch gsr-valid gsr-test" in cmd
    assert log.startswith("/home/iec/footlytics-bench/runs/fetch-")

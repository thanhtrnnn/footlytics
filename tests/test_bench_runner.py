"""Suite loop: failure isolation, checks, status, preflight."""
import json
import zipfile

import pytest

from footlytics.bench import runner
from footlytics.bench.evaluators import GsrScores
from footlytics.bench.paths import BenchPaths
from tests.bench_fixtures import make_labels


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "contention", lambda: {"load1": 0.0})
    return BenchPaths(tmp_path)


def spec(*tasks):
    return {"name": "t", "tasks": list(tasks)}


def test_crashing_task_makes_run_partial_not_dead(paths, monkeypatch):
    monkeypatch.setitem(runner.KINDS, "ok", lambda task, ctx: {"aggregate": {"m": 1.0}})
    monkeypatch.setitem(runner.KINDS, "boom", lambda task, ctx: 1 / 0)
    row = runner.run_suite("t", "r1", paths, "sha", "h", suite_spec=spec(
        {"id": "a", "kind": "ok", "check": {"metric": "m", "min": 0.5}},
        {"id": "b", "kind": "boom"}))
    assert row["status"] == "partial"
    assert row["tasks"]["a"]["check"]["passed"] is True
    assert "ZeroDivisionError" in row["tasks"]["b"]["error"]
    assert json.loads((paths.run_dir("r1") / "status.json").read_text())["state"] == "partial"
    assert json.loads((paths.run_dir("r1") / "result.json").read_text())["run_id"] == "r1"


def test_checks_min_and_expected():
    assert runner.evaluate_check({"metric": "m", "min": 99}, {"m": 98.0})["passed"] is False
    c = runner.evaluate_check({"metric": "m", "expected": 22.26, "tolerance": 1.0}, {"m": 22.9})
    assert c["passed"] is True and c["delta"] == pytest.approx(0.64)
    assert runner.evaluate_check({"metric": "m", "min": 1}, {})["passed"] is False


def test_preflight_fails_when_torch_changed(paths, monkeypatch):
    paths.venv.mkdir(parents=True)
    paths.torch_pin.write_text("2.7.1+cu128\n")
    monkeypatch.setattr(runner, "torch_info", lambda: {"torch": "2.14.1", "cuda": True})
    monkeypatch.setitem(runner.KINDS, "ok", lambda task, ctx: pytest.fail("ran a task"))
    row = runner.run_suite("t", "r2", paths, "sha", "h", suite_spec=spec({"id": "a", "kind": "ok"}))
    assert row["status"] == "failed" and "torch changed" in row["error"]


def write_gt(paths, split, seqs, bad=()):
    for s in seqs:
        d = paths.gsr_labels / split / s
        d.mkdir(parents=True)
        labels = make_labels(seq=s, version="1.2" if s in bad else "1.3")
        (d / "Labels-GameState.json").write_text(json.dumps(labels))


def fake_scores(calls):
    def run_gsr(python, te, gt_root, trackers_root, split, tracker, seqs, out_dir, use_jersey):
        calls.append((split, tracker, tuple(seqs), use_jersey))
        for s in seqs:
            assert (trackers_root / f"SoccerNetGS-{split}" / tracker / f"{s}.json").exists()
        v = 99.9 if not use_jersey else 99.8
        return GsrScores({s: {"hota": v, "deta": v, "assa": v} for s in seqs},
                         {"hota": v, "deta": v, "assa": v})
    return run_gsr


def test_gt_roundtrip_isolates_a_bad_clip(paths, monkeypatch):
    write_gt(paths, "valid", ["SNGS-990", "SNGS-991"], bad={"SNGS-991"})
    calls = []
    monkeypatch.setattr(runner, "run_gsr", fake_scores(calls))
    row = runner.run_suite("t", "r3", paths, "sha", "h", suite_spec=spec(
        {"id": "rt", "kind": "gsr_gt_roundtrip", "split": "valid", "clips": "all",
         "check": {"metric": "hota_jersey_off", "min": 99.0}}))
    t = row["tasks"]["rt"]
    assert row["status"] == "partial" and t["status"] == "partial"
    assert "1.3" in t["failed"]["SNGS-991"]
    assert t["aggregate"]["hota_jersey_off"] == 99.9 and t["aggregate"]["hota_jersey_on"] == 99.8
    assert t["clips"]["SNGS-990"]["hota_jersey_off"] == 99.9
    assert {c[3] for c in calls} == {False, True} and all(c[2] == ("SNGS-990",) for c in calls)


def test_rescore_reads_tracker_files_from_the_zip(paths, monkeypatch):
    write_gt(paths, "test", ["SNGS-990", "SNGS-991"])
    paths.baseline_zip.parent.mkdir(parents=True)
    with zipfile.ZipFile(paths.baseline_zip, "w") as z:
        z.writestr("tracklab/SNGS-990.json", json.dumps({"predictions": []}))
    calls = []
    monkeypatch.setattr(runner, "run_gsr", fake_scores(calls))
    row = runner.run_suite("t", "r4", paths, "sha", "h", suite_spec=spec(
        {"id": "bl", "kind": "gsr_rescore", "split": "test", "clips": "all",
         "predictions": "baseline/SoccerNetGS-test.zip", "tracker": "tracklab"}))
    t = row["tasks"]["bl"]
    assert t["failed"] == {"SNGS-991": "no predictions in baseline/SoccerNetGS-test.zip"}
    assert calls[0][1] == "tracklab"


def test_prune_keeps_the_last_runs_work(paths):
    for i in range(7):
        w = paths.run_dir(f"2026101{i}-000000-t-sha") / "work"
        w.mkdir(parents=True)
        (w.parent / "result.json").write_text("{}")
    runner.prune_runs(paths, keep=5)
    kept = sorted(p.parent.name for p in paths.runs.glob("*/work"))
    assert len(kept) == 5 and kept[0].startswith("20261012")
    assert len(list(paths.runs.glob("*/result.json"))) == 7


def test_harness_suite_is_valid():
    s = runner.load_suite("harness")
    assert {t["kind"] for t in s["tasks"]} <= set(runner.KINDS)


def test_rescore_isolates_bad_ground_truth_and_bad_predictions(paths, monkeypatch):
    write_gt(paths, "test", ["SNGS-990", "SNGS-991", "SNGS-992"], bad={"SNGS-992"})
    paths.baseline_zip.parent.mkdir(parents=True)
    with zipfile.ZipFile(paths.baseline_zip, "w") as z:
        z.writestr("tracklab/SNGS-990.json", json.dumps({"predictions": []}))
        z.writestr("tracklab/SNGS-991.json", "{not json")
        z.writestr("tracklab/SNGS-992.json", json.dumps({"predictions": []}))
    calls = []
    monkeypatch.setattr(runner, "run_gsr", fake_scores(calls))
    row = runner.run_suite("t", "r5", paths, "sha", "h", suite_spec=spec(
        {"id": "bl", "kind": "gsr_rescore", "split": "test", "clips": "all",
         "predictions": "baseline/SoccerNetGS-test.zip", "tracker": "tracklab"}))
    t = row["tasks"]["bl"]
    assert t["status"] == "partial"
    assert "JSON" in t["failed"]["SNGS-991"] and "1.3" in t["failed"]["SNGS-992"]
    assert all(c[2] == ("SNGS-990",) for c in calls)


def test_run_refuses_to_start_below_the_disk_floor(paths, monkeypatch):
    from footlytics.bench import host
    monkeypatch.setattr(host, "free_disk_bytes", lambda p: 2_000_000_000)
    monkeypatch.setitem(runner.KINDS, "ok", lambda task, ctx: pytest.fail("ran a task"))
    row = runner.run_suite("t", "r6", paths, "sha", "h", suite_spec=spec({"id": "a", "kind": "ok"}))
    assert row["status"] == "failed" and "free" in row["error"]


def test_preflight_records_the_evaluator_commit(paths, monkeypatch):
    import subprocess
    repo = paths.trackeval
    repo.mkdir(parents=True)
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "x"], check=True)
    sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    monkeypatch.setitem(runner.KINDS, "ok", lambda task, ctx: {"aggregate": {}})
    row = runner.run_suite("t", "r7", paths, "sha", "h", suite_spec=spec({"id": "a", "kind": "ok"}))
    assert row["env"]["evaluators"]["sn-trackeval"] == sha

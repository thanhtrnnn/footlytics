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


# ---------------------------------------------------------------- calibration kinds
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from footlytics.bench.evaluators import CalibScores  # noqa: E402
from footlytics.geometry.calib_nn import CameraFit  # noqa: E402
from footlytics.geometry.sn_camera import ground_homography, make_camera  # noqa: E402

CAM = make_camera(0, 80, 0, 750, (0, 55, -15), 960, 540)


def jpg(w=96, h=54):
    return cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))[1].tobytes()


class FakeCal:
    def decode(self, frames, **kw):
        return [({1: {"x": 0.1, "y": 0.1, "p": 1}}, {}) for _ in frames]

    def fit(self, frames, which=None, **kw):
        H = np.linalg.inv(ground_homography(CAM))
        return {i: CameraFit(CAM, 3.0, H / H[2, 2]) for i in (which or range(len(frames)))}


@pytest.fixture
def calib(paths, monkeypatch):
    monkeypatch.setattr(runner, "_calibrator", lambda p: FakeCal())
    monkeypatch.setattr(runner, "require_vram", lambda mib: None)
    evals = []

    def fake_eval(python, sc, gt_root, pred_root, split, out_dir, **kw):
        gts = sorted(p.stem for p in (gt_root / split).glob("*.json"))
        preds = sorted(p.stem for p in (pred_root / split).glob("camera_*.json"))
        evals.append((gts, preds))
        return CalibScores(score=50.0, completeness=100.0 * len(preds) / max(len(gts), 1),
                           jac=50.0, precision=60.0, recall=70.0)
    monkeypatch.setattr(runner, "run_sncalib", fake_eval)
    return evals


def write_calib_zip(paths, frames, match=True):
    paths.calib_zip("valid").parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(paths.calib_zip("valid"), "w") as z:
        for f in frames:
            z.writestr(f"valid/{f}.jpg", jpg())
            z.writestr(f"valid/{f}.json", json.dumps({"Middle line": [{"x": 0.5, "y": 0.2},
                                                                      {"x": 0.5, "y": 0.8}]}))
        if match:
            z.writestr("valid/match_info.json", json.dumps(
                {f"{f}.jpg": {"league": "l", "season": "s", "match": "m", "date": "d"}
                 for f in frames}))


def test_sncalib_scores_only_the_selected_frames(paths, calib, monkeypatch):
    write_calib_zip(paths, ["00000", "00001", "00002"])

    def solve(code, jobs, workers=1, **kw):
        H = np.linalg.inv(ground_homography(CAM))
        fit = CameraFit(CAM, 2.0, H / H[2, 2])
        return [(fit, None), (None, "LinAlgError: singular")]
    monkeypatch.setattr(runner, "solve_many", solve)
    row = runner.run_suite("t", "c1", paths, "sha", "h", suite_spec=spec(
        {"id": "sc", "kind": "sncalib", "split": "valid", "frames": ["00000", "00002"]}))
    t = row["tasks"]["sc"]
    assert t["status"] == "complete", t
    gts, preds = calib[0]
    assert gts == ["00000", "00002"] and preds == ["camera_00000"]    # no match_info, no 00001
    assert t["aggregate"]["solver_errors"] == 1 and t["aggregate"]["frames"] == 2
    assert t["aggregate"]["score"] == 50.0 and t["leakage"]["status"] == "unverified"
    assert t["clips"]["00000"]["calibrated"] == 1 and t["clips"]["00002"]["calibrated"] == 0


def test_sncalib_frame_stride_with_exclusions(paths, calib, monkeypatch):
    write_calib_zip(paths, [f"{i:05d}" for i in range(40)])
    monkeypatch.setattr(runner, "solve_many", lambda code, jobs, workers=1, **kw:
                        [(None, None)] * len(jobs))
    row = runner.run_suite("t", "c2", paths, "sha", "h", suite_spec=spec(
        {"id": "sc", "kind": "sncalib", "split": "valid",
         "frames": {"every": 16, "offset": 12, "exclude": ["00028"]}}))
    assert calib[0][0] == ["00012"]


def test_sncalib_leakage_subset_is_scored_when_games_are_listed(paths, calib, monkeypatch):
    write_calib_zip(paths, ["00000", "00001"])
    monkeypatch.setattr(runner, "solve_many", lambda code, jobs, workers=1, **kw:
                        [(None, None)] * len(jobs))
    row = runner.run_suite("t", "c3", paths, "sha", "h", suite_spec=spec(
        {"id": "sc", "kind": "sncalib", "split": "valid", "frames": "all",
         "leakage_games": ["l|s|m|d"]}))
    lk = row["tasks"]["sc"]["leakage"]
    assert lk["status"] == "checked" and lk["frames_kept"] == 0


def test_sncalib_writes_cameras_at_the_evaluation_resolution(paths, calib, monkeypatch):
    write_calib_zip(paths, ["00000"])
    big = make_camera(0, 80, 0, 1500, (0, 55, -15), 1920, 1080)
    H = np.linalg.inv(ground_homography(big))
    monkeypatch.setattr(runner, "solve_many", lambda code, jobs, workers=1, **kw:
                        [(CameraFit(big, 2.0, H / H[2, 2]), None)] * len(jobs))
    seen = {}
    real = runner.run_sncalib

    def spy(python, sc, gt_root, pred_root, split, out_dir, **kw):
        seen.update(json.loads((pred_root / split / "camera_00000.json").read_text()))
        return real(python, sc, gt_root, pred_root, split, out_dir, **kw)
    monkeypatch.setattr(runner, "run_sncalib", spy)
    runner.run_suite("t", "c3b", paths, "sha", "h", suite_spec=spec(
        {"id": "sc", "kind": "sncalib", "split": "valid", "frames": "all"}))
    assert seen["principal_point"] == [480.0, 270.0]


def write_gsr_clip(paths, seq="SNGS-990", n=3, n_images=None):
    labels = make_labels(seq=seq, n_frames=n)
    for img in labels["images"]:
        img["has_labeled_pitch"] = img["is_labeled"]
    for a in labels["annotations"]:
        if a["supercategory"] == "pitch":
            a["lines"] = {"Middle line": [{"x": 0.5, "y": 0.2}, {"x": 0.5, "y": 0.8}]}
    d = paths.gsr_labels / "valid" / seq
    d.mkdir(parents=True, exist_ok=True)
    (d / "Labels-GameState.json").write_text(json.dumps(labels))
    paths.gsr_zip("valid").parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(paths.gsr_zip("valid"), "a") as z:
        for f in range(n if n_images is None else n_images):
            z.writestr(f"{seq}/img1/{f + 1:06d}.jpg", jpg(192, 108))


def test_gsr_calibration_scores_labelled_frames_per_clip(paths, calib):
    write_gsr_clip(paths)
    row = runner.run_suite("t", "c4", paths, "sha", "h", suite_spec=spec(
        {"id": "gc", "kind": "gsr_calibration", "split": "valid", "clips": ["SNGS-990"],
         "every": 1}))
    t = row["tasks"]["gc"]
    assert t["status"] == "complete", t
    c = t["clips"]["SNGS-990"]
    assert c["calibrated_share"] == 1.0 and c["keyframes"] == 3 and c["keyframes_accepted"] == 3
    assert c["solver_errors"] == 0
    assert len(calib) == 1
    gts, preds = calib[0]
    assert gts == ["SNGS-990-000001", "SNGS-990-000002", "SNGS-990-000003"]
    assert preds == ["camera_SNGS-990-000001", "camera_SNGS-990-000002",
                     "camera_SNGS-990-000003"]
    assert c["labelled"] == 3 and c["predicted"] == 3
    assert "seconds_per_frame" in t["aggregate"]


def test_gsr_calibration_counts_labelled_frames_without_an_image_as_missed(paths, calib):
    write_gsr_clip(paths, n=3, n_images=2)
    row = runner.run_suite("t", "c4d", paths, "sha", "h", suite_spec=spec(
        {"id": "gc", "kind": "gsr_calibration", "split": "valid", "clips": ["SNGS-990"],
         "every": 1}))
    t = row["tasks"]["gc"]
    assert t["status"] == "complete", t
    assert t["clips"]["SNGS-990"]["calibrated_share"] == round(2 / 3, 4)
    names = ["SNGS-990-000001", "SNGS-990-000002", "SNGS-990-000003"]
    assert len(calib) == 1
    gts, preds = calib[0]
    assert gts == names and preds == [f"camera_{n}" for n in names[:2]]
    assert t["clips"]["SNGS-990"]["completeness"] == pytest.approx(100.0 * 2 / 3)
    assert t["aggregate"]["completeness"] == pytest.approx(100.0 * 2 / 3)


def test_gsr_calibration_writes_cameras_at_the_evaluation_resolution(paths, calib, monkeypatch):
    write_gsr_clip(paths)
    big = make_camera(0, 80, 0, 1500, (0, 55, -15), 1920, 1080)
    H = np.linalg.inv(ground_homography(big))

    class BigCal(FakeCal):
        def fit(self, frames, which=None, **kw):
            return {i: CameraFit(big, 3.0, H / H[2, 2]) for i in (which or range(len(frames)))}
    monkeypatch.setattr(runner, "_calibrator", lambda p: BigCal())
    seen = {}
    real = runner.run_sncalib

    def spy(python, sc, gt_root, pred_root, split, out_dir, **kw):
        for p in (pred_root / split).glob("camera_*.json"):
            seen[p.name] = json.loads(p.read_text())
        return real(python, sc, gt_root, pred_root, split, out_dir, **kw)
    monkeypatch.setattr(runner, "run_sncalib", spy)
    row = runner.run_suite("t", "c4b", paths, "sha", "h", suite_spec=spec(
        {"id": "gc", "kind": "gsr_calibration", "split": "valid", "clips": ["SNGS-990"],
         "every": 1}))
    assert row["tasks"]["gc"]["status"] == "complete", row["tasks"]["gc"]
    assert seen and all(c["principal_point"] == [480.0, 270.0] for c in seen.values())


def test_gsr_calibration_reports_solver_errors(paths, calib, monkeypatch):
    write_gsr_clip(paths)

    class ErrCal(FakeCal):
        def fit(self, frames, which=None, **kw):
            self.last_errors = {0: "ZeroDivisionError: x"}
            out = super().fit(frames, which, **kw)
            out[0] = None
            return out
    monkeypatch.setattr(runner, "_calibrator", lambda p: ErrCal())
    row = runner.run_suite("t", "c4c", paths, "sha", "h", suite_spec=spec(
        {"id": "gc", "kind": "gsr_calibration", "split": "valid", "clips": ["SNGS-990"],
         "every": 1}))
    t = row["tasks"]["gc"]
    assert t["status"] == "complete", t
    assert t["clips"]["SNGS-990"]["solver_errors"] > 0
    assert t["aggregate"]["solver_errors"] == t["clips"]["SNGS-990"]["solver_errors"]


def test_calibration_suites_are_valid():
    for name in ("quick", "milestone", "calib-tune"):
        s = runner.load_suite(name)
        assert {t["kind"] for t in s["tasks"]} <= set(runner.KINDS)
    q = runner.load_suite("quick")
    sc = next(t for t in q["tasks"] if t["kind"] == "sncalib")
    assert len(sc["frames"]) == 60 and len(set(sc["frames"])) == 60


# ---------------------------------------------------------------- review fixes


def test_gsr_calibration_survives_a_clip_whose_scoring_fails(paths, calib, monkeypatch):
    write_gsr_clip(paths, seq="SNGS-990")
    write_gsr_clip(paths, seq="SNGS-991")
    good = runner.run_sncalib

    def flaky(python, sc, gt_root, pred_root, split, out_dir, **kw):
        if "SNGS-991" in str(gt_root):
            raise KeyError("Penalty area left")
        return good(python, sc, gt_root, pred_root, split, out_dir, **kw)
    monkeypatch.setattr(runner, "run_sncalib", flaky)
    row = runner.run_suite("t", "c5", paths, "sha", "h", suite_spec=spec(
        {"id": "gc", "kind": "gsr_calibration", "split": "valid",
         "clips": ["SNGS-990", "SNGS-991"], "every": 1}))
    t = row["tasks"]["gc"]
    assert t["status"] == "partial", t
    assert t["clips"]["SNGS-990"]["score"] == 50.0 and "SNGS-991" not in t["clips"]
    assert "KeyError" in t["failed"]["SNGS-991"]
    # the aggregate is the surviving clip alone
    assert t["aggregate"]["score"] == 50.0 and t["aggregate"]["completeness"] == 100.0
    assert t["aggregate"]["jac5"] == 50.0


def slow_eval(monkeypatch, seconds=0.3):
    import time
    inner = runner.run_sncalib

    def slow(*a, **kw):
        time.sleep(seconds)
        return inner(*a, **kw)
    monkeypatch.setattr(runner, "run_sncalib", slow)


def test_sncalib_seconds_per_frame_excludes_the_evaluator(paths, calib, monkeypatch):
    write_calib_zip(paths, ["00000", "00001", "00002"])
    monkeypatch.setattr(runner, "solve_many", lambda code, jobs, workers=1, **kw:
                        [(None, None)] * len(jobs))
    slow_eval(monkeypatch)
    row = runner.run_suite("t", "c6", paths, "sha", "h", suite_spec=spec(
        {"id": "sc", "kind": "sncalib", "split": "valid", "frames": "all",
         "leakage_games": ["l|s|m|other"]}))
    t = row["tasks"]["sc"]
    assert t["status"] == "complete", t
    assert t["aggregate"]["seconds_per_frame"] < 0.1


def test_gsr_calibration_seconds_per_frame_excludes_the_evaluator(paths, calib, monkeypatch):
    write_gsr_clip(paths)
    slow_eval(monkeypatch)
    row = runner.run_suite("t", "c7", paths, "sha", "h", suite_spec=spec(
        {"id": "gc", "kind": "gsr_calibration", "split": "valid", "clips": ["SNGS-990"],
         "every": 1}))
    t = row["tasks"]["gc"]
    assert t["status"] == "complete", t
    assert t["aggregate"]["seconds_per_frame"] < 0.1


def test_gsr_calibration_rows_carry_the_unverified_leakage_flag(paths, calib):
    write_gsr_clip(paths)
    row = runner.run_suite("t", "c8", paths, "sha", "h", suite_spec=spec(
        {"id": "gc", "kind": "gsr_calibration", "split": "valid", "clips": ["SNGS-990"],
         "every": 1}))
    assert row["tasks"]["gc"]["leakage"] == {"status": "unverified"}


# ---------------------------------------------------------------- speed-up: batching, one pass

def test_sncalib_decodes_stills_in_groups(paths, calib, monkeypatch):
    write_calib_zip(paths, ["00000", "00001", "00002"])
    sizes = []

    class Spy(FakeCal):
        def decode(self, frames, **kw):
            sizes.append(len(frames))
            return super().decode(frames, **kw)
    monkeypatch.setattr(runner, "_calibrator", lambda p: Spy())
    monkeypatch.setattr(runner, "solve_many", lambda code, jobs, workers=1, **kw:
                        [(None, None)] * len(jobs))
    row = runner.run_suite("t", "b1", paths, "sha", "h", suite_spec=spec(
        {"id": "sc", "kind": "sncalib", "split": "valid", "frames": "all", "batch": 2}))
    assert row["tasks"]["sc"]["status"] == "complete", row["tasks"]["sc"]
    assert sizes == [2, 1]
    assert calib[0][0] == ["00000", "00001", "00002"]


def test_sncalib_unreadable_image_fails_alone_before_batching(paths, calib, monkeypatch):
    write_calib_zip(paths, ["00000", "00001", "00002"])
    with zipfile.ZipFile(paths.calib_zip("valid")) as z:
        members = {n: z.read(n) for n in z.namelist()}
    members["valid/00001.jpg"] = b"not an image"
    with zipfile.ZipFile(paths.calib_zip("valid"), "w") as z:
        for n, b in members.items():
            z.writestr(n, b)
    sizes = []

    class Spy(FakeCal):
        def decode(self, frames, **kw):
            sizes.append(len(frames))
            return super().decode(frames, **kw)
    monkeypatch.setattr(runner, "_calibrator", lambda p: Spy())
    monkeypatch.setattr(runner, "solve_many", lambda code, jobs, workers=1, **kw:
                        [(None, None)] * len(jobs))
    row = runner.run_suite("t", "b2", paths, "sha", "h", suite_spec=spec(
        {"id": "sc", "kind": "sncalib", "split": "valid", "frames": "all", "batch": 2}))
    t = row["tasks"]["sc"]
    assert t["status"] == "partial" and list(t["failed"]) == ["00001"]
    assert sizes == [2]                                      # 00000 and 00002 share a group
    assert calib[0][0] == ["00000", "00001", "00002"]        # GT written for every frame first


def test_sncalib_a_group_that_fails_to_decode_is_retried_frame_by_frame(paths, calib, monkeypatch):
    write_calib_zip(paths, ["00000", "00001", "00002"])
    with zipfile.ZipFile(paths.calib_zip("valid")) as z:
        members = {n: z.read(n) for n in z.namelist()}
    members["valid/00001.jpg"] = cv2.imencode(".jpg", np.full((54, 96, 3), 255, np.uint8))[1].tobytes()
    with zipfile.ZipFile(paths.calib_zip("valid"), "w") as z:
        for n, b in members.items():
            z.writestr(n, b)
    sizes = []

    class PathologicalFrame(FakeCal):
        def decode(self, frames, **kw):          # raises for any call containing the white frame
            sizes.append(len(frames))
            if any(f.mean() > 128 for f in frames):
                raise ZeroDivisionError("degenerate detections")
            return super().decode(frames, **kw)
    monkeypatch.setattr(runner, "_calibrator", lambda p: PathologicalFrame())
    monkeypatch.setattr(runner, "solve_many", lambda code, jobs, workers=1, **kw:
                        [(None, None)] * len(jobs))
    row = runner.run_suite("t", "b3", paths, "sha", "h", suite_spec=spec(
        {"id": "sc", "kind": "sncalib", "split": "valid", "frames": "all", "batch": 2}))
    t = row["tasks"]["sc"]
    assert t["status"] == "partial"
    assert list(t["failed"]) == ["00001"] and "ZeroDivisionError" in t["failed"]["00001"]
    assert sorted(t["clips"]) == ["00000", "00002"]          # the good neighbour is still solved
    assert sizes == [2, 1, 1, 1]                             # group, then its frames singly, then the rest


def two_clips(paths):
    write_gsr_clip(paths, seq="SNGS-990", n=3)                  # 3 labelled, 3 predicted
    write_gsr_clip(paths, seq="SNGS-991", n=3, n_images=2)      # 3 labelled, 2 predicted


JAC = {"SNGS-990": 80.0, "SNGS-991": 20.0}


@pytest.fixture
def per_clip_eval(paths, monkeypatch):
    """Evaluator fake: a different JaC per clip, completeness from the files, call log."""
    monkeypatch.setattr(runner, "_calibrator", lambda p: FakeCal())
    monkeypatch.setattr(runner, "require_vram", lambda mib: None)
    calls = []

    def fake_eval(python, sc, gt_root, pred_root, split, out_dir, **kw):
        seq = next(p.stem for p in (gt_root / split).glob("*.json")).rsplit("-", 1)[0]
        gts = sorted(p.stem for p in (gt_root / split).glob("*.json"))
        preds = sorted(p.stem for p in (pred_root / split).glob("camera_*.json"))
        calls.append((seq, gts, preds))
        return CalibScores(score=JAC[seq] * len(preds) / len(gts),
                           completeness=100.0 * len(preds) / len(gts), jac=JAC[seq],
                           precision=1.0, recall=1.0)
    monkeypatch.setattr(runner, "run_sncalib", fake_eval)
    return calls


def test_gsr_calibration_one_evaluator_pass_per_clip_and_exact_aggregate(paths, per_clip_eval):
    two_clips(paths)
    row = runner.run_suite("t", "b4", paths, "sha", "h", suite_spec=spec(
        {"id": "gc", "kind": "gsr_calibration", "split": "valid",
         "clips": ["SNGS-990", "SNGS-991"], "every": 1, "eval_workers": 2}))
    t = row["tasks"]["gc"]
    assert t["status"] == "complete", t
    assert sorted(c[0] for c in per_clip_eval) == ["SNGS-990", "SNGS-991"]    # once per clip
    assert not (paths.run_dir("b4") / "work" / "gc" / "all").exists()
    a, b = t["clips"]["SNGS-990"], t["clips"]["SNGS-991"]
    assert (a["labelled"], a["predicted"], b["labelled"], b["predicted"]) == (3, 3, 3, 2)
    assert a["jac5"] == 80.0 and b["jac5"] == 20.0
    completeness = 100.0 * (3 + 2) / (3 + 3)
    jac5 = (80.0 * 3 + 20.0 * 2) / (3 + 2)
    agg = t["aggregate"]
    assert agg["completeness"] == pytest.approx(completeness)
    assert agg["jac5"] == pytest.approx(jac5) and jac5 == 56.0
    assert agg["score"] == pytest.approx(completeness * jac5 / 100)
    assert agg["calibrated_share"] == round((1.0 + 0.6667) / 2, 4)   # mean of rounded clip shares
    assert agg["keyframe_accept_rate"] == 1.0 and agg["solver_errors"] == 0
    assert "seconds_per_frame" in agg and t["leakage"] == {"status": "unverified"}


def test_gsr_calibration_a_clip_whose_evaluation_raises_is_dropped_from_the_aggregate(
        paths, per_clip_eval, monkeypatch):
    two_clips(paths)
    good = runner.run_sncalib

    def flaky(python, sc, gt_root, pred_root, split, out_dir, **kw):
        if "SNGS-991" in str(gt_root):
            raise KeyError("Penalty area left")
        return good(python, sc, gt_root, pred_root, split, out_dir, **kw)
    monkeypatch.setattr(runner, "run_sncalib", flaky)
    row = runner.run_suite("t", "b5", paths, "sha", "h", suite_spec=spec(
        {"id": "gc", "kind": "gsr_calibration", "split": "valid",
         "clips": ["SNGS-990", "SNGS-991"], "every": 1}))
    t = row["tasks"]["gc"]
    assert t["status"] == "partial" and "KeyError" in t["failed"]["SNGS-991"]
    assert list(t["clips"]) == ["SNGS-990"]
    agg = t["aggregate"]
    assert agg["completeness"] == 100.0 and agg["jac5"] == 80.0 and agg["score"] == 80.0
    assert agg["calibrated_share"] == 1.0 and agg["keyframe_accept_rate"] == 1.0


def test_gsr_calibration_evaluations_overlap_the_next_clips_calibration(paths, calib, monkeypatch):
    """Calibration stays sequential, evaluation runs beside it: with a slow evaluator the whole
    task takes about one evaluation per worker-wave, not one per clip."""
    import time
    for seq in ("SNGS-990", "SNGS-991", "SNGS-992", "SNGS-993"):
        write_gsr_clip(paths, seq=seq)
    slow_eval(monkeypatch, seconds=0.5)
    t0 = time.perf_counter()
    row = runner.run_suite("t", "b6", paths, "sha", "h", suite_spec=spec(
        {"id": "gc", "kind": "gsr_calibration", "split": "valid", "every": 1,
         "clips": ["SNGS-990", "SNGS-991", "SNGS-992", "SNGS-993"], "eval_workers": 4}))
    assert row["tasks"]["gc"]["status"] == "complete", row["tasks"]["gc"]
    assert time.perf_counter() - t0 < 1.5                    # sequential would be >= 2.0 s


def test_quick_suite_samples_the_gsr_clips_every_10_frames():
    q = runner.load_suite("quick")
    gc = next(t for t in q["tasks"] if t["id"] == "gsr-calibration-valid-6")
    assert gc["every"] == 10

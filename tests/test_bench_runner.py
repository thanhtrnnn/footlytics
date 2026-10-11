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


# ---------------------------------------------------------------- gsr_pipeline (plan 3)

from footlytics.bench.adapters.gsr import gt_to_state  # noqa: E402
from footlytics.geometry.clip_calib import FrameCalib  # noqa: E402
from footlytics.state.schema import Team  # noqa: E402

CALIB_STATS = {"keyframes": 1, "keyframes_accepted": 1, "solver_errors": 0}
_SWAP = {Team.HOME.value: Team.AWAY.value, Team.AWAY.value: Team.HOME.value}


def pipeline_fakes(monkeypatch, empty=(), uncalibrated=(), crash=(), real_calibration=False,
                   ball_shift=None):
    """No GPU, no model. Calibration is a stub (identity homographies), the pipeline returns the
    ground truth with HOME and AWAY swapped (a side-mapping bug would show as wrong teams).
    `empty`: run_frames raises "no detections at all"; `uncalibrated`: no homography on any
    frame, so the pipeline sees nothing; `crash`: the calibrator itself raises;
    `ball_shift`: `{clip: px}` moves that clip's predicted ball box right by `px` pixels."""
    ball_shift = ball_shift or {}
    seen = {"vram": [], "calibrated": [], "run": [], "mapped": {}}
    monkeypatch.setattr(runner, "require_vram", lambda mib: seen["vram"].append(mib))
    monkeypatch.setattr(runner, "_detector", lambda paths, cfg: "detector")
    if not real_calibration:
        monkeypatch.setattr(runner, "_calibrator", lambda paths: FakeCal())   # loaded up front

        def calibration(ctx, z, split, seq, cfg, every, gate, fresh=False):
            seen["calibrated"].append((seq, every, gate, fresh))
            if seq in crash:
                raise RuntimeError("calibrator crashed")
            n = sum(1 for m in z.namelist() if m.startswith(f"{seq}/img1/"))
            H = None if seq in uncalibrated else np.eye(3)
            return ([FrameCalib(i, H, None, "keyframe" if H is not None else "none")
                     for i in range(n)], 0.0, CALIB_STATS)
        monkeypatch.setattr(runner, "_clip_calibration", calibration)

    def run_frames(frames, fps, mapper, detector, meta, cfg=None, tracker_cfg=None,
                   source="frames"):
        n = sum(1 for _ in frames)
        seen["run"].append((meta.match_id, n, fps, cfg.montage, detector))
        seen["mapped"][meta.match_id] = mapper(0, None, np.array([[0.0, 0.0, 10.0, 10.0, 0.9, 0]]))
        if meta.match_id in empty or meta.match_id in uncalibrated:
            raise RuntimeError("no detections at all")
        state = gt_to_state(make_labels(seq=meta.match_id, n_frames=3))
        state.tracks["team"] = (state.tracks["team"].astype(str)
                                .map(lambda t: _SWAP.get(t, t)).astype("category"))
        if meta.match_id in ball_shift:
            tr = state.tracks
            tr["bbox_x"] = tr["bbox_x"].astype(float)
            tr.loc[tr["role"].astype(str) == "ball", "bbox_x"] += ball_shift[meta.match_id]
        return state, {"frames_processed": n, "frames_calibrated": n, "calib_success_rate": 1.0}
    monkeypatch.setattr(runner, "run_frames", run_frames)
    return seen


def pipeline_task(clips, **kw):
    return {"id": "gp", "kind": "gsr_pipeline", "split": "valid", "clips": clips, **kw}


def pipeline_dir(paths, run_id):
    return (paths.run_dir(run_id) / "work" / "gp" / "trackers" / "SoccerNetGS-valid"
            / "footlytics")


def test_gsr_pipeline_scores_a_clip_and_its_extras(paths, monkeypatch):
    write_gsr_clip(paths)
    seen = pipeline_fakes(monkeypatch)
    calls = []
    monkeypatch.setattr(runner, "run_gsr", fake_scores(calls))
    row = runner.run_suite("t", "p1", paths, "sha", "h", suite_spec=spec(
        pipeline_task(["SNGS-990"])))
    t = row["tasks"]["gp"]
    assert t["status"] == "complete", t
    c, agg = t["clips"]["SNGS-990"], t["aggregate"]
    assert c["hota_jersey_off"] == agg["hota_jersey_off"] == 99.9
    assert c["hota_jersey_on"] == agg["hota_jersey_on"] == 99.8
    assert c["deta_jersey_off"] == 99.9 and c["assa_jersey_off"] == 99.9
    assert c["team_accuracy"] == 1.0 and agg["team_accuracy"] == 1.0   # the swap is not an error
    assert c["shape_error_m"] is None and agg["shape_error_m"] is None   # 3 a side < min_players
    assert c["ball_recall"] == 1.0 and agg["ball_recall"] == 1.0
    assert c["ball_px_median_hits"] == agg["ball_px_median_hits"] == 0.0
    assert c["ball_wrong_share"] == agg["ball_wrong_share"] == 0.0
    assert "ball_px_median" not in c and "ball_px_median" not in agg
    assert not [k for k in c if k.startswith("_")]               # raw pooling data stays private
    assert c["calib_success_rate"] == 1.0 and agg["calib_success_rate"] == 1.0
    assert c["seconds_per_match_minute"] >= 0 and agg["seconds_per_match_minute"] >= 0
    assert t["leakage"] == {"status": "unverified"} and t["failed"] == {}
    assert {x[3] for x in calls} == {False, True} and all(x[2] == ("SNGS-990",) for x in calls)
    preds = json.loads((pipeline_dir(paths, "p1") / "SNGS-990.json").read_text())["predictions"]
    assert preds and all(p["attributes"]["jersey"] is None for p in preds)
    # what reached the pipeline: the clip's frames at 25 fps, no montage, the cached detector,
    # and the calibration's homographies behind the mapper (identity: feet = box bottom middle)
    assert seen["vram"] == [runner.PIPELINE_VRAM_MIB] == [14_000]
    assert seen["calibrated"] == [("SNGS-990", 10, 25.0, False)]
    assert seen["run"] == [("SNGS-990", 3, 25.0, False, "detector")]
    assert seen["mapped"]["SNGS-990"].tolist() == [[5.0, 10.0]]


def run_pool(paths, monkeypatch, run_id, **fakes):
    """Every clip of the split through the fake pipeline; the task's result."""
    pipeline_fakes(monkeypatch, **fakes)
    monkeypatch.setattr(runner, "run_gsr", fake_scores([]))
    return runner.run_suite("t", run_id, paths, "sha", "h", suite_spec=spec(
        pipeline_task("all")))["tasks"]["gp"]


def test_gsr_pipeline_pools_ball_frames_across_clips(paths, monkeypatch):
    """One clip localises every ball at 4 px, the other picks a wrong object 900 px away on
    every frame: the aggregate is over the pooled frames, never an average of clip numbers, so
    the wrong object cannot drag the localisation error and the hits cannot hide the wrongs."""
    write_gsr_clip(paths, seq="SNGS-990")
    write_gsr_clip(paths, seq="SNGS-991")
    t = run_pool(paths, monkeypatch, "pl1",
                 ball_shift={"SNGS-990": 4.0, "SNGS-991": 900.0})
    assert t["status"] == "complete", t
    a, b = t["clips"]["SNGS-990"], t["clips"]["SNGS-991"]
    assert (a["ball_recall"], a["ball_px_median_hits"], a["ball_wrong_share"]) == (1.0, 4.0, 0.0)
    assert (b["ball_recall"], b["ball_px_median_hits"], b["ball_wrong_share"]) == (0.0, None, 1.0)
    agg = t["aggregate"]
    assert agg["ball_recall"] == 0.5 and agg["ball_px_median_hits"] == 4.0
    assert agg["ball_wrong_share"] == 0.5
    assert "ball_px_median" not in agg
    assert not [k for c in t["clips"].values() for k in c if k.startswith("_")]


def test_gsr_pipeline_ball_aggregate_weights_clips_by_their_gt_ball_frames(paths, monkeypatch):
    """A clip with more labelled ball frames counts for more: 3 hits of 3 and 3 hits of 6 pool
    to 6 of 9, where the mean of the clips' recalls would say 0.75."""
    write_gsr_clip(paths, seq="SNGS-990", n=3)
    write_gsr_clip(paths, seq="SNGS-991", n=6)            # the fake pipeline sees 3 of its frames
    t = run_pool(paths, monkeypatch, "pl2",
                 ball_shift={"SNGS-990": 4.0, "SNGS-991": 10.0})
    assert t["clips"]["SNGS-991"]["ball_recall"] == 0.5
    agg = t["aggregate"]
    assert agg["ball_recall"] == round(6 / 9, 4) and agg["ball_wrong_share"] == 0.0
    assert agg["ball_px_median_hits"] == 7.0                      # median of 4, 4, 4, 10, 10, 10


def test_gsr_pipeline_isolates_a_clip_with_no_detections(paths, monkeypatch):
    """A clip whose pipeline produces nothing is still scored (empty predictions: all its GT
    people are misses), so it cannot shrink the GS-HOTA denominator; the reason is recorded."""
    write_gsr_clip(paths, seq="SNGS-990")
    write_gsr_clip(paths, seq="SNGS-991")
    pipeline_fakes(monkeypatch, empty=("SNGS-991",))
    calls = []
    monkeypatch.setattr(runner, "run_gsr", fake_scores(calls))
    row = runner.run_suite("t", "p2", paths, "sha", "h", suite_spec=spec(
        pipeline_task("all")))
    t = row["tasks"]["gp"]
    assert t["status"] == "partial" and row["status"] == "partial"
    assert t["failed"] == {"SNGS-991": "RuntimeError: no detections at all"}
    assert t["clips"]["SNGS-991"]["error"] == "RuntimeError: no detections at all"
    # the empty predictions file reached the scorer, beside the good clip's
    out = pipeline_dir(paths, "p2")
    assert json.loads((out / "SNGS-991.json").read_text()) == {"predictions": []}
    assert all(c[2] == ("SNGS-990", "SNGS-991") for c in calls) and len(calls) == 2
    assert t["clips"]["SNGS-991"]["hota_jersey_off"] == 99.9       # scored, not skipped
    # the good clip is scored normally
    good = t["clips"]["SNGS-990"]
    assert "error" not in good and good["team_accuracy"] == 1.0 and good["ball_recall"] == 1.0
    assert json.loads((out / "SNGS-990.json").read_text())["predictions"]
    # clip means skip the clip that has no value rather than counting a zero for it
    assert t["aggregate"]["team_accuracy"] == 1.0 and t["aggregate"]["ball_recall"] == 1.0


def test_gsr_pipeline_counts_a_clip_calibration_never_found_as_uncalibrated(paths, monkeypatch):
    write_gsr_clip(paths, seq="SNGS-990")
    write_gsr_clip(paths, seq="SNGS-991")
    pipeline_fakes(monkeypatch, uncalibrated=("SNGS-991",))
    monkeypatch.setattr(runner, "run_gsr", fake_scores([]))
    row = runner.run_suite("t", "p3", paths, "sha", "h", suite_spec=spec(pipeline_task("all")))
    t = row["tasks"]["gp"]
    assert t["status"] == "partial" and "no detections at all" in t["failed"]["SNGS-991"]
    assert t["clips"]["SNGS-991"]["calib_success_rate"] == 0.0     # no inflating the mean
    assert t["aggregate"]["calib_success_rate"] == 0.5
    assert json.loads((pipeline_dir(paths, "p3") / "SNGS-991.json").read_text()) == \
        {"predictions": []}


def test_gsr_pipeline_a_calibrator_crash_is_scored_like_an_empty_pipeline(paths, monkeypatch):
    write_gsr_clip(paths, seq="SNGS-990")
    write_gsr_clip(paths, seq="SNGS-991")
    seen = pipeline_fakes(monkeypatch, crash=("SNGS-991",))
    calls = []
    monkeypatch.setattr(runner, "run_gsr", fake_scores(calls))
    row = runner.run_suite("t", "p4", paths, "sha", "h", suite_spec=spec(pipeline_task("all")))
    t = row["tasks"]["gp"]
    assert t["status"] == "partial"
    assert t["failed"] == {"SNGS-991": "RuntimeError: calibrator crashed"}
    assert calls[0][2] == ("SNGS-990", "SNGS-991")
    assert json.loads((pipeline_dir(paths, "p4") / "SNGS-991.json").read_text()) == \
        {"predictions": []}
    assert [r[0] for r in seen["run"]] == ["SNGS-990"]               # never reached the pipeline


def test_gsr_pipeline_leaves_out_a_clip_it_cannot_even_prepare(paths, monkeypatch):
    """Bad labels or no frames: there is nothing to write a predictions file against, so the
    clip is in `failed` only and the scorer never sees it."""
    write_gsr_clip(paths, seq="SNGS-990")
    write_gsr_clip(paths, seq="SNGS-991")
    write_gsr_clip(paths, seq="SNGS-992", n_images=0)
    labels_file = paths.gsr_labels / "valid" / "SNGS-991" / "Labels-GameState.json"
    labels = json.loads(labels_file.read_text())
    labels["info"]["version"] = "1.2"
    labels_file.write_text(json.dumps(labels))
    seen = pipeline_fakes(monkeypatch)
    calls = []
    monkeypatch.setattr(runner, "run_gsr", fake_scores(calls))
    row = runner.run_suite("t", "p5", paths, "sha", "h", suite_spec=spec(pipeline_task("all")))
    t = row["tasks"]["gp"]
    assert t["status"] == "partial"
    assert "1.3" in t["failed"]["SNGS-991"] and "FileNotFoundError" in t["failed"]["SNGS-992"]
    assert list(t["clips"]) == ["SNGS-990"]
    assert all(c[2] == ("SNGS-990",) for c in calls) and calls
    out = pipeline_dir(paths, "p5")
    assert not (out / "SNGS-991.json").exists() and not (out / "SNGS-992.json").exists()
    assert [r[0] for r in seen["run"]] == ["SNGS-990"]


def test_gsr_pipeline_fails_the_task_when_no_clip_produced_anything(paths, monkeypatch):
    """Every clip empty is a broken run (wrong weights, dead GPU), not a score of zero."""
    write_gsr_clip(paths, seq="SNGS-990")
    write_gsr_clip(paths, seq="SNGS-991")
    pipeline_fakes(monkeypatch, empty=("SNGS-990", "SNGS-991"))
    calls = []
    monkeypatch.setattr(runner, "run_gsr", fake_scores(calls))
    row = runner.run_suite("t", "p6", paths, "sha", "h", suite_spec=spec(pipeline_task("all")))
    t = row["tasks"]["gp"]
    assert t["status"] == "failed" and "no clip ran" in t["error"]
    assert "no detections at all" in t["error"] and calls == []


def test_gsr_pipeline_a_clip_whose_metrics_fail_is_still_scored(paths, monkeypatch):
    """Its predictions file is written before the metrics, so the scorer counts it; the reason
    is recorded and the clip keeps what the scorer gives."""
    write_gsr_clip(paths)
    pipeline_fakes(monkeypatch)
    calls = []
    monkeypatch.setattr(runner, "run_gsr", fake_scores(calls))
    monkeypatch.setattr(runner, "match_people",
                        lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad frames")))
    row = runner.run_suite("t", "p7", paths, "sha", "h", suite_spec=spec(
        pipeline_task(["SNGS-990"])))
    t = row["tasks"]["gp"]
    assert t["status"] == "partial" and t["failed"] == {"SNGS-990": "ValueError: bad frames"}
    assert t["clips"]["SNGS-990"]["hota_jersey_off"] == 99.9 and calls


def test_clip_calibration_is_shared_and_fresh_bypasses_it(paths, monkeypatch):
    write_gsr_clip(paths)
    fits = []

    class Spy(FakeCal):
        def fit(self, frames, which=None, **kw):
            fits.append(list(which))
            return super().fit(frames, which, **kw)
    spy = Spy()
    monkeypatch.setattr(runner, "_calibrator", lambda p: spy)
    ctx = runner.TaskContext(paths, paths.root / "work", lambda m: None)
    with zipfile.ZipFile(paths.gsr_zip("valid")) as z:
        first = runner._clip_calibration(ctx, z, "valid", "SNGS-990", {}, 1, 25.0)
        res, seconds, stats = first
        assert len(res) == 3 and seconds >= 0
        assert stats == {"keyframes": 3, "keyframes_accepted": 3, "solver_errors": 0}
        assert len(fits) == 1
        again = runner._clip_calibration(ctx, z, "valid", "SNGS-990", {}, 1, 25.0)
        assert again is first and len(fits) == 1                    # shared, not recomputed
        redo = runner._clip_calibration(ctx, z, "valid", "SNGS-990", {}, 1, 25.0, fresh=True)
        assert redo is not first and len(fits) == 2                 # fresh calibrates again ...
        assert runner._clip_calibration(ctx, z, "valid", "SNGS-990", {}, 1, 25.0) is first
        assert len(fits) == 2                                       # ... and leaves the cache alone
        runner._clip_calibration(ctx, z, "valid", "SNGS-990", {}, 2, 25.0)       # other settings
        runner._clip_calibration(ctx, z, "valid", "SNGS-990", {"mode": "x"}, 1, 25.0)
        assert len(fits) == 4


def test_clip_calibration_counts_solver_errors_and_names_a_missing_clip(paths, monkeypatch):
    write_gsr_clip(paths)

    class ErrCal(FakeCal):
        def fit(self, frames, which=None, **kw):
            self.last_errors = {0: "ZeroDivisionError: x"}
            out = super().fit(frames, which, **kw)
            out[0] = None
            return out
    monkeypatch.setattr(runner, "_calibrator", lambda p: ErrCal())
    ctx = runner.TaskContext(paths, paths.root / "work", lambda m: None)
    with zipfile.ZipFile(paths.gsr_zip("valid")) as z:
        _, _, stats = runner._clip_calibration(ctx, z, "valid", "SNGS-990", {}, 1, 25.0)
        assert stats["solver_errors"] == 1 and stats["keyframes_accepted"] == 2
        with pytest.raises(FileNotFoundError, match="SNGS-123"):
            runner._clip_calibration(ctx, z, "valid", "SNGS-123", {}, 1, 25.0)


def test_gsr_calibration_and_gsr_pipeline_calibrate_a_clip_once(paths, calib, monkeypatch):
    write_gsr_clip(paths)
    fits = []

    class Spy(FakeCal):
        def fit(self, frames, which=None, **kw):
            fits.append(list(which))
            return super().fit(frames, which, **kw)
    monkeypatch.setattr(runner, "_calibrator", lambda p: Spy())
    pipeline_fakes(monkeypatch, real_calibration=True)
    monkeypatch.setattr(runner, "run_gsr", fake_scores([]))
    row = runner.run_suite("t", "p8", paths, "sha", "h", suite_spec=spec(
        {"id": "gc", "kind": "gsr_calibration", "split": "valid", "clips": ["SNGS-990"],
         "every": 1},
        {**pipeline_task(["SNGS-990"]), "id": "gp", "every": 1}))
    assert row["tasks"]["gc"]["status"] == "complete", row["tasks"]["gc"]
    assert row["tasks"]["gp"]["status"] == "complete", row["tasks"]["gp"]
    assert len(fits) == 1
    assert row["tasks"]["gp"]["clips"]["SNGS-990"]["calib_success_rate"] == 1.0


def test_detector_is_built_once_per_config_from_the_pinned_weights(paths, monkeypatch):
    from footlytics.perception import detect
    built = []

    class FakeDetector:
        def __init__(self, weights, cfg):
            built.append((weights, cfg))
    monkeypatch.setattr(detect, "Detector", FakeDetector)
    monkeypatch.setattr(runner, "_DETECTORS", {})
    a = runner._detector(paths, {})
    assert runner._detector(paths, {}) is a and len(built) == 1
    weights, cfg = built[0]
    assert weights == paths.detector_weights and cfg.ball_weights == str(paths.ball_weights)
    assert (cfg.tile, cfg.imgsz, cfg.conf, cfg.device, cfg.half) == (0, 1280, 0.25, "cuda", True)
    b = runner._detector(paths, {"tile": 1280, "conf": 0.3})
    assert b is not a and len(built) == 2
    assert (built[1][1].tile, built[1][1].conf) == (1280, 0.3)


def test_noise_of_reports_absolute_differences(paths, monkeypatch):
    monkeypatch.setitem(runner.KINDS, "n1", lambda task, ctx: {
        "aggregate": {"m": 1.0, "s": "x", "flag": True}})
    monkeypatch.setitem(runner.KINDS, "n2", lambda task, ctx: {
        "aggregate": {"m": 1.25, "flag": False}})
    row = runner.run_suite("t", "n1", paths, "sha", "h", suite_spec=spec(
        {"id": "a", "kind": "n1"},
        {"id": "b", "kind": "n2", "noise_of": "a"}))
    assert row["tasks"]["b"]["noise"] == {"m": 0.25}          # a bool is not a measurement
    assert "noise" not in row["tasks"]["a"] and "noise_skipped" not in row["tasks"]["b"]


def test_noise_of_is_only_measured_between_two_complete_runs(paths, monkeypatch):
    """A repeat that lost a clip scores differently for that reason, not by chance: its
    difference is not noise, and must not widen the floor."""
    agg = {"aggregate": {"m": 1.0}}
    monkeypatch.setitem(runner.KINDS, "ok", lambda task, ctx: agg)
    monkeypatch.setitem(runner.KINDS, "part", lambda task, ctx: {**agg, "failed": {"SNGS-1": "x"}})
    monkeypatch.setitem(runner.KINDS, "boom", lambda task, ctx: 1 / 0)
    row = runner.run_suite("t", "n2", paths, "sha", "h", suite_spec=spec(
        {"id": "a", "kind": "ok"}, {"id": "p", "kind": "part"}, {"id": "x", "kind": "boom"},
        {"id": "b", "kind": "ok", "noise_of": "a"},
        {"id": "c", "kind": "ok", "noise_of": "p"},         # the reference is partial
        {"id": "d", "kind": "part", "noise_of": "a"},       # this repeat is partial
        {"id": "e", "kind": "ok", "noise_of": "x"},         # the reference crashed
        {"id": "f", "kind": "ok", "noise_of": "not-run"}))  # a typo is not silence
    t = row["tasks"]
    assert t["b"]["noise"] == {"m": 0.0} and "noise_skipped" not in t["b"]
    assert t["c"]["noise_skipped"] == "p is partial"
    assert t["d"]["noise_skipped"] == "d is partial"
    assert t["e"]["noise_skipped"] == "x is failed"
    assert t["f"]["noise_skipped"] == "not-run has not run"
    assert all("noise" not in t[k] for k in "cdef")


def test_pipeline_suites_are_valid():
    for name in ("quick", "nightly", "milestone"):
        assert {t["kind"] for t in runner.load_suite(name)["tasks"]} <= set(runner.KINDS)
    quick = {t["id"]: t for t in runner.load_suite("quick")["tasks"]}
    gp = quick["gsr-pipeline-valid-6"]
    assert gp["kind"] == "gsr_pipeline" and gp["split"] == "valid"
    assert gp["clips"] == quick["gsr-calibration-valid-6"]["clips"] == [
        "SNGS-025", "SNGS-028", "SNGS-050", "SNGS-056", "SNGS-081", "SNGS-086"]
    first, second = runner.load_suite("nightly")["tasks"]
    assert first["kind"] == second["kind"] == "gsr_pipeline"
    assert first["clips"] == second["clips"] == "all" and first["split"] == "valid"
    assert second["noise_of"] == first["id"] and second["fresh"] is True
    assert "fresh" not in first and "noise_of" not in first
    milestone = {t["id"]: t for t in runner.load_suite("milestone")["tasks"]}
    assert milestone["gsr-pipeline-test"]["kind"] == "gsr_pipeline"
    assert milestone["gsr-pipeline-test"]["split"] == "test"
    assert milestone["gsr-baseline-test"]["kind"] == "gsr_rescore"
    assert milestone["gsr-baseline-test"]["split"] == "test"


# ---------------------------------------------------------------- calibrator load: task level

def broken_calibrator(monkeypatch, loads):
    def load(paths):
        loads.append(1)
        raise RuntimeError("weights missing")
    monkeypatch.setattr(runner, "_calibrator", load)


def test_gsr_pipeline_loads_the_calibrator_once_up_front_and_fails_loudly(paths, monkeypatch):
    """A broken calibrator is a broken run, not a clip scored as all misses: it is loaded
    before the clip loop, once, and its failure fails the task with that error."""
    write_gsr_clip(paths, seq="SNGS-990")
    write_gsr_clip(paths, seq="SNGS-991")
    seen = pipeline_fakes(monkeypatch, real_calibration=True)
    loads = []
    broken_calibrator(monkeypatch, loads)
    calls = []
    monkeypatch.setattr(runner, "run_gsr", fake_scores(calls))
    row = runner.run_suite("t", "p9", paths, "sha", "h", suite_spec=spec(pipeline_task("all")))
    t = row["tasks"]["gp"]
    assert t["status"] == "failed" and t["error"] == "RuntimeError: weights missing"
    assert len(loads) == 1 and calls == [] and seen["run"] == []
    out = pipeline_dir(paths, "p9")
    assert not out.exists() or not list(out.glob("*.json"))      # no clip was written or scored


def test_gsr_calibration_loads_the_calibrator_once_up_front_and_fails_loudly(
        paths, calib, monkeypatch):
    write_gsr_clip(paths, seq="SNGS-990")
    write_gsr_clip(paths, seq="SNGS-991")
    loads = []
    broken_calibrator(monkeypatch, loads)
    row = runner.run_suite("t", "c9", paths, "sha", "h", suite_spec=spec(
        {"id": "gc", "kind": "gsr_calibration", "split": "valid",
         "clips": ["SNGS-990", "SNGS-991"], "every": 1}))
    t = row["tasks"]["gc"]
    assert t["status"] == "failed" and t["error"] == "RuntimeError: weights missing"
    assert len(loads) == 1 and calib == []                       # nothing reached the evaluator

"""Calibration adapter, our median-px metric, and the SN-Calibration evaluator wrapper."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from footlytics.bench.adapters.calib import camera_json, gsr_line_frames, mirror, write_camera
from footlytics.bench.evaluators import EvaluatorError, parse_sncalib, run_sncalib
from footlytics.bench.metrics import median_reprojection_px
from footlytics.geometry.sn_camera import ground_lines, make_camera, project, scale_camera

CAM = make_camera(0, 80, 0, 1500, (0, 55, -15), 1920, 1080)


def labels_for(cam, width=1920, height=1080, n=12):
    """Ground-truth lines as an annotator would draw them: points along each visible line,
    normalised the way the official evaluator de-normalises them (x * (width - 1))."""
    out = {}
    for name, (a, b) in ground_lines().items():
        w = np.linspace(0, 1, n)[:, None]
        pts = project(cam, np.c_[(1 - w) * a + w * np.array(b), np.zeros(n)])
        ok = np.isfinite(pts).all(1) & (pts[:, 0] >= 0) & (pts[:, 0] < width) & \
            (pts[:, 1] >= 0) & (pts[:, 1] < height)
        if ok.sum() >= 2:
            out[name] = [{"x": float(x / (width - 1)), "y": float(y / (height - 1))}
                         for x, y in pts[ok]]
    return out


def test_camera_json_is_plain_and_complete(tmp_path):
    cam = {**CAM, "pan_degrees": np.float64(1.5), "rotation_matrix": [[1, 0, 0]] * 3}
    j = camera_json(cam)
    assert "rotation_matrix" not in j and isinstance(j["pan_degrees"], float)
    write_camera(tmp_path / "camera_00001.json", cam)
    assert json.loads((tmp_path / "camera_00001.json").read_text())["x_focal_length"] == 1500.0


def test_mirror_swaps_left_right_and_top_bottom():
    m = mirror({"Big rect. left top": [1], "Side line bottom": [2], "Middle line": [3]})
    assert m == {"Big rect. right bottom": [1], "Side line top": [2], "Middle line": [3]}


def test_gsr_line_frames_keeps_labelled_frames_only():
    labels = {"images": [
        {"image_id": "2990000001", "file_name": "000001.jpg", "has_labeled_pitch": True},
        {"image_id": "2990000002", "file_name": "000002.jpg", "has_labeled_pitch": False}],
        "annotations": [
            {"image_id": "2990000001", "supercategory": "pitch", "lines": {"Middle line": [
                {"x": 0.5, "y": 0.1}, {"x": 0.5, "y": 0.9}]}},
            {"image_id": "2990000002", "supercategory": "pitch", "lines": {"Middle line": []}}]}
    assert gsr_line_frames(labels) == {0: {"Middle line": [{"x": 0.5, "y": 0.1},
                                                           {"x": 0.5, "y": 0.9}]}}


def test_gsr_line_frames_uses_the_evaluator_spelling():
    labels = {"images": [
        {"image_id": "2990000001", "file_name": "000001.jpg", "has_labeled_pitch": True}],
        "annotations": [
            {"image_id": "2990000001", "supercategory": "pitch", "lines": {
                "Goal left post left": [{"x": 0.1, "y": 0.4}, {"x": 0.1, "y": 0.6}],
                "Goal right post left": [{"x": 0.9, "y": 0.4}, {"x": 0.9, "y": 0.6}]}}]}
    assert set(gsr_line_frames(labels)[0]) == {"Goal left post left ", "Goal right post left"}


def test_median_px_is_zero_for_the_true_camera_and_grows_with_error():
    gt = labels_for(CAM)
    assert median_reprojection_px(gt, CAM) < 0.5
    off = make_camera(1.0, 80, 0, 1500, (0, 55, -15), 1920, 1080)
    assert median_reprojection_px(gt, off) > 3


def test_median_px_takes_the_better_mirror():
    gt = labels_for(CAM)
    assert median_reprojection_px(mirror(gt), CAM) < 0.5


def test_median_px_ignores_points_projected_behind_the_camera():
    wide = make_camera(0, 89.5, 0, 400, (0, 40, -2), 1920, 1080)   # near-horizontal view
    gt = labels_for(wide, n=60)          # same sampling as the metric's polyline
    v = median_reprojection_px(gt, wide)
    assert v is not None and np.isfinite(v) and v < 0.5


def test_median_px_without_ground_lines_is_none():
    assert median_reprojection_px({"Circle central": [{"x": 0.5, "y": 0.5}]}, CAM) is None


OUT = """ On SoccerNet valid set, final score of : 0.5123
 On SoccerNet valid set, completeness rate of : 0.95
 On SoccerNet valid set, recall mean value : 61.00% with standard deviation of 1.00% and median of 1.00%
 On SoccerNet valid set, precision mean value : 70.00% with standard deviation of 1.00% and median of 1.00%
 On SoccerNet valid set, accuracy mean value :  53.93% with standard deviation of 1.00% and median of 1.00%
"""


def test_parse_sncalib_reads_percentages():
    s = parse_sncalib(OUT)
    assert (s.score, s.completeness, s.jac, s.precision, s.recall) == (51.23, 95.0, 53.93, 70.0, 61.0)


def test_parse_sncalib_with_nothing_predicted_is_zero_not_nan():
    s = parse_sncalib(" final score of : nan\n completeness rate of : 0.0\n")
    assert s.score == 0.0 and s.completeness == 0.0 and s.jac == 0.0


def test_sncalib_failure_carries_output(tmp_path):
    def fake(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Invalid dataset path !")
    with pytest.raises(EvaluatorError, match="Invalid dataset path"):
        run_sncalib(Path("/py"), Path("/sc"), tmp_path, tmp_path, "valid", tmp_path / "o", run=fake)


SC = os.getenv("FOOTLYTICS_SNCALIB")
EVAL_PY = os.getenv("FOOTLYTICS_EVAL_PYTHON")


@pytest.mark.skipif(not (SC and EVAL_PY), reason="set FOOTLYTICS_SNCALIB and FOOTLYTICS_EVAL_PYTHON")
def test_official_evaluator_prefers_the_true_camera(tmp_path):
    gt_dir, pred_dir = tmp_path / "gt" / "valid", tmp_path / "pred" / "valid"
    gt_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)
    off = make_camera(4.0, 80, 0, 1500, (0, 55, -15), 1920, 1080)
    for name, cam in (("00000", CAM), ("00001", off)):
        (gt_dir / f"{name}.json").write_text(json.dumps(labels_for(CAM)))
    # the evaluator reads the image size from the camera's principal point, so cameras are
    # written at the evaluation resolution (960 x 540)
    write_camera(pred_dir / "camera_00000.json", scale_camera(CAM, 960))
    s_true = run_sncalib(Path(EVAL_PY), Path(SC), tmp_path / "gt", tmp_path / "pred", "valid",
                         tmp_path / "o1")
    assert s_true.completeness == 50.0 and s_true.jac > 0
    # Absolute floor: our camera convention against SoccerNet's own camera.py. Measured 72.73
    # (recall 100; precision < 100 as circles and goals are projected but not labelled); a
    # flipped tilt or y axis scores 0.
    assert s_true.jac > 60 and s_true.recall == 100.0
    write_camera(pred_dir / "camera_00001.json", scale_camera(off, 960))
    s_both = run_sncalib(Path(EVAL_PY), Path(SC), tmp_path / "gt", tmp_path / "pred", "valid",
                         tmp_path / "o2")
    assert s_both.completeness == 100.0 and s_both.jac < s_true.jac

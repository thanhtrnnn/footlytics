# Benchmark calibration Implementation Plan (plan 2 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score our camera calibration with the official SN-Calibration evaluator, both on
SN-Calibration still frames and per frame on GSR broadcast clips. The per-clip calibrator
(solve on keyframes, bridge with the camera tracker) is the one plan 3 feeds into the pipeline.

**Architecture:** A full-camera calibrator built from SoccerMaster's keypoint and line heads and
PnLCalib's solver. That code is imported from the unlicensed SoccerMaster checkout and never
copied into this repo. Our own SoccerNet camera math converts between cameras and homographies.
Two new runner task kinds use it: `sncalib` (stills) and `gsr_calibration` (clips; the ground
truth is the GSR pitch-line labels, which use the SN-Calibration format). The evaluator,
`sn-calibration/src/evaluate_camera.py`, runs as a subprocess in `venv-eval`.

**Tech Stack:** numpy, OpenCV, torch (system 2.7.1+cu128 on the host), transformers, timm,
einops, accelerate, shapely (new `calib` extra); SoccerMaster code + weights; sn-calibration at
`ab38f461`.

**Spec:** `docs/superpowers/specs/2026-10-11-benchmark-design.md` (build-order step 2; suites;
validity rules "Calibration output" and "Leakage").

## Global Constraints

- Everything from plan 1's Global Constraints still holds: shared host `iec4090_2`, all state under `~/footlytics-bench/`, 1.5 GB disk floor, torch pin checked every run, evaluators run as subprocesses and are never imported, the test split is for milestones only and never for tuning.
- SoccerMaster code has no licence: internal R&D only. It is imported from `code_dir` at runtime and never vendored. The weights (Apache-2.0) are fetched from HF `xleprime/SoccerMaster` at revision `0d94573662dbd678df22aa5c61ae77f474b939c9`.
- The calibration output is the full SoccerNet camera (pan/tilt/roll, position, focal, principal point, zero distortion), not just the ground homography.
- Official score = JaC@5 × completeness at 960×540, from the official evaluator. A frame we fail to calibrate gets no prediction file and counts as missed. Median reprojection px is ours and auxiliary.
- Calibrator settings are tuned only on SN-Calibration valid frames and GSR valid clips that the quick suite does not use.
- Tests: `uv run --extra dev --extra calib pytest`. Tests that need SoccerMaster code, weights or data skip unless their env vars are set.

## Facts verified while planning (2026-10-11)

- **SN-Calibration valid zip** (`data/calibration/valid.zip` on the host): 3,212 frames
  `valid/00000.jpg` (960×540) + `valid/00000.json`. Each JSON maps a line name (e.g.
  `"Side line top"`) to a list of `{"x","y"}` points normalised to [0, 1]. There is also
  `valid/match_info.json` (frame → league, season, match, date) and `valid/per_match_info.json`.
  The frames come from 55 games.
- **Evaluator** `src/evaluate_camera.py -s GT -p PRED --split S -t 5 [--resolution_width 960
  --resolution_height 540]`, imported as `from src.camera ...` (so `PYTHONPATH` = repo root).
  - Inputs: GT JSONs are read from `GT/S/*.json`, predictions from `PRED/S/camera_<frame>.json`.
  - Its listing takes every `*.json`, so `match_info.json` would count as a missed frame: the
    GT directory we hand it holds only frame JSONs.
  - It tries each frame's labels mirrored through the centre (left↔right, top↔bottom) and keeps
    the better result.
  - It prints `final score of : <frac>`, `completeness rate of : <frac>` and
    `accuracy mean value :  <pct>%` (= JaC@T), plus precision and recall.
  - It plots matplotlib histograms, so it runs with `MPLBACKEND=Agg`.
- **Camera convention** (SN `camera.py`, reimplemented, not imported):
  - R = (Rz(pan)·Rx(tilt)·Rz(roll))ᵀ, a world point maps to the camera frame as
    x_cam = R(X − position), pixels = f·x/z + principal point.
  - World axes: origin at the centre spot, x along the length, y negative toward the far (top)
    touchline, z negative up (crossbar at z = −2.44).
  - The evaluator recovers the image size as 2 × principal point and rescales to the evaluation
    resolution.
- **PnLCalib in the SoccerMaster checkout** (`data/pnlcalib_utils/`):
  - `FramebyFrameCalib(iwidth, iheight, denormalize=True)`: `.update(kp, lines)`;
    `.heuristic_voting(refine_lines=...)` (3 modes × 6 RANSAC settings) returns
    `{"cam_params": <SN camera + rotation_matrix>, "rep_err": px}`;
    `.get_cam_params(mode, use_ransac, refine_w_lines)` returns `(cam_params, rep_err)` for one
    setting.
  - Keypoint world coordinates are in `utils_calib.keypoint_world_coords_2D` (corner origin,
    105×68; ids 12, 15, 16, 19 are crossbar points at z = −2.44).
  - Decoding:
    - `utils_heatmap.get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1])` (and `_l` for lines,
      two endpoints each) returns a tensor with coordinates at 2× the heatmap size (512).
    - `coords_to_dict(coords, threshold)`, then `complete_keypoints(kp, lines, w=512, h=512,
      normalize=True)` returns `(kp, lines)` normalised.
  - PnLCalib's published inference thresholds are kp 0.3434 and line 0.7867; the earlier
    V.League probe used 0.1. Task 7 tunes between them.
- **Heads:**
  - `KeypointsDetection(backbone_num_channels=[1024], num_keypoints=58, backbone_type="video")`
    returns `pred_keypoints_heatmap` [B, T, 58, 256, 256].
  - The lines head (loaded by `footlytics/geometry/lines_nn.py`) returns `pred_lines_heatmap`
    [B, T, 24, 256, 256].
  - The backbone's temporal embedding is fixed at 30 frames, so a still is fed as 30 repeats.
  - The SoccerMaster checkout lives at `../SoccerMaster-code/codes/SoccerMaster` (3.7 MB). Its
    weights on the M2 are in `weights/soccermaster/`.
  - Cost from the earlier probe: about 5.5 s per frame for full voting, almost all of it in the
    CPU solver.
- **GSR valid comes from only 3 games** (game ids 2, 3, 5: 18 + 21 + 19 clips). GSR pitch
  annotations (`supercategory: "pitch"`) carry `lines` in the SN-Calibration format, per labelled
  frame.
- **Pins:**
  - Quick suite: 60 SN-Calibration valid frames, one per game plus 5, with seed 2026.
  - Quick suite: 6 GSR valid clips, 2 visible clips per game with different actions:
    SNGS-025, 028, 050, 056, 081, 086.
  - Tuning set: every 16th SN-Calibration valid frame from offset 12, minus 00972, which is in
    the quick suite (199 frames). Tuning clips: SNGS-021, SNGS-033.

## Review Focus

- **A frame where the solver raises** (degenerate detections) → counted as missed, never a crashed task; how many is recorded as `solver_errors`. (Task 6 test.)
- **A clip whose first keyframe is rejected** → its frames stay uncalibrated until an accepted keyframe arrives, and are never bridged from nothing. (Task 4 test.)
- **A static camera clip** (overlay detection sees everything as static) → falls back to masking a top band, not the whole frame. (Task 4 test.)
- **A camera with points behind it** (wide shots project parts of a line behind the camera) → those points are dropped from the median px, never wrapped onto the image. (Task 2 test.)
- **The evaluator finds no prediction at all** → score 0 with completeness 0, not NaN in the row. (Task 2 test.)

## File structure

```
footlytics/geometry/sn_camera.py     SoccerNet camera math: rotation, projection, homography <-> camera, line model
footlytics/geometry/calib_nn.py      PnlSolver (PnLCalib decode/solve), SoccerMasterCalibrator (heads), solve_many
footlytics/geometry/clip_calib.py    ClipCalibrator (keyframes + gate + tracker bridge), static_overlay_mask
footlytics/bench/adapters/calib.py   camera -> prediction JSON; GSR pitch lines -> SN-Calibration GT; mirror
footlytics/bench/metrics.py          median_reprojection_px
footlytics/bench/evaluators.py       + run_sncalib / parse_sncalib
footlytics/bench/runner.py           + kinds sncalib, gsr_calibration
footlytics/bench/paths.py            + weights, soccermaster_code, sn_calibration, calib_zip
footlytics/bench/datasets.py         + soccermaster-weights (repo_type model)
footlytics/bench/remote.py, cli.py   + ship_dir for the SoccerMaster checkout
footlytics/bench/host_setup.sh       + calib extra
footlytics/bench/suites/{quick,milestone,calib-tune}.yaml
pyproject.toml                       + [project.optional-dependencies] calib
tests/test_sn_camera.py, test_bench_calib.py, test_calib_nn.py, test_clip_calib.py, + additions to test_bench_{runner,datasets,remote,cli}.py
```

---

### Task 1: SoccerNet camera math

**Files:**
- Create: `footlytics/geometry/sn_camera.py`
- Test: `tests/test_sn_camera.py`

**Interfaces:**
- Produces: `HALF_L, HALF_W`, `rotation(pan, tilt, roll) -> R`, `pan_tilt_roll(R) -> (pan, tilt, roll)`
  (degrees), `make_camera(pan, tilt, roll, focal, position, width, height) -> dict`,
  `intrinsics(cam) -> K`, `project(cam, pts3d) -> (N,2)` (NaN behind the camera),
  `ground_homography(cam) -> H` (pitch xy → image px), `camera_from_homography(H, width, height) -> dict | None`,
  `scale_camera(cam, width) -> dict`, `ground_lines() -> dict[str, ((x,y),(x,y))]`.

- [ ] **Step 1: Write the failing tests**

```python
"""SoccerNet camera convention: rotation, projection, homography <-> camera."""
import numpy as np
import pytest

from footlytics.geometry.sn_camera import (HALF_L, HALF_W, camera_from_homography,
                                           ground_homography, ground_lines, make_camera,
                                           pan_tilt_roll, project, rotation, scale_camera)

BROADCAST = dict(pan=0.0, tilt=80.0, roll=0.0, focal=1500.0, position=(0.0, 55.0, -15.0),
                 width=1920, height=1080)


def cam(**kw):
    a = {**BROADCAST, **kw}
    return make_camera(a["pan"], a["tilt"], a["roll"], a["focal"], a["position"], a["width"],
                       a["height"])


@pytest.mark.parametrize("ptr", [(0, 80, 0), (-25, 72, 1.5), (30, 85, -2)])
def test_rotation_roundtrip(ptr):
    assert pan_tilt_roll(rotation(*ptr)) == pytest.approx(ptr, abs=1e-9)


def test_broadcast_camera_sees_the_pitch_the_right_way_up():
    c = cam()
    centre, far, near, left = project(c, [[0, 0, 0], [0, -HALF_W, 0], [0, HALF_W, 0], [-20, 0, 0]])
    assert centre[0] == pytest.approx(960)
    assert far[1] < near[1]                 # far touchline higher in the image
    assert left[0] < 960                    # left goal on the left
    bar, foot = project(c, [[-HALF_L, 0, -2.44], [-HALF_L, 0, 0]])
    assert bar[1] < foot[1]                 # z negative is up


def test_points_behind_the_camera_are_nan():
    assert np.isnan(project(cam(), [[0, 80, 0]])).all()


def test_ground_homography_matches_projection():
    c = cam(pan=-12, tilt=76, roll=0.8)
    xy = np.random.default_rng(0).uniform([-50, -30], [50, 30], (20, 2))
    H = ground_homography(c)
    h = np.c_[xy, np.ones(len(xy))] @ H.T
    assert h[:, :2] / h[:, 2:] == pytest.approx(project(c, np.c_[xy, np.zeros(len(xy))]), abs=1e-6)


@pytest.mark.parametrize("kw", [{}, dict(pan=-20, tilt=70, roll=1), dict(pan=15, focal=2600,
                                                                     position=(-10, 60, -22))])
def test_camera_from_homography_roundtrip(kw):
    c = cam(**kw)
    back = camera_from_homography(ground_homography(c), c["principal_point"][0] * 2,
                                  c["principal_point"][1] * 2)
    for k in ("pan_degrees", "tilt_degrees", "roll_degrees", "x_focal_length"):
        assert back[k] == pytest.approx(c[k], abs=1e-4)
    assert back["position_meters"] == pytest.approx(c["position_meters"], abs=1e-3)


def test_scale_camera_keeps_angles_and_scales_pixels():
    s = scale_camera(cam(), 960)
    assert s["principal_point"] == [480.0, 270.0] and s["x_focal_length"] == 750.0
    assert s["pan_degrees"] == cam()["pan_degrees"]


def test_ground_lines_cover_the_ground_plane_classes():
    lines = ground_lines()
    assert len(lines) == 17 and "Big rect. left top" in lines and "Circle central" not in lines
    pts = np.array([p for seg in lines.values() for p in seg])
    assert (np.abs(pts[:, 0]) <= HALF_L).all() and (np.abs(pts[:, 1]) <= HALF_W).all()
    assert lines["Side line top"][0][1] == -HALF_W     # top = far = y negative
```

- [ ] **Step 2: Run to verify failure:** `uv run --extra dev pytest tests/test_sn_camera.py -q`. Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement `footlytics/geometry/sn_camera.py`**

```python
"""SoccerNet camera parameters: the full pinhole camera that SN-Calibration scores.

World frame (SoccerNet's; GSR and PnLCalib use it too): metres from the centre spot, x along
the length, y negative towards the far ("top") touchline, z negative UP (a crossbar sits at
z = -2.44). A world point X goes to the camera frame as R (X - position), with
R = (Rz(pan) Rx(tilt) Rz(roll))^T, then to pixels through the focal lengths and principal point.
Distortion is always zero in what we write.

Our implementation of the published convention; the evaluator's code is never imported.
"""
from __future__ import annotations

import numpy as np

HALF_L, HALF_W = 52.5, 34.0
_PA_DEPTH, _PA_HALF_W = 16.5, 20.16
_GA_DEPTH, _GA_HALF_W = 5.5, 9.16


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation(pan: float, tilt: float, roll: float) -> np.ndarray:
    """World -> camera rotation from SoccerNet's pan/tilt/roll in degrees."""
    p, t, r = np.radians([pan, tilt, roll])
    return (_rz(p) @ _rx(t) @ _rz(r)).T


def pan_tilt_roll(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of `rotation`; of the two solutions, the one with less roll (operators keep it
    near zero), as SoccerNet does."""
    o = np.asarray(R).T
    best = None
    t0 = float(np.arccos(np.clip(o[2, 2], -1.0, 1.0)))
    for tilt in (t0, -t0):
        s = 1.0 if np.sin(tilt) > 0 else -1.0
        pan = np.arctan2(s * o[0, 2], -s * o[1, 2])
        roll = np.arctan2(s * o[2, 0], s * o[2, 1])
        if best is None or abs(roll) < abs(best[2]):
            best = (pan, tilt, roll)
    return tuple(float(np.degrees(v)) for v in best)


def make_camera(pan: float, tilt: float, roll: float, focal: float, position, width: float,
                height: float) -> dict:
    """A SoccerNet camera dict, as the evaluator reads it."""
    return {"pan_degrees": float(pan), "tilt_degrees": float(tilt), "roll_degrees": float(roll),
            "position_meters": [float(v) for v in position],
            "x_focal_length": float(focal), "y_focal_length": float(focal),
            "principal_point": [float(width) / 2, float(height) / 2],
            "radial_distortion": [0.0] * 6, "tangential_distortion": [0.0] * 2,
            "thin_prism_distortion": [0.0] * 4}


def intrinsics(cam: dict) -> np.ndarray:
    cx, cy = cam["principal_point"]
    return np.array([[cam["x_focal_length"], 0.0, cx], [0.0, cam["y_focal_length"], cy],
                     [0.0, 0.0, 1.0]])


def _pose(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    R = rotation(cam["pan_degrees"], cam["tilt_degrees"], cam["roll_degrees"])
    return R, np.asarray(cam["position_meters"], float)


def project(cam: dict, pts3d) -> np.ndarray:
    """World points (N, 3) -> pixels (N, 2); NaN for points at or behind the camera."""
    R, C = _pose(cam)
    X = (np.asarray(pts3d, float).reshape(-1, 3) - C) @ R.T
    out = np.full((len(X), 2), np.nan)
    ok = X[:, 2] > 1e-3
    K = intrinsics(cam)
    uv = X[ok] @ K.T
    out[ok] = uv[:, :2] / uv[:, 2:]
    return out


def ground_homography(cam: dict) -> np.ndarray:
    """Pitch plane (x, y, z=0) -> image pixels."""
    R, C = _pose(cam)
    return intrinsics(cam) @ np.column_stack([R[:, 0], R[:, 1], -R @ C])


def camera_from_homography(H: np.ndarray, width: float, height: float) -> dict | None:
    """A camera from a pitch -> image homography, assuming square pixels and the principal
    point at the image centre (as PnLCalib does). None when the homography admits no camera."""
    cx, cy = width / 2, height / 2
    M = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]]) @ np.asarray(H, float)
    cands = []                       # (|denominator|, f^2) from the two orthonormality constraints
    d1 = M[2, 0] * M[2, 1]
    if abs(d1) > 1e-12:
        cands.append((abs(d1), -(M[0, 0] * M[0, 1] + M[1, 0] * M[1, 1]) / d1))
    d2 = M[2, 0] ** 2 - M[2, 1] ** 2
    if abs(d2) > 1e-12:
        cands.append((abs(d2), (M[0, 1] ** 2 + M[1, 1] ** 2 - M[0, 0] ** 2 - M[1, 0] ** 2) / d2))
    cands = [c for c in cands if c[1] > 0]
    if not cands:
        return None
    f = float(np.sqrt(max(cands)[1]))
    A = np.diag([1 / f, 1 / f, 1.0]) @ M
    lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
    if t[2] < 0:                     # the pitch must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    pan, tilt, roll = pan_tilt_roll(R)
    return make_camera(pan, tilt, roll, f, -R.T @ t, width, height)


def scale_camera(cam: dict, width: float) -> dict:
    """The same camera at another image width (the image size is 2 x principal point)."""
    k = width / (2 * cam["principal_point"][0])
    return {**cam, "x_focal_length": cam["x_focal_length"] * k,
            "y_focal_length": cam["y_focal_length"] * k,
            "principal_point": [cam["principal_point"][0] * k, cam["principal_point"][1] * k]}


def ground_lines() -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """The straight ground-plane line classes of SN-Calibration, as world segments."""
    L, W = HALF_L, HALF_W
    lines = {"Side line top": ((-L, -W), (L, -W)), "Side line bottom": ((-L, W), (L, W)),
             "Side line left": ((-L, -W), (-L, W)), "Side line right": ((L, -W), (L, W)),
             "Middle line": ((0.0, -W), (0.0, W))}
    for side, s in (("left", -1), ("right", 1)):
        for name, depth, hw in (("Big rect.", _PA_DEPTH, _PA_HALF_W),
                                ("Small rect.", _GA_DEPTH, _GA_HALF_W)):
            x0, x1 = s * L, s * (L - depth)
            lines[f"{name} {side} top"] = ((x0, -hw), (x1, -hw))
            lines[f"{name} {side} main"] = ((x1, -hw), (x1, hw))
            lines[f"{name} {side} bottom"] = ((x0, hw), (x1, hw))
    return lines
```

- [ ] **Step 4: Run:** expect PASS.
- [ ] **Step 5: Commit:** `git add footlytics/geometry/sn_camera.py tests/test_sn_camera.py && git commit -m "feat(geometry): SoccerNet camera math, homography <-> camera"`

---

### Task 2: Calibration adapter, median px metric, SN-Calibration evaluator wrapper

**Files:**
- Create: `footlytics/bench/adapters/calib.py`, `footlytics/bench/metrics.py`
- Modify: `footlytics/bench/evaluators.py` (append)
- Test: `tests/test_bench_calib.py`

**Interfaces:**
- Consumes: `sn_camera.{project, scale_camera, ground_lines}`.
- Produces: `adapters.calib.camera_json(cam) -> dict`, `write_camera(path, cam)`,
  `mirror(lines) -> dict`, `gsr_line_frames(labels) -> dict[int, dict]` (0-based frame →
  lines, labelled frames only); `metrics.median_reprojection_px(gt_lines, cam, width=960,
  height=540) -> float | None`; `evaluators.CalibScores(score, completeness, jac, precision,
  recall)` (percent), `parse_sncalib(text) -> CalibScores`,
  `run_sncalib(python, sncal_dir, gt_root, pred_root, split, out_dir, threshold=5, width=960, height=540, run=subprocess.run) -> CalibScores`.

- [ ] **Step 1: Write the failing tests**

```python
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
from footlytics.geometry.sn_camera import ground_lines, make_camera, project

CAM = make_camera(0, 80, 0, 1500, (0, 55, -15), 1920, 1080)


def labels_for(cam, width=1920, height=1080, n=12):
    """Ground-truth lines as an annotator would draw them: points along each visible line."""
    out = {}
    for name, (a, b) in ground_lines().items():
        w = np.linspace(0, 1, n)[:, None]
        pts = project(cam, np.c_[(1 - w) * a + w * np.array(b), np.zeros(n)])
        ok = np.isfinite(pts).all(1) & (pts[:, 0] >= 0) & (pts[:, 0] < width) & \
            (pts[:, 1] >= 0) & (pts[:, 1] < height)
        if ok.sum() >= 2:
            out[name] = [{"x": float(x / width), "y": float(y / height)} for x, y in pts[ok]]
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
    write_camera(pred_dir / "camera_00000.json", CAM)
    s_true = run_sncalib(Path(EVAL_PY), Path(SC), tmp_path / "gt", tmp_path / "pred", "valid",
                         tmp_path / "o1")
    assert s_true.completeness == 50.0 and s_true.jac > 0
    write_camera(pred_dir / "camera_00001.json", off)
    s_both = run_sncalib(Path(EVAL_PY), Path(SC), tmp_path / "gt", tmp_path / "pred", "valid",
                         tmp_path / "o2")
    assert s_both.completeness == 100.0 and s_both.jac < s_true.jac
```

- [ ] **Step 2: Run to verify failure:** `uv run --extra dev pytest tests/test_bench_calib.py -q`. Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`footlytics/bench/adapters/calib.py`:
```python
"""Our cameras -> SN-Calibration prediction files; GSR pitch lines -> SN-Calibration labels.

GSR pitch annotations already use the SN-Calibration line format (line name -> points
normalised to [0, 1]), so a GSR clip becomes calibration ground truth frame by frame.
"""
from __future__ import annotations

import json
from pathlib import Path

CAMERA_KEYS = ("pan_degrees", "tilt_degrees", "roll_degrees", "position_meters",
               "x_focal_length", "y_focal_length", "principal_point", "radial_distortion",
               "tangential_distortion", "thin_prism_distortion")


def _plain(v):
    if isinstance(v, (list, tuple)) or hasattr(v, "tolist"):
        return [_plain(x) for x in (v.tolist() if hasattr(v, "tolist") else v)]
    return float(v)


def camera_json(cam: dict) -> dict:
    return {k: _plain(cam[k]) for k in CAMERA_KEYS}


def write_camera(path: str | Path, cam: dict) -> None:
    Path(path).write_text(json.dumps(camera_json(cam)))


_SWAP = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


def mirror(lines: dict) -> dict:
    """Labels seen through the centre spot: left <-> right, top <-> bottom."""
    return {" ".join(_SWAP.get(w, w) for w in name.split(" ")): v for name, v in lines.items()}


def gsr_line_frames(labels: dict) -> dict[int, dict]:
    """0-based frame -> SN-Calibration lines, for frames whose pitch is labelled."""
    frame_of = {img["image_id"]: int(Path(img["file_name"]).stem) - 1
                for img in labels["images"] if img.get("has_labeled_pitch")}
    out = {}
    for a in labels["annotations"]:
        if a.get("supercategory") == "pitch" and a["image_id"] in frame_of:
            lines = {k: v for k, v in (a.get("lines") or {}).items() if v}
            if lines:
                out[frame_of[a["image_id"]]] = lines
    return out
```

`footlytics/bench/metrics.py`:
```python
"""Our own auxiliary metrics. Official scores come from the evaluators (evaluators.py)."""
from __future__ import annotations

import numpy as np

from ..geometry.sn_camera import ground_lines, project, scale_camera
from .adapters.calib import mirror

_SAMPLES = 60


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _distances(lines: dict, cam: dict, width: int, height: int) -> list[float]:
    model = ground_lines()
    out: list[float] = []
    for name, pts in lines.items():
        if name not in model or not pts:
            continue
        a, b = (np.array(p) for p in model[name])
        w = np.linspace(0, 1, _SAMPLES)[:, None]
        poly = project(cam, np.c_[(1 - w) * a + w * b, np.zeros(_SAMPLES)])
        poly = poly[np.isfinite(poly).all(1)]           # drop the part behind the camera
        if len(poly) < 2:
            continue
        gt = np.array([[q["x"] * (width - 1), q["y"] * (height - 1)] for q in pts])
        d = np.min([_seg_dist(gt, poly[i], poly[i + 1]) for i in range(len(poly) - 1)], axis=0)
        out.extend(d.tolist())
    return out


def median_reprojection_px(gt_lines: dict, cam: dict, width: int = 960,
                           height: int = 540) -> float | None:
    """Median distance (px at width x height) from each labelled point of a straight
    ground-plane line to the camera's projection of that line, taking the better of the labels
    and their mirror as the official evaluator does. None if no such line is labelled."""
    c = scale_camera(cam, width)
    meds = [float(np.median(d)) for d in (_distances(gt_lines, c, width, height),
                                          _distances(mirror(gt_lines), c, width, height)) if d]
    return round(min(meds), 3) if meds else None
```

Append to `footlytics/bench/evaluators.py` (and add `import os`, `import re` at the top):
```python
@dataclass
class CalibScores:
    score: float          # JaC x completeness, percent
    completeness: float
    jac: float
    precision: float
    recall: float


_CALIB_RE = {"score": r"final score of : (\S+)", "completeness": r"completeness rate of : (\S+)",
             "jac": r"accuracy mean value :\s+(\S+)%", "precision": r"precision mean value : (\S+)%",
             "recall": r"recall mean value : (\S+)%"}


def parse_sncalib(text: str) -> CalibScores:
    vals = {}
    for k, pat in _CALIB_RE.items():
        m = re.search(pat, text)
        v = float(m.group(1)) if m else float("nan")
        if k in ("score", "completeness"):
            v *= 100
        vals[k] = 0.0 if v != v else round(v, 3)     # nothing predicted: 0, never NaN
    return CalibScores(**vals)


def run_sncalib(python: Path, sncal_dir: Path, gt_root: Path, pred_root: Path, split: str,
                out_dir: Path, threshold: int = 5, width: int = 960, height: int = 540,
                run=subprocess.run) -> CalibScores:
    """JaC@threshold x completeness from sn-calibration's evaluate_camera.py.

    `gt_root/<split>/` must hold only frame JSONs: the script counts every *.json in it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(python), str(Path(sncal_dir) / "src" / "evaluate_camera.py"), "-s", str(gt_root),
           "-p", str(pred_root), "--split", split, "-t", str(threshold),
           "--resolution_width", str(width), "--resolution_height", str(height)]
    env = {**os.environ, "PYTHONPATH": str(sncal_dir), "MPLBACKEND": "Agg"}
    p = run(cmd, capture_output=True, text=True, cwd=str(sncal_dir), env=env)
    (out_dir / "evaluator.log").write_text(f"{' '.join(cmd)}\n\n{p.stdout}\n{p.stderr}")
    tail = "\n".join((p.stdout + "\n" + p.stderr).strip().splitlines()[-15:])
    if p.returncode != 0 or "final score of" not in p.stdout:
        raise EvaluatorError(f"sn-calibration exited {p.returncode}:\n{tail}")
    return parse_sncalib(p.stdout)
```

- [ ] **Step 4: Run:** unit tests PASS. Integration: in the scratchpad, `VIRTUAL_ENV=<scratchpad>/te-venv uv pip install opencv-python-headless tqdm matplotlib`, then
  `FOOTLYTICS_SNCALIB=<scratchpad>/ext/sn-calibration FOOTLYTICS_EVAL_PYTHON=<scratchpad>/te-venv/bin/python uv run --extra dev pytest tests/test_bench_calib.py -q`. Expected: all PASS.
- [ ] **Step 5: Commit:** `git add footlytics/bench/adapters/calib.py footlytics/bench/metrics.py footlytics/bench/evaluators.py tests/test_bench_calib.py && git commit -m "feat(bench): SN-Calibration adapter, evaluator wrapper, median px"`

---

### Task 3: Calibrator (PnLCalib solver + SoccerMaster heads)

**Files:**
- Create: `footlytics/geometry/calib_nn.py`
- Modify: `pyproject.toml` (`[project.optional-dependencies] calib = ["transformers>=4.50", "accelerate>=1.0", "einops>=0.8", "timm>=1.0", "shapely>=2.0"]`), `uv.lock`
- Test: `tests/test_calib_nn.py`

**Interfaces:**
- Consumes: `lines_nn.SoccerMasterLines`, `lines_nn.NUM_FRAMES`, `sn_camera.{ground_homography, HALF_L, HALF_W}`.
- Produces: `CameraFit(cam, rep_err, H_img_to_pitch)`, `DEFAULT_KP_THRESHOLD`, `DEFAULT_LINE_THRESHOLD`,
  `PnlSolver(code_dir)` with `.keypoint_world(kp_id)`, `.decode(kp_hm, line_hm, kp_threshold, line_threshold) -> list[(kp, lines)]`,
  `.solve(kp, lines, width, height, voting=True, mode="full", ransac=0.0, refine_lines=True) -> CameraFit | None`;
  `solve_many(code_dir, jobs, workers=1, **solve_kw) -> list[(CameraFit | None, error | None)]`;
  `SoccerMasterCalibrator(code_dir, weights_dir, device=None)` with `.heatmaps(frames_rgb)`,
  `.decode(frames_rgb, kp_threshold, line_threshold)`, `.fit(frames_rgb, which=None, kp_threshold, line_threshold, **solve_kw) -> dict[int, CameraFit | None]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Calibrator: PnLCalib's solver recovers a known camera; the heads calibrate a real frame."""
import io
import json
import os
import zipfile

import numpy as np
import pytest

from footlytics.geometry.sn_camera import make_camera, project

CODE = os.getenv("FOOTLYTICS_SOCCERMASTER_CODE")
WEIGHTS = os.getenv("FOOTLYTICS_SOCCERMASTER_WEIGHTS")
SNCAL_ZIP = os.getenv("FOOTLYTICS_SNCAL_ZIP")
needs_code = pytest.mark.skipif(not CODE, reason="set FOOTLYTICS_SOCCERMASTER_CODE")


def synthetic_keypoints(solver, cam, width, height):
    kp = {}
    for i in range(1, 31):
        uv = project(cam, [solver.keypoint_world(i)])[0]
        if np.isfinite(uv).all() and 0 <= uv[0] < width and 0 <= uv[1] < height:
            kp[i] = {"x": uv[0] / width, "y": uv[1] / height, "p": 1.0}
    return kp


@needs_code
@pytest.mark.parametrize("voting", [True, False])
def test_solver_recovers_a_known_camera(voting):
    from footlytics.geometry.calib_nn import PnlSolver

    solver = PnlSolver(CODE)
    cam = make_camera(-18, 77, 0.5, 1900, (-8, 52, -16), 1920, 1080)
    kp = synthetic_keypoints(solver, cam, 1920, 1080)
    assert len(kp) >= 6
    fit = solver.solve(kp, {}, 1920, 1080, voting=voting, refine_lines=False)
    assert fit is not None and fit.rep_err < 2.0
    for k, tol in (("pan_degrees", 0.5), ("tilt_degrees", 0.5)):
        assert fit.cam[k] == pytest.approx(cam[k], abs=tol)
    assert fit.cam["x_focal_length"] == pytest.approx(1900, rel=0.03)
    assert "rotation_matrix" not in fit.cam
    assert np.isfinite(fit.H_img_to_pitch @ np.array([960.0, 540.0, 1.0])).all()


@needs_code
def test_solve_many_reports_each_job():
    from footlytics.geometry.calib_nn import PnlSolver, solve_many

    solver = PnlSolver(CODE)
    cam = make_camera(10, 78, 0, 1700, (5, 54, -15), 1920, 1080)
    good = (synthetic_keypoints(solver, cam, 1920, 1080), {}, 1920, 1080)
    out = solve_many(CODE, [good, ({}, {}, 1920, 1080)], workers=1, refine_lines=False)
    assert out[0][0] is not None and out[0][1] is None
    assert out[1] == (None, None)                    # no keypoints: no camera, not an error


@pytest.mark.skipif(not (CODE and WEIGHTS and SNCAL_ZIP),
                    reason="set FOOTLYTICS_SOCCERMASTER_CODE/_WEIGHTS and FOOTLYTICS_SNCAL_ZIP")
def test_calibrates_a_real_sn_calibration_frame():
    import cv2

    from footlytics.bench.metrics import median_reprojection_px
    from footlytics.geometry.calib_nn import SoccerMasterCalibrator

    z = zipfile.ZipFile(SNCAL_ZIP)
    img = cv2.imdecode(np.frombuffer(z.read("valid/00007.jpg"), np.uint8), cv2.IMREAD_COLOR)
    gt = json.loads(z.read("valid/00007.json"))
    cal = SoccerMasterCalibrator(CODE, WEIGHTS)
    fit = cal.fit([img[:, :, ::-1]])[0]
    assert fit is not None
    assert median_reprojection_px(gt, fit.cam) < 15.0
```

- [ ] **Step 2: Run to verify failure:** on the M2,
  `FOOTLYTICS_SOCCERMASTER_CODE="../SoccerMaster-code/codes/SoccerMaster" uv run --extra dev --extra calib pytest tests/test_calib_nn.py -q`.
  Expected: FAIL (`ModuleNotFoundError: footlytics.geometry.calib_nn`).

- [ ] **Step 3: Implement `footlytics/geometry/calib_nn.py`**

```python
"""Per-frame camera calibration: SoccerMaster's keypoint and line heads, solved by PnLCalib.

One backbone pass over up to 30 consecutive frames feeds both heads (`SoccerMasterLines`
loads the backbone and the lines head; this adds the keypoints head). Heatmaps are decoded and
solved with PnLCalib's code as vendored in the SoccerMaster repository. That repository has no
licence (internal R&D), so it is imported from `code_dir` at runtime and never copied here. The
output is the full SoccerNet camera (`sn_camera`), which SN-Calibration scores, plus the
image->pitch homography the pipeline uses.
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .sn_camera import HALF_L, HALF_W, ground_homography

NUM_KEYPOINTS = 58                 # 57 pitch keypoints + background
#: PnLCalib's published inference thresholds (tuned in the benchmark, plan 2 task 7)
DEFAULT_KP_THRESHOLD = 0.3434
DEFAULT_LINE_THRESHOLD = 0.7867
_CROSSBAR = {12, 15, 16, 19}       # keypoints 2.44 m up


@dataclass
class CameraFit:
    cam: dict                      # SoccerNet camera at the source resolution
    rep_err: float                 # solver's keypoint reprojection error, source px
    H_img_to_pitch: np.ndarray


def _plain(v):
    if isinstance(v, (list, tuple)) or hasattr(v, "tolist"):
        return [_plain(x) for x in (v.tolist() if hasattr(v, "tolist") else v)]
    return float(v)


class PnlSolver:
    """PnLCalib's decoding and solving, imported from the SoccerMaster checkout."""

    def __init__(self, code_dir: str | Path):
        code_dir = str(Path(code_dir))
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from data.pnlcalib_utils import utils_calib, utils_heatmap

        self._calib, self._hm = utils_calib, utils_heatmap

    def keypoint_world(self, kp_id: int) -> tuple[float, float, float]:
        """PnLCalib keypoint `kp_id` (1-based) in SoccerNet world metres."""
        c = self._calib
        wp = (c.keypoint_world_coords_2D[kp_id - 1] if kp_id <= 57
              else c.keypoint_aux_world_coords_2D[kp_id - 58])
        return wp[0] - HALF_L, wp[1] - HALF_W, -2.44 if kp_id in _CROSSBAR else 0.0

    def decode(self, kp_hm, line_hm, kp_threshold: float = DEFAULT_KP_THRESHOLD,
               line_threshold: float = DEFAULT_LINE_THRESHOLD) -> list[tuple[dict, dict]]:
        """Heatmaps [T, 58, h, w] and [T, 24, h, w] (torch) -> per frame (keypoints, lines),
        normalised to [0, 1]."""
        hm = self._hm
        kp = hm.coords_to_dict(hm.get_keypoints_from_heatmap_batch_maxpool(kp_hm[:, :-1]),
                               threshold=kp_threshold)
        ln = hm.coords_to_dict(hm.get_keypoints_from_heatmap_batch_maxpool_l(line_hm[:, :-1]),
                               threshold=line_threshold)
        side = 2 * kp_hm.shape[-1]          # coordinates come back at twice the heatmap size
        return [hm.complete_keypoints(k, l, w=side, h=side, normalize=True)
                for k, l in zip(kp, ln)]

    def solve(self, keypoints: dict, lines: dict, width: int, height: int, voting: bool = True,
              mode: str = "full", ransac: float = 0.0,
              refine_lines: bool = True) -> Optional[CameraFit]:
        """A camera from normalised detections; `voting` runs PnLCalib's 18-setting search,
        otherwise the single (`mode`, `ransac`) setting. None when nothing fits."""
        c = self._calib.FramebyFrameCalib(iwidth=width, iheight=height, denormalize=True)
        c.update(copy.deepcopy(keypoints), copy.deepcopy(lines))
        if voting:
            res = c.heuristic_voting(refine_lines=refine_lines)
            cam, err = (res["cam_params"], res["rep_err"]) if res else (None, None)
        else:
            cam, err = c.get_cam_params(mode=mode, use_ransac=ransac, refine_w_lines=refine_lines)
        if cam is None or err is None or not np.isfinite(err):
            return None
        cam = {k: _plain(v) for k, v in cam.items() if k != "rotation_matrix"}
        H = np.linalg.inv(ground_homography(cam))
        return CameraFit(cam, float(err), H / H[2, 2])


_WORKER: dict = {}


def _solve_job(args) -> tuple[Optional[CameraFit], Optional[str]]:
    code_dir, kp, ln, w, h, kw = args
    if _WORKER.get("code_dir") != code_dir:
        _WORKER.update(code_dir=code_dir, solver=PnlSolver(code_dir))
    try:
        return _WORKER["solver"].solve(kp, ln, w, h, **kw), None
    except Exception as e:  # noqa: BLE001 - degenerate detections; the caller counts them
        return None, f"{type(e).__name__}: {e}"


def solve_many(code_dir: str | Path, jobs: Sequence[tuple[dict, dict, int, int]],
               workers: int = 1, **solve_kw) -> list[tuple[Optional[CameraFit], Optional[str]]]:
    """Solve (keypoints, lines, width, height) jobs, on `workers` CPU processes: the solver,
    not the network, is the slow part (about 5 s a frame with voting)."""
    args = [(str(code_dir), k, l, w, h, solve_kw) for k, l, w, h in jobs]
    if workers <= 1:
        return [_solve_job(a) for a in args]
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(_solve_job, args, chunksize=1)


class SoccerMasterCalibrator:
    """Backbone + lines head + keypoints head, decoded and solved per frame."""

    def __init__(self, code_dir: str | Path, weights_dir: str | Path,
                 device: Optional[str] = None):
        import torch

        from .lines_nn import SoccerMasterLines

        self.code_dir = Path(code_dir)
        self.lines = SoccerMasterLines(code_dir, weights_dir, device=device)
        self.solver = PnlSolver(code_dir)
        from models.keypoints_detection import KeypointsDetection

        self.kp_head = KeypointsDetection(backbone_num_channels=[1024],
                                          num_keypoints=NUM_KEYPOINTS, backbone_type="video")
        state = torch.load(Path(weights_dir) / "KeypointsDetection.pt", map_location="cpu",
                           weights_only=True)
        res = self.kp_head.load_state_dict(state, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"keypoints head is missing {len(res.missing_keys)} weights, "
                               f"e.g. {res.missing_keys[:3]}")
        self.kp_head.to(device=self.lines.device, dtype=self.lines.dtype).eval()

    def heatmaps(self, frames_rgb: Sequence[np.ndarray]):
        """Up to 30 consecutive RGB frames (a still is repeated) -> (kp, lines) heatmaps on CPU."""
        import torch

        from .lines_nn import NUM_FRAMES

        frames = list(frames_rgb)
        if not 0 < len(frames) <= NUM_FRAMES:
            raise ValueError(f"pass 1 to {NUM_FRAMES} frames, got {len(frames)}")
        n = len(frames)
        x = self.lines._preprocess(frames + [frames[-1]] * (NUM_FRAMES - n))
        with torch.no_grad():
            local, pooled = self.lines.backbone(x)
            feats = {"global_features": pooled, "local_features": local}
            kp = self.kp_head(feats, metas=None)["pred_keypoints_heatmap"][0, :n]
            ln = self.lines.head(feats, metas=None)["pred_lines_heatmap"][0, :n]
        return kp.float().cpu(), ln.float().cpu()

    def decode(self, frames_rgb: Sequence[np.ndarray], kp_threshold: float = DEFAULT_KP_THRESHOLD,
               line_threshold: float = DEFAULT_LINE_THRESHOLD) -> list[tuple[dict, dict]]:
        kp, ln = self.heatmaps(frames_rgb)
        return self.solver.decode(kp, ln, kp_threshold, line_threshold)

    def fit(self, frames_rgb: Sequence[np.ndarray], which: Optional[Sequence[int]] = None,
            kp_threshold: float = DEFAULT_KP_THRESHOLD,
            line_threshold: float = DEFAULT_LINE_THRESHOLD,
            **solve_kw) -> dict[int, Optional[CameraFit]]:
        """Heatmaps for all frames (temporal context), cameras for `which` (default all).
        A frame whose detections the solver cannot use gets None."""
        frames = list(frames_rgb)
        h, w = frames[0].shape[:2]
        kp, ln = self.heatmaps(frames)
        which = list(range(len(frames)) if which is None else which)
        if not which:
            return {}
        decoded = self.solver.decode(kp[which], ln[which], kp_threshold, line_threshold)
        out: dict[int, Optional[CameraFit]] = {}
        for i, (k, l) in zip(which, decoded):
            try:
                out[i] = self.solver.solve(k, l, w, h, **solve_kw)
            except Exception:  # noqa: BLE001 - degenerate detections: no camera this frame
                out[i] = None
        return out
```

Add the `calib` extra to `pyproject.toml`, then run `uv lock`.

- [ ] **Step 4: Run:** with the env var set, the solver tests PASS on the M2; the real-frame test is skipped (no SN-Calibration zip on the M2) and runs on the host in Task 7.
- [ ] **Step 5: Commit:** `git add footlytics/geometry/calib_nn.py tests/test_calib_nn.py pyproject.toml uv.lock && git commit -m "feat(geometry): full-camera calibrator from SoccerMaster heads + PnLCalib"`

---

### Task 4: Clip calibrator (keyframes, gate, tracker bridge, overlay mask)

**Files:**
- Create: `footlytics/geometry/clip_calib.py`
- Test: `tests/test_clip_calib.py`

**Interfaces:**
- Consumes: `calib_nn.CameraFit`, `sn_camera.camera_from_homography`, `camera_motion.AnchoredCamera` (constructor `(anchor_bgr, mask=None, ...)`, `.update(frame_bgr, mask) -> H_anchor_to_t | None`).
- Produces: `FrameCalib(frame_idx, H_img_to_pitch, cam, source, rep_err)`;
  `static_overlay_mask(frames_bgr, std_thresh=3.0, dilate_px=9, max_share=0.25, band=0.12) -> uint8 mask (255 usable)`;
  `ClipCalibrator(fit_keyframes, every=25, gate_px=25.0, tracker_factory=AnchoredCamera, chunk=30)`
  with `.run(frames_bgr, mask=None) -> list[FrameCalib]` and counters `.keyframes`, `.keyframes_accepted`.
  `fit_keyframes(chunk_bgr: list[np.ndarray], keys: list[int]) -> dict[int, CameraFit | None]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Clip calibration: keyframes, the gate, bridging, and the overlay mask."""
import numpy as np
import pytest

from footlytics.geometry.calib_nn import CameraFit
from footlytics.geometry.clip_calib import ClipCalibrator, static_overlay_mask
from footlytics.geometry.sn_camera import ground_homography, make_camera

# a broadcast camera scaled to tiny 16x9 frames, so the bridged camera is recoverable
CAM = make_camera(0, 80, 0, 12.5, (0, 55, -15), 16, 9)
H0 = np.linalg.inv(ground_homography(CAM))
H0 /= H0[2, 2]
STEP = 0.01                                            # px of pan per frame


def frame(t):
    f = np.zeros((9, 16, 3), np.uint8)
    f[0, 0, 0] = t
    return f


def shift(dx):
    return np.array([[1.0, 0, dx * STEP], [0, 1, 0], [0, 0, 1]])


class FakeTracker:
    """Anchor->t is a small pan of (t - t_anchor) steps; frames in LOST are untrackable."""
    LOST: set = set()

    def __init__(self, anchor, mask=None):
        self.t0 = int(anchor[0, 0, 0])

    def update(self, f, mask=None):
        t = int(f[0, 0, 0])
        return None if t in self.LOST else shift(t - self.t0)


def fits(rep_errs):
    def fit_keyframes(chunk, keys):
        return {k: (None if rep_errs.get(int(chunk[k][0, 0, 0])) is None
                    else CameraFit(CAM, rep_errs[int(chunk[k][0, 0, 0])], H0)) for k in keys}
    return fit_keyframes


def run(rep_errs, n=60, lost=(), every=25):
    FakeTracker.LOST = set(lost)
    cc = ClipCalibrator(fits(rep_errs), every=every, gate_px=25.0, tracker_factory=FakeTracker,
                        chunk=30)
    return cc, cc.run(frame(t) for t in range(n))


def test_keyframes_anchor_and_frames_between_are_bridged():
    cc, out = run({0: 4.0, 25: 6.0, 50: 5.0})
    assert [o.source for o in out[:3]] == ["keyframe", "bridged", "bridged"]
    assert np.allclose(out[3].H_img_to_pitch, H0 @ np.linalg.inv(shift(3)) /
                       (H0 @ np.linalg.inv(shift(3)))[2, 2])
    assert out[3].cam is not None                       # camera recovered from the homography
    assert cc.keyframes == 3 and cc.keyframes_accepted == 3


def test_a_keyframe_over_the_gate_is_rejected_and_bridging_continues():
    cc, out = run({0: 4.0, 25: 40.0, 50: 5.0})
    assert out[25].source == "bridged" and out[25].rep_err == 40.0
    assert out[30].source == "bridged"
    assert cc.keyframes_accepted == 2


def test_frames_before_the_first_accepted_keyframe_stay_uncalibrated():
    _, out = run({0: 99.0, 25: 5.0, 50: 5.0})
    assert all(o.source == "none" and o.H_img_to_pitch is None for o in out[:25])
    assert out[25].source == "keyframe"


def test_a_lost_track_leaves_frames_uncalibrated_until_the_next_keyframe():
    _, out = run({0: 4.0, 25: None, 50: 5.0}, lost=range(10, 50))
    assert out[9].source == "bridged" and out[10].source == "none" and out[49].source == "none"
    assert out[50].source == "keyframe"


def test_overlay_mask_finds_a_static_logo_on_a_moving_picture():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (120, 200), np.uint8)
    frames = []
    for t in range(20):
        g = np.roll(base, 3 * t, axis=1)
        g[5:20, 5:40] = 200                           # a score bug that never moves
        frames.append(np.dstack([g, g, g]))
    m = static_overlay_mask(frames)
    assert (m[5:20, 5:40] == 0).all()
    assert (m == 255).mean() > 0.8


def test_overlay_mask_on_a_static_camera_falls_back_to_a_top_band():
    f = np.random.default_rng(1).integers(0, 255, (100, 100, 3), np.uint8)
    m = static_overlay_mask([f] * 10)
    assert (m[:12] == 0).all() and (m[12:] == 255).all()
```

- [ ] **Step 2: Run to verify failure:** `uv run --extra dev pytest tests/test_clip_calib.py -q`. Expected: FAIL.

- [ ] **Step 3: Implement `footlytics/geometry/clip_calib.py`**

```python
"""Calibrate every frame of a broadcast clip: solve on keyframes, bridge between them by tracking.

A solve costs about a second (5.5 s with PnLCalib's full voting), so only every `every`-th
frame is solved, with one solver setting. A solve whose reprojection error exceeds `gate_px` is
rejected: on the V.League clip the good windows were at 3-24 px and the bad ones above 40.
Frames between accepted keyframes are related to the last one by `AnchoredCamera`. The
broadcast's static overlays (score bug, channel logo) are masked out of its feature search,
because features that never move would pin the motion estimate to zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

from .calib_nn import CameraFit
from .camera_motion import AnchoredCamera
from .sn_camera import camera_from_homography


@dataclass
class FrameCalib:
    frame_idx: int
    H_img_to_pitch: Optional[np.ndarray]
    cam: Optional[dict]
    source: str                      # "keyframe" | "bridged" | "none"
    rep_err: Optional[float] = None


def static_overlay_mask(frames_bgr: list[np.ndarray], std_thresh: float = 3.0,
                        dilate_px: int = 9, max_share: float = 0.25,
                        band: float = 0.12) -> np.ndarray:
    """255 where features may be tracked, 0 on pixels that do not change while the picture
    does. If more than `max_share` of the frame is static the camera itself is still and the
    test says nothing; then only the top `band` (where broadcasters put overlays) is masked."""
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]).astype(np.float32)
    static = g.std(axis=0) < std_thresh
    h = g.shape[1]
    mask = np.full(g.shape[1:], 255, np.uint8)
    if static.mean() > max_share:
        mask[: int(round(band * h))] = 0
        return mask
    k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    mask[cv2.dilate(static.astype(np.uint8), k) > 0] = 0
    return mask


class ClipCalibrator:
    def __init__(self, fit_keyframes: Callable[[list, list], dict[int, Optional[CameraFit]]],
                 every: int = 25, gate_px: float = 25.0,
                 tracker_factory: Callable = AnchoredCamera, chunk: int = 30):
        self.fit_keyframes, self.every, self.gate_px = fit_keyframes, every, gate_px
        self.tracker_factory, self.chunk = tracker_factory, chunk
        self.keyframes = self.keyframes_accepted = 0

    def run(self, frames_bgr: Iterable[np.ndarray],
            mask: Optional[np.ndarray] = None) -> list[FrameCalib]:
        out: list[FrameCalib] = []
        anchor = None                                   # (tracker, H_img_to_pitch at anchor)
        buf: list[np.ndarray] = []

        def flush(start: int) -> None:
            nonlocal anchor
            keys = [i for i in range(len(buf)) if (start + i) % self.every == 0]
            fits = self.fit_keyframes(buf, keys) if keys else {}
            self.keyframes += len(keys)
            for i, f in enumerate(buf):
                idx = start + i
                fit = fits.get(i)
                if fit is not None and fit.rep_err <= self.gate_px:
                    self.keyframes_accepted += 1
                    anchor = (self.tracker_factory(f, mask), fit.H_img_to_pitch)
                    out.append(FrameCalib(idx, fit.H_img_to_pitch, fit.cam, "keyframe",
                                          fit.rep_err))
                    continue
                err = fit.rep_err if fit is not None else None
                if anchor is not None:
                    H_a_t = anchor[0].update(f, mask)
                    if H_a_t is not None:
                        H = anchor[1] @ np.linalg.inv(H_a_t)
                        H /= H[2, 2]
                        h, w = f.shape[:2]
                        cam = camera_from_homography(np.linalg.inv(H), w, h)
                        out.append(FrameCalib(idx, H, cam, "bridged", err))
                        continue
                out.append(FrameCalib(idx, None, None, "none", err))

        start = 0
        for f in frames_bgr:
            buf.append(f)
            if len(buf) == self.chunk:
                flush(start)
                start += len(buf)
                buf = []
        if buf:
            flush(start)
        return out
```

- [ ] **Step 4: Run:** PASS.
- [ ] **Step 5: Commit:** `git add footlytics/geometry/clip_calib.py tests/test_clip_calib.py && git commit -m "feat(geometry): clip calibration, keyframes gated at 25 px, bridged by the camera tracker"`

---

### Task 5: Bench plumbing: paths, weights, SoccerMaster checkout, calib extra on the host

**Files:**
- Modify: `footlytics/bench/paths.py`, `footlytics/bench/datasets.py`, `footlytics/bench/remote.py`, `footlytics/bench/cli.py`, `footlytics/bench/host_setup.sh`
- Test: additions to `tests/test_bench_host.py`, `tests/test_bench_datasets.py`, `tests/test_bench_remote.py`, `tests/test_bench_cli.py`

**Interfaces:**
- Produces: `BenchPaths.weights` (`root/weights/soccermaster`), `.soccermaster_code`
  (`ext/SoccerMaster`), `.sn_calibration` (`ext/sn-calibration`), `.calib_zip(split)`;
  `datasets.fetch_hf(..., repo_type="dataset")`, `SOCCERMASTER_REPO/REVISION/FILES`,
  `FETCHABLE["soccermaster-weights"]`; `Remote.ship_dir(name, archive) -> str`;
  `cli._tar_dir(path) -> bytes`; `bench setup --soccermaster-code PATH`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bench_host.py`:
```python
def test_calibration_paths_live_under_the_root(tmp_path):
    p = BenchPaths(tmp_path)
    for q in (p.weights, p.soccermaster_code, p.sn_calibration, p.calib_zip("valid")):
        assert tmp_path in q.parents
    assert p.calib_zip("test").name == "test.zip"
```

Append to `tests/test_bench_datasets.py`:
```python
def test_soccermaster_weights_come_from_the_model_repo(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(host, "free_disk_bytes", lambda p: 50_000_000_000)
    sha = hashlib.sha256(b"w").hexdigest()
    seen = []

    def paths_info(repo, files, repo_type, revision):
        seen.append((repo, files[0], repo_type, revision))
        return [SimpleNamespace(size=1, lfs=SimpleNamespace(sha256=sha))]
    api = SimpleNamespace(get_paths_info=paths_info)

    def download(repo_id, filename, repo_type, revision, local_dir):
        p = tmp_path / "root" / "weights" / "soccermaster" / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"w")
        return str(p)
    monkeypatch.setattr(datasets, "_HF", (api, download))
    d = datasets.fetch(BenchPaths(tmp_path / "root"), "soccermaster-weights")
    assert sorted(f.name for f in d.glob("*.pt")) == sorted(datasets.SOCCERMASTER_FILES)
    assert {s[2] for s in seen} == {"model"} and {s[3] for s in seen} == {datasets.SOCCERMASTER_REVISION}
```
(Implementation note: `fetch_hf` takes `api`/`download` parameters as today. The FETCHABLE
entries pass `*datasets._HF` when `_HF` is set, so tests can inject both without a network.
`_HF = None` by default.)

Append to `tests/test_bench_remote.py`:
```python
def test_ship_dir_replaces_the_checkout_atomically():
    ssh = FakeSsh(ROOT)
    d = rm.Remote("h", runner=ssh).ship_dir("SoccerMaster", b"tar")
    assert d == "/home/iec/footlytics-bench/ext/SoccerMaster"
    cmd = next(c for c in ssh.calls if "tar -x" in c)
    assert "ext/SoccerMaster.tmp" in cmd and "mv " in cmd
    with pytest.raises(ValueError):
        rm.Remote("h", runner=FakeSsh(ROOT)).ship_dir("../x", b"")


def test_host_setup_installs_the_calib_extra():
    from pathlib import Path
    sh = (Path(rm.__file__).parent / "host_setup.sh").read_text()
    assert '"$SRC[dev,calib]"' in sh
```

Append to `tests/test_bench_cli.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure:** `uv run --extra dev pytest tests/test_bench_host.py tests/test_bench_datasets.py tests/test_bench_remote.py tests/test_bench_cli.py -q` → the new tests FAIL.

- [ ] **Step 3: Implement**
  - `paths.py`: add
    ```python
    @property
    def weights(self) -> Path:
        return self.root / "weights" / "soccermaster"

    @property
    def soccermaster_code(self) -> Path:
        """The SoccerMaster checkout (no licence: internal R&D), shipped by `bench setup`."""
        return self.ext / "SoccerMaster"

    @property
    def sn_calibration(self) -> Path:
        return self.ext / "sn-calibration"

    def calib_zip(self, split: str) -> Path:
        return self.data / "calibration" / f"{split}.zip"
    ```
  - `datasets.py`:
    - Add `repo_type: str = "dataset"` to `fetch_hf` and pass it to `get_paths_info` and the
      download.
    - Add `_HF = None` (test seam: `(api, download)`) and use
      `api, download = api or (_HF[0] if _HF else None), download or (_HF[1] if _HF else None)`.
    - Add `SOCCERMASTER_REPO = "xleprime/SoccerMaster"`,
      `SOCCERMASTER_REVISION = "0d94573662dbd678df22aa5c61ae77f474b939c9"`,
      `SOCCERMASTER_FILES = ("backbone.pt", "LinesDetection.pt", "KeypointsDetection.pt")`.
    - Add
      `def _weights(paths): [fetch_hf(SOCCERMASTER_REPO, f, SOCCERMASTER_REVISION, paths.weights / f, repo_type="model") for f in SOCCERMASTER_FILES]; return paths.weights`
      and `FETCHABLE["soccermaster-weights"] = _weights`.
    - Make `_calib` use `paths.calib_zip(split)`.
  - `remote.py`:
    ```python
    def ship_dir(self, name: str, archive: bytes) -> str:
        """Replace `ext/<name>` with the contents of a tar archive."""
        _safe(name)
        d = f"{self.root}/ext/{name}"
        self.sh(f"rm -rf {d}.tmp && mkdir -p {d}.tmp && tar -x -C {d}.tmp && rm -rf {d} "
                f"&& mv {d}.tmp {d}", input=archive, timeout=600)
        return d
    ```
  - `cli.py`:
    ```python
    SOCCERMASTER_CODE = REPO.parent / "SoccerMaster-code" / "codes" / "SoccerMaster"


    def _tar_dir(path: Path) -> bytes:
        import io
        import tarfile

        def keep(ti):
            parts = Path(ti.name).parts
            return None if "__pycache__" in parts or ".git" in parts else ti
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as t:
            t.add(str(path), arcname=".", filter=keep)
        return buf.getvalue()
    ```
    In `setup`, add the option
    `soccermaster_code: Path = typer.Option(SOCCERMASTER_CODE, help="SoccerMaster checkout to ship (internal R&D, no licence)")`.
    Inside `go()`, after `r.ship(...)`: if `soccermaster_code.exists()`, call
    `r.ship_dir("SoccerMaster", _tar_dir(soccermaster_code))`; otherwise echo a warning that
    calibration tasks will fail without it. Then call `r.setup(sha)`.
  - `host_setup.sh`: `"$ROOT/venv/bin/pip" install -q "$SRC[dev,calib]"`.
- [ ] **Step 4: Run:** the four files PASS.
- [ ] **Step 5: Commit:** `git add footlytics/bench tests/test_bench_*.py && git commit -m "feat(bench): SoccerMaster weights and checkout on the host, calib extra"`

---

### Task 6: Runner kinds `sncalib` and `gsr_calibration`; quick, milestone, calib-tune suites

**Files:**
- Modify: `footlytics/bench/runner.py`
- Create: `footlytics/bench/suites/quick.yaml`, `footlytics/bench/suites/milestone.yaml`, `footlytics/bench/suites/calib-tune.yaml`
- Test: additions to `tests/test_bench_runner.py`

**Interfaces:**
- Consumes: Tasks 1-5; plan 1's `_labels`, `require_vram`, `run_sncalib`.
- Produces: kinds `sncalib` (aggregate: `score, jac5, completeness, median_px, seconds_per_frame, frames, solver_errors`; plus `leakage`) and `gsr_calibration` (per clip: `score, jac5, completeness, calibrated_share, keyframes, keyframes_accepted`; aggregate: the same over all clips plus `seconds_per_frame`). Module seams for tests: `runner._calibrator(paths)`, `runner.solve_many`, `runner.run_sncalib`, `runner.require_vram`, `runner.TRACKER`. `VRAM_MIB = 10_000`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_bench_runner.py`)

```python
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


def write_gsr_clip(paths, seq="SNGS-990", n=3):
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
        for f in range(n):
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
    per_clip, combined = calib[0], calib[-1]
    assert per_clip[0] == ["SNGS-990-000001", "SNGS-990-000002", "SNGS-990-000003"]
    assert combined[1] == ["camera_SNGS-990-000001", "camera_SNGS-990-000002",
                           "camera_SNGS-990-000003"]
    assert "seconds_per_frame" in t["aggregate"]


def test_calibration_suites_are_valid():
    for name in ("quick", "milestone", "calib-tune"):
        s = runner.load_suite(name)
        assert {t["kind"] for t in s["tasks"]} <= set(runner.KINDS)
    q = runner.load_suite("quick")
    sc = next(t for t in q["tasks"] if t["kind"] == "sncalib")
    assert len(sc["frames"]) == 60 and len(set(sc["frames"])) == 60
```

(`make_labels` gives the unlabelled extra frame `has_labeled_pitch: False`, so 3 labelled
frames remain; its pitch annotations get lines here.)

- [ ] **Step 2: Run to verify failure:** `uv run --extra dev pytest tests/test_bench_runner.py -q`. Expected: the new tests FAIL (unknown kinds / missing seams).

- [ ] **Step 3: Implement** (append to `runner.py`; add imports
  `import re`, `import numpy as np`, `from .adapters.calib import gsr_line_frames, write_camera`,
  `from .evaluators import run_sncalib`, `from .host import require_vram`,
  `from .metrics import median_reprojection_px`,
  `from ..geometry.calib_nn import solve_many, DEFAULT_KP_THRESHOLD, DEFAULT_LINE_THRESHOLD`,
  `from ..geometry.camera_motion import AnchoredCamera`,
  `from ..geometry.clip_calib import ClipCalibrator, static_overlay_mask`)

```python
VRAM_MIB = 10_000
TRACKER = AnchoredCamera
_CALIBRATORS: dict = {}


def _calibrator(paths: BenchPaths):
    """One SoccerMaster model per process (1.4 GB of weights; thresholds are per call)."""
    key = (str(paths.soccermaster_code), str(paths.weights))
    if key not in _CALIBRATORS:
        from ..geometry.calib_nn import SoccerMasterCalibrator

        _CALIBRATORS[key] = SoccerMasterCalibrator(paths.soccermaster_code, paths.weights,
                                                   device="cuda")
    return _CALIBRATORS[key]


def _thresholds(cfg: dict) -> dict:
    return {"kp_threshold": cfg.get("kp_threshold", DEFAULT_KP_THRESHOLD),
            "line_threshold": cfg.get("line_threshold", DEFAULT_LINE_THRESHOLD)}


def _imread(raw: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("unreadable image")
    return img


def _sncal_frames(z: zipfile.ZipFile, split: str, spec) -> list[str]:
    have = sorted(m.group(1) for n in z.namelist()
                  if (m := re.match(rf"^{split}/(\d{{5}})\.json$", n)))
    if spec == "all":
        return have
    if isinstance(spec, dict):
        drop = set(spec.get("exclude", []))
        return [f for f in have
                if int(f) % spec["every"] == spec.get("offset", 0) and f not in drop]
    missing = sorted(set(spec) - set(have))
    if missing:
        raise KeyError(f"frames not in {split}: {missing[:5]}")
    return list(spec)


def _game(info: dict) -> str:
    return "|".join(str(info.get(k, "")).strip() for k in ("league", "season", "match", "date"))


def _copy_subset(src_gt: Path, src_pred: Path, dst: Path, split: str, frames: list[str]) -> None:
    for kind_, src, pat in (("gt", src_gt, "{}.json"), ("pred", src_pred, "camera_{}.json")):
        d = dst / kind_ / split
        d.mkdir(parents=True, exist_ok=True)
        for f in frames:
            p = src / pat.format(f)
            if p.exists():
                shutil.copy(p, d / p.name)


@kind("sncalib")
def sncalib(task: dict, ctx: TaskContext) -> dict:
    """Stills from SN-Calibration: one camera per frame, scored by the official evaluator."""
    split, cfg = task["split"], task.get("calibrator", {})
    require_vram(task.get("vram_mib", VRAM_MIB))
    gt_dir, pred_dir = ctx.work / "gt" / split, ctx.work / "pred" / split
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    cal = _calibrator(ctx.paths)
    t0 = time.time()
    failed, jobs, ids, gts = {}, [], [], {}
    with zipfile.ZipFile(ctx.paths.calib_zip(split)) as z:
        frames = _sncal_frames(z, split, task.get("frames", "all"))
        names = set(z.namelist())
        info = (json.loads(z.read(f"{split}/match_info.json"))
                if f"{split}/match_info.json" in names else {})
        for f in frames:
            try:
                raw = z.read(f"{split}/{f}.json")
                (gt_dir / f"{f}.json").write_bytes(raw)        # the evaluator's denominator
                gts[f] = json.loads(raw)
                img = _imread(z.read(f"{split}/{f}.jpg"))
                (kp, ln), = cal.decode([img[:, :, ::-1]], **_thresholds(cfg))
            except Exception as e:  # noqa: BLE001
                failed[f] = f"{type(e).__name__}: {e}"
                continue
            jobs.append((kp, ln, img.shape[1], img.shape[0]))
            ids.append(f)
    ctx.log(f"decoded {len(ids)} frames, solving on {task.get('workers', 8)} processes")
    solve_kw = {"voting": cfg.get("voting", True), "refine_lines": cfg.get("refine_lines", True)}
    results = solve_many(ctx.paths.soccermaster_code, jobs, workers=task.get("workers", 8),
                         **solve_kw)
    per, errors = {}, 0
    for f, (fit, err) in zip(ids, results):
        errors += err is not None
        if fit is None:
            per[f] = {"calibrated": 0}
            continue
        write_camera(pred_dir / f"camera_{f}.json", fit.cam)
        per[f] = {"calibrated": 1, "rep_err": round(fit.rep_err, 2),
                  "median_px": median_reprojection_px(gts[f], fit.cam)}
    s = run_sncalib(ctx.paths.eval_python, ctx.paths.sn_calibration, ctx.work / "gt",
                    ctx.work / "pred", split, ctx.work / "eval")
    med = [c["median_px"] for c in per.values() if c.get("median_px") is not None]
    agg = {"score": s.score, "jac5": s.jac, "completeness": s.completeness,
           "median_px": round(float(np.median(med)), 2) if med else None,
           "seconds_per_frame": round((time.time() - t0) / max(len(frames), 1), 2),
           "frames": len(frames), "solver_errors": errors}
    games = {f: _game(info.get(f"{f}.jpg", {})) for f in frames}
    listed = set(task.get("leakage_games", []))
    if not listed:
        leakage = {"status": "unverified", "games": len(set(games.values()))}
    else:
        keep = [f for f in frames if games[f] not in listed]
        leakage = {"status": "checked", "excluded_games": sorted(listed),
                   "frames_kept": len(keep)}
        if keep:
            _copy_subset(gt_dir, pred_dir, ctx.work / "clean", split, keep)
            sub = run_sncalib(ctx.paths.eval_python, ctx.paths.sn_calibration,
                              ctx.work / "clean" / "gt", ctx.work / "clean" / "pred", split,
                              ctx.work / "eval-clean")
            leakage["score_non_overlap"] = sub.score
    return {"clips": per, "aggregate": agg, "failed": failed, "leakage": leakage}


@kind("gsr_calibration")
def gsr_calibration(task: dict, ctx: TaskContext) -> dict:
    """GSR clips calibrated frame by frame (keyframes + tracker), scored on the GSR pitch-line
    labels by the SN-Calibration evaluator."""
    split, cfg = task["split"], task.get("calibrator", {})
    every, gate = task.get("every", 25), float(task.get("gate_px", 25.0))
    require_vram(task.get("vram_mib", VRAM_MIB))
    cal = _calibrator(ctx.paths)
    thr = _thresholds(cfg)
    solve_kw = {"voting": False, "mode": cfg.get("mode", "full"),
                "ransac": float(cfg.get("ransac", 10.0)),
                "refine_lines": cfg.get("refine_lines", True)}

    def fit_keyframes(chunk, keys):
        return cal.fit([f[:, :, ::-1] for f in chunk], which=keys, **thr, **solve_kw)

    per, failed = {}, {}
    all_gt, all_pred = ctx.work / "all" / "gt" / split, ctx.work / "all" / "pred" / split
    all_gt.mkdir(parents=True, exist_ok=True)
    all_pred.mkdir(parents=True, exist_ok=True)
    t0, n_frames = time.time(), 0
    with zipfile.ZipFile(ctx.paths.gsr_zip(split)) as z:
        for seq in _resolve(ctx.paths, split, task.get("clips", "all")):
            try:
                truth = gsr_line_frames(_labels(ctx.paths, split, seq))
                names = sorted(n for n in z.namelist()
                               if n.startswith(f"{seq}/img1/") and n.endswith(".jpg"))
                if not names:
                    raise FileNotFoundError(f"no frames for {seq} in {ctx.paths.gsr_zip(split)}")
                pick = sorted(set(np.linspace(0, len(names) - 1, min(25, len(names)))
                                  .astype(int).tolist()))
                mask = static_overlay_mask([_imread(z.read(names[i])) for i in pick])
                cc = ClipCalibrator(fit_keyframes, every=every, gate_px=gate,
                                    tracker_factory=TRACKER)
                res = cc.run((_imread(z.read(n)) for n in names), mask=mask)
                n_frames += len(res)
                gt_d, pred_d = (ctx.work / "clips" / seq / k / split for k in ("gt", "pred"))
                gt_d.mkdir(parents=True, exist_ok=True)
                pred_d.mkdir(parents=True, exist_ok=True)
                calibrated = 0
                for fc in res:
                    if fc.frame_idx not in truth:
                        continue
                    name = f"{seq}-{fc.frame_idx + 1:06d}"
                    for d in (gt_d, all_gt):
                        (d / f"{name}.json").write_text(json.dumps(truth[fc.frame_idx]))
                    if fc.cam is not None:
                        calibrated += 1
                        for d in (pred_d, all_pred):
                            write_camera(d / f"camera_{name}.json", fc.cam)
                s = run_sncalib(ctx.paths.eval_python, ctx.paths.sn_calibration,
                                gt_d.parent, pred_d.parent, split, ctx.work / "eval" / seq)
                per[seq] = {"score": s.score, "jac5": s.jac, "completeness": s.completeness,
                            "calibrated_share": round(calibrated / max(len(truth), 1), 4),
                            "keyframes": cc.keyframes, "keyframes_accepted": cc.keyframes_accepted}
                ctx.log(f"{seq}: score {s.score:.1f}, {calibrated}/{len(truth)} labelled "
                        f"frames calibrated")
            except Exception as e:  # noqa: BLE001
                failed[seq] = f"{type(e).__name__}: {e}"
    if not per:
        raise RuntimeError(f"no clip calibrated: {failed}")
    s = run_sncalib(ctx.paths.eval_python, ctx.paths.sn_calibration, all_gt.parent,
                    all_pred.parent, split, ctx.work / "eval" / "all")
    keys = sum(c["keyframes"] for c in per.values())
    agg = {"score": s.score, "jac5": s.jac, "completeness": s.completeness,
           "calibrated_share": round(float(np.mean([c["calibrated_share"] for c in per.values()])), 4),
           "keyframe_accept_rate": round(sum(c["keyframes_accepted"] for c in per.values())
                                         / max(keys, 1), 4),
           "seconds_per_frame": round((time.time() - t0) / max(n_frames, 1), 3)}
    return {"clips": per, "aggregate": agg, "failed": failed}
```
Also add `import cv2` at the top of `runner.py`.

`suites/quick.yaml`:
```yaml
# Engineering compass: run after any change. Plan 3 adds the GSR pipeline tasks.
name: quick
tasks:
  - id: sncalib-valid-60
    kind: sncalib
    split: valid
    # one frame per game (55) plus 5, seed 2026: see the plan's "Pins"
    frames: ["00007", "00091", "00155", "00250", "00260", "00322", "00357", "00417", "00486",
             "00532", "00598", "00621", "00680", "00753", "00783", "00841", "00882", "00966",
             "00972", "01027", "01111", "01247", "01261", "01366", "01384", "01491", "01533",
             "01581", "01609", "01622", "01655", "01716", "01728", "01797", "01808", "01831",
             "01885", "01924", "01987", "01999", "02075", "02080", "02130", "02176", "02205",
             "02270", "02333", "02378", "02415", "02531", "02578", "02650", "02706", "02766",
             "02799", "02887", "02941", "03046", "03062", "03194"]
  - id: gsr-calibration-valid-6
    kind: gsr_calibration
    split: valid
    clips: [SNGS-025, SNGS-028, SNGS-050, SNGS-056, SNGS-081, SNGS-086]
    every: 25
    gate_px: 25
```

`suites/milestone.yaml`:
```yaml
# Milestones only: the test split is never used for tuning.
name: milestone
tasks:
  - id: sncalib-test
    kind: sncalib
    split: test
    frames: all
    # SoccerMaster's training games are not published; until they are, the score is reported
    # with leakage "unverified" and is not quoted externally (spec: Leakage).
    leakage_games: []
```

`suites/calib-tune.yaml`:
```yaml
# Settings search on valid frames and clips the quick suite does not use (never test).
name: calib-tune
tasks:
  - id: tune-pnl-thresholds
    kind: sncalib
    split: valid
    frames: {every: 16, offset: 12, exclude: ["00972"]}
    calibrator: {kp_threshold: 0.3434, line_threshold: 0.7867, refine_lines: false}
  - id: tune-pnl-thresholds-refine
    kind: sncalib
    split: valid
    frames: {every: 16, offset: 12, exclude: ["00972"]}
    calibrator: {kp_threshold: 0.3434, line_threshold: 0.7867, refine_lines: true}
  - id: tune-low-thresholds-refine
    kind: sncalib
    split: valid
    frames: {every: 16, offset: 12, exclude: ["00972"]}
    calibrator: {kp_threshold: 0.1, line_threshold: 0.1, refine_lines: true}
  - id: tune-clip-every-10
    kind: gsr_calibration
    split: valid
    clips: [SNGS-021, SNGS-033]
    every: 10
  - id: tune-clip-every-25
    kind: gsr_calibration
    split: valid
    clips: [SNGS-021, SNGS-033]
    every: 25
```

- [ ] **Step 4: Run:** `uv run --extra dev pytest tests/test_bench_runner.py -q` → PASS; full suite `uv run --extra dev --extra calib pytest -q` → green.
- [ ] **Step 5: Commit:** `git add footlytics/bench/runner.py footlytics/bench/suites tests/test_bench_runner.py && git commit -m "feat(bench): sncalib and gsr_calibration tasks; quick, milestone, calib-tune suites"`

---

### Task 7: Host runs (operational)

There is no new code unless a step fails. If one does, fix it with a test in the owning task's
file.

- [ ] **Step 1: Setup:** `uv run footlytics bench setup --remote iec4090_2`. Expected: the
  checkout ships to `ext/SoccerMaster`, the calib extra installs, and the torch line still reads
  `2.7.1+cu128`. Then check that no second torch came in:
  `ssh iec4090_2 '~/footlytics-bench/venv/bin/pip list 2>/dev/null | grep -i -E "^torch |transformers|timm"'`.
- [ ] **Step 2: Fetch:** `uv run footlytics bench fetch soccermaster-weights calib-test --remote iec4090_2`
  (about 2 GB; check the free space stays well above the floor).
- [ ] **Step 3: Real-frame test on the host:** run pytest from the shipped source:
  `ssh iec4090_2 'cd ~/footlytics-bench/src/<sha> && FOOTLYTICS_SOCCERMASTER_CODE=~/footlytics-bench/ext/SoccerMaster FOOTLYTICS_SOCCERMASTER_WEIGHTS=~/footlytics-bench/weights/soccermaster FOOTLYTICS_SNCAL_ZIP=~/footlytics-bench/data/calibration/valid.zip FOOTLYTICS_SNCALIB=~/footlytics-bench/ext/sn-calibration FOOTLYTICS_EVAL_PYTHON=~/footlytics-bench/venv-eval/bin/python YOLO_CONFIG_DIR=~/footlytics-bench/cache/ultralytics ~/footlytics-bench/venv/bin/python -m pytest tests/test_calib_nn.py tests/test_bench_calib.py -q'`.
  Expected: all PASS, including the real frame (median px < 15).
- [ ] **Step 4: Tune:** `uv run footlytics bench run --suite calib-tune --remote iec4090_2`.
  - Set `DEFAULT_KP_THRESHOLD`, `DEFAULT_LINE_THRESHOLD` and the `refine_lines` default to the
    sncalib setting with the best `score`.
  - Set the quick suite's `every` from the two clip tasks: the smaller `every` only if its score
    is higher by more than 1 point; otherwise 25, which is cheaper.
  - Ledger the choice as a ruling. Commit the defaults with the tune row:
    `git commit -m "bench: calibrator settings from calib-tune (valid, non-quick frames)"`.
- [ ] **Step 5: Quick:** `uv run footlytics bench run --suite quick --remote iec4090_2`; commit the row.
- [ ] **Step 6: Milestone:** `uv run footlytics bench run --suite milestone --remote iec4090_2`
  (SN-Calibration test, about 3,100 frames; on 8 solver processes expect about 40 min). Commit
  the row.
- [ ] **Step 7: Spec:** under Reference points, add the measured numbers: quick valid score,
  GSR clip score, and the test score next to PnLCalib 51.8 / SoccerMaster 56.2, with leakage
  "unverified". Commit with `git commit -m "bench: calibration measured on iec4090_2"`.

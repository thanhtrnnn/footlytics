# Benchmark full-pipeline GSR Implementation Plan (plan 3 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the whole footlytics pipeline on GSR broadcast clips and score it.
- GS-HOTA with jersey matching off (primary) and on (official), with DetA and AssA.
- Team accuracy, ball recall and pixel error, tactical-shape error in metres, and seconds per
  match minute.
- Suites: quick (6 pinned clips), nightly (all 58 valid clips, run twice for the noise floor) and
  milestone (49 test clips, next to the re-scored baseline).

**Architecture:**
- `pipeline.radar.run` is split. The frame loop becomes `run_frames(frames, fps, mapper, ...)`,
  where a *mapper* turns each frame's detections into pitch positions, or returns None when the
  frame cannot be related to the pitch.
- `run(video, calibration, camera)` becomes a thin wrapper over it, with unchanged behaviour.
- GSR clips feed `run_frames` straight from the zip. The mapper uses the per-frame homographies
  from plan 2's `ClipCalibrator`, which are computed once per clip and cached in process, so the
  `gsr_calibration` and `gsr_pipeline` tasks of one run share them.
- Our metrics compare the predicted Match State with the ground-truth Match State (plan 1's
  `gt_to_state`), after mapping our teams to left and right with `team_sides`.

**Tech Stack:** as plans 1-2, plus Ultralytics YOLO (SoccerMaster's `yolo_v8x6_finetuned.pt`,
person only) and the roboflow `football-ball-detection` model (AGPL-3.0, internal benchmarking
only), and scipy's `linear_sum_assignment`.

**Spec:** `docs/superpowers/specs/2026-10-11-benchmark-design.md` (suites table rows quick / nightly / milestone for GSR; validity rules Teams, Leakage, Noise; build-order step 3).

## Global Constraints

- Everything from plans 1-2 still holds:
  - Shared host `iec4090_2`, everything under `~/footlytics-bench/`, 1.5 GB disk floor.
  - VRAM is checked against what other processes leave us.
  - Evaluators run as subprocesses and are never imported.
  - A clip failure is recorded and makes the task `partial`.
  - The test split is for milestones only.
  - SoccerMaster code is never vendored.
- Primary score: GS-HOTA with jersey off. The official jersey-on score is reported next to it.
  We have no shirt-number reader, so predictions always carry a null jersey.
- Teams: HOME/AWAY are mapped to left/right per clip from our own output (`team_sides`), never
  from ground truth.
- Rows from the full pipeline carry `leakage: {"status": "unverified"}`, because the detector
  weights are SoccerMaster's.
- Noise: the nightly suite runs the same commit twice, and a change smaller than that spread is
  reported as no change.
- Possession on SoccerTrack v2 is out of scope (the dataset is gated). Plan 4 (V.League mini-GT)
  waits for footage.
- Tests: `uv run --extra dev --extra calib pytest`.

## Facts verified while planning (2026-10-12)

- `pipeline/radar.py:run(video_path, calibration, detector, meta, cfg, tracker_cfg, camera)`:
  - It reads frames with `cv2.VideoCapture` and checks the calibration's image size against the
    video.
  - With `camera` (an `AnchoredCamera`), every frame calls
    `camera.update(frame_bgr, mask=<255 except dilated detection boxes>)`, and a None result
    clears that frame's detections and counts it uncalibrated. Feet go to the pitch through
    `MovingCalibration(calibration).feet_to_pitch(dets[:, :4], H)`. Without a camera they go
    through `calibration.feet_to_pitch(dets[:, :4])`.
  - The report has `frames_uncalibrated / frames_calibrated / calib_success_rate / reanchors`
    only when a camera is given.
  - It keeps up to 600 full frames for a team montage (`report["_montage"]`). For 1080p frames
    that is 3.7 GB of RAM.
- Callers of `run`: `footlytics/cli.py`, `scripts/{compare_conf,run_moving_cam,run_pipeline_real}.py`,
  `tests/test_pipeline_camera.py`. Its signature stays.
- Detector: `weights/yolo_v8x6_finetuned.pt` exposes only `person`. Ball:
  `weights/football-ball-detection.pt` (one class, `ball`; `DetectorConfig.ball_weights`, run at
  1920).
  - HF sources: `xleprime/SoccerMaster` revision `0d94573662dbd678df22aa5c61ae77f474b939c9`
    (`yolo_v8x6_finetuned.pt`, lfs sha256
    `c85259af82ae919294e9ec87457d73646682159a5511ec1df033fed84b15d03d`).
  - `martinjolif/yolo-football-ball-detection` revision
    `63666b30e2a4973e089c982769f146af54530750` (`yolo-football-ball-detection.pt`, sha256
    `fb37942448e7de08745e8aab148d0794f680a738ddd55e5f17abe9ab2d6313fb`, AGPL-3.0, the same file
    as the local one).
- `geometry/homography.py`: `apply_homography(H, pts) -> (N, 2)`.
- `bench/adapters/gsr.py`:
  - `gt_to_state(labels)` maps GT left to HOME and right to AWAY, referees to OFFICIAL, and gives
    0-based `frame_idx` with bbox columns.
  - `state_to_predictions(state, split, use_jersey=False)`, and
    `team_sides(tracks) -> {"home": "left"|"right", ...}`.
- Plan 2 runner pieces reused:
  - `_calibrator(paths)`, `ClipCalibrator(fit_keyframes, every, gate_px, tracker_factory=TRACKER)`
    with `.run(frames_bgr, mask) -> [FrameCalib(frame_idx, H_img_to_pitch, cam, source, rep_err)]`.
  - `static_overlay_mask`, `_imread`, `_labels`, `_resolve`, and `_score(ctx, split, tracker,
    seqs)` (plan 1: GS-HOTA jersey off and on, per clip plus combined).
  - `require_vram`, `VRAM_MIB`.
- Measured: clip calibration ran at 0.229 s/frame before plan 2's keyframes-only change. GSR
  clips are 750 frames at 25 fps, 1920×1080.

## Review Focus

- **A clip where calibration fails on most frames** → the pipeline still runs; uncalibrated frames have no detections; GS-HOTA counts them as misses; nothing crashes. (Task 1 test.)
- **A clip where the pipeline finds no one at all** (`run_frames` raises "no detections") → that clip is failed with its reason, others are scored. (Task 4 test.)
- **Our team labels are swapped relative to GT** (HOME is on the right) → team accuracy and shape error use `team_sides`, so a swap does not count as an error; only a wrong side does. (Task 2 test.)
- **A predicted ball that was interpolated** (no image box) → excluded from the pixel error, not counted as a hit at (0, 0). (Task 2 test.)
- **The repeat task in nightly** → it re-calibrates (`fresh: true`), so the noise floor includes calibration, and `noise` is the absolute difference of every numeric aggregate. (Task 4 test.)

## File structure

```
footlytics/pipeline/radar.py        run_frames + FeetMapper + homography_mapper; run() becomes a wrapper; PipelineConfig.montage
footlytics/bench/metrics.py         + match_people, team_accuracy, ball_metrics, shape_error, seconds_per_match_minute
footlytics/bench/paths.py           + detector_weights, ball_weights
footlytics/bench/datasets.py        + FETCHABLE["detector-weights"]
footlytics/bench/runner.py          + clip-calibration cache, gsr_pipeline kind, noise_of, PIPELINE_VRAM_MIB
footlytics/bench/suites/quick.yaml  + gsr-pipeline-valid-6
footlytics/bench/suites/nightly.yaml (new) all 58 valid clips, twice
footlytics/bench/suites/milestone.yaml + gsr-pipeline-test, gsr-baseline-test
tests/test_pipeline_frames.py (new), tests/test_bench_metrics.py (new), additions to tests/test_bench_{host,datasets,runner}.py
```

---

### Task 1: `run_frames`: the pipeline over any frames with a per-frame mapper

**Files:**
- Modify: `footlytics/pipeline/radar.py`
- Test: `tests/test_pipeline_frames.py` (new); existing `tests/test_pipeline_camera.py`, `tests/test_pipeline_filters.py`, `tests/test_fixed_camera_regression.py` must still pass

**Interfaces:**
- Produces: `FeetMapper = Callable[[int, np.ndarray, np.ndarray], Optional[np.ndarray]]`.
  It is called once per processed source frame as `mapper(src_idx, frame_bgr, dets)`, where
  `dets` are `[N, >=6]` rows `(x, y, w, h, conf, cls)` and N may be 0. It returns feet in pitch
  metres `[N, 2]` (an empty `[0, 2]` for no detections), or None when the frame cannot be related
  to the pitch.
- `run_frames(frames: Iterable[np.ndarray], fps: float, mapper: FeetMapper, detector, meta,
  cfg=None, tracker_cfg=None, source: str = "frames") -> tuple[MatchState, dict]`.
- `homography_mapper(Hs: Sequence[Optional[np.ndarray]]) -> FeetMapper`: per-frame image→pitch
  homographies; None or an index past the end gives None.
- `PipelineConfig.montage: bool = True`.
- `run(...)` keeps its signature and behaviour. The report now always has `frames_uncalibrated`,
  `frames_calibrated` and `calib_success_rate`; `reanchors` only with a camera.

- [ ] **Step 1: Write the failing tests** in `tests/test_pipeline_frames.py`:

```python
"""run_frames: the pipeline over an iterator of frames with a per-frame pitch mapper."""
import numpy as np
import pytest

from footlytics.pipeline.radar import PipelineConfig, homography_mapper, run_frames
from footlytics.state.schema import MatchMeta, Role


class FakeDetector:
    """Three people standing still and a ball that moves; one class map like the x6 weights."""
    roles = {0: Role.PLAYER.value, 1: Role.BALL.value}

    def __init__(self, n_frames):
        self.t = 0

    def detect(self, frame):
        t = int(frame[0, 0, 0])
        people = [[100 + 60 * k, 300, 20, 60, 0.9, 0] for k in range(3)]
        ball = [[200 + 2 * t, 340, 6, 6, 0.5, 1]]
        return np.array(people + ball, float)

    def role_of(self, c):
        return self.roles[int(c)]


def frames(n, h=480, w=640):
    out = []
    for t in range(n):
        f = np.full((h, w, 3), 90, np.uint8)
        f[0, 0] = t                      # all channels: the pipeline hands the detector RGB
        out.append(f)
    return out


# image (px) -> pitch (m): 10 cm per pixel, origin at image (320, 240)
H = np.array([[0.1, 0, -32.0], [0, 0.1, -24.0], [0, 0, 1.0]])


def test_homography_mapper_maps_feet_and_handles_gaps():
    m = homography_mapper([H, None])
    dets = np.array([[100, 300, 20, 60, 0.9, 0]], float)
    assert m(0, None, dets) == pytest.approx(np.array([[0.1 * 110 - 32, 0.1 * 360 - 24]]))
    assert m(0, None, dets[:0]).shape == (0, 2)
    assert m(1, None, dets) is None and m(5, None, dets) is None


def test_run_frames_tracks_people_and_ball_with_a_homography_per_frame():
    n = 12
    state, report = run_frames(frames(n), 25.0, homography_mapper([H] * n), FakeDetector(n),
                               MatchMeta(match_id="t", fps=25.0),
                               PipelineConfig(verbose=False, montage=False, use_appearance=False,
                                              stitch=False, touchline_people=False))
    assert report["frames_processed"] == n and report["frames_calibrated"] == n
    assert report["calib_success_rate"] == 1.0 and "_montage" not in report
    people = state.tracks[state.tracks["role"] != Role.BALL.value]
    assert people["track_id"].nunique() == 3          # three steady tracks
    assert set(state.tracks["frame_idx"]) == set(range(n))


def test_uncalibrated_frames_lose_their_detections_but_the_run_goes_on():
    n = 12
    Hs = [H if t % 3 else None for t in range(n)]         # every third frame uncalibrated
    state, report = run_frames(frames(n), 25.0, homography_mapper(Hs), FakeDetector(n),
                               MatchMeta(match_id="t", fps=25.0),
                               PipelineConfig(verbose=False, montage=False, use_appearance=False,
                                              stitch=False, touchline_people=False,
                                              ball_max_gap=0))
    assert report["frames_uncalibrated"] == 4 and report["frames_calibrated"] == 8
    seen = set(state.tracks.loc[~state.tracks["interpolated"], "frame_idx"])
    assert not seen & {0, 3, 6, 9}
```

- [ ] **Step 2: Run to verify failure:** `uv run --extra dev pytest tests/test_pipeline_frames.py -q`. Expected: ImportError (`run_frames` missing).

- [ ] **Step 3: Implement** in `footlytics/pipeline/radar.py`:
  - Add `montage: bool = True` to `PipelineConfig`, with the comment: "Keep up to 600 frames for
    the team montage (`report['_montage']`). Off for batch runs: at 1080p that is 3.7 GB of RAM."
  - Add near the top:
    ```python
    #: (source frame index, BGR frame, detections [N, >=6]) -> feet in pitch metres [N, 2], or
    #: None when this frame cannot be related to the pitch: its detections are dropped and it
    #: counts as uncalibrated. Called once per processed frame, also when N == 0, so a camera
    #: tracker behind it sees every frame.
    FeetMapper = Callable[[int, np.ndarray, np.ndarray], Optional[np.ndarray]]


    def homography_mapper(Hs: Sequence[Optional[np.ndarray]]) -> FeetMapper:
        """Per-frame image -> pitch homographies (e.g. from `ClipCalibrator`)."""
        def mapper(idx: int, frame_bgr, dets: np.ndarray) -> Optional[np.ndarray]:
            H = Hs[idx] if 0 <= idx < len(Hs) else None
            if H is None:
                return None
            if not len(dets):
                return np.empty((0, 2))
            b = np.asarray(dets[:, :4], float)
            feet = np.column_stack([b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3]])
            return apply_homography(H, feet)
        return mapper
    ```
    (import `apply_homography` from `..geometry.homography`, and `Callable, Iterable, Sequence`
    from `typing`).
  - Create `run_frames(frames, fps, mapper, detector, meta, cfg=None, tracker_cfg=None,
    source="frames")` by moving the body of today's `run` into it, starting at
    `cfg = cfg or PipelineConfig()`, with exactly these changes:
    - No `cv2.VideoCapture`, no image-size check, no `cap.release()`. Iterate `for frame_bgr in
      frames:`, keeping the stride and `max_frames` logic, and using `src_idx` and `out_idx` as
      today. `src_fps = fps`. Log the size from the first frame.
    - Replace the `camera` block and the `xy = (moving.feet_to_pitch(...) if moving ... else
      calibration.feet_to_pitch(...))` line with:
      ```python
      xy_all = mapper(src_idx, frame_bgr, dets)
      if xy_all is None:
          n_uncalibrated += 1
          dets = dets[:0]
      ```
      Then under `if len(dets):` use `xy = xy_all` (same length and order as `dets`).
    - Only collect `montage_frames` and `montage_boxes`, and only set `report["_montage"]`, when
      `cfg.montage` is true.
    - Always report `frames_uncalibrated`, `frames_calibrated` and `calib_success_rate`.
  - Rewrite `run(...)` as a wrapper. Same signature and docstring purpose. It:
    - opens the video and reads `fps` and the size;
    - keeps the existing calibration-size check and its error message;
    - builds the mapper: with `camera`, a closure that builds the box mask exactly as today, calls
      `camera.update(frame_bgr, mask=mask)`, and returns None or
      `MovingCalibration(calibration).feet_to_pitch(dets[:, :4], H)` (an empty `[0, 2]` for no
      detections); without `camera`, `calibration.feet_to_pitch(dets[:, :4])` (empty for none);
    - streams frames from the capture with a generator that releases the capture at the end;
    - calls `run_frames(...)`;
    - adds `report["reanchors"] = camera.n_reanchors` when there is a camera.
- [ ] **Step 4: Run:** `uv run --extra dev pytest tests/test_pipeline_frames.py tests/test_pipeline_camera.py tests/test_pipeline_filters.py tests/test_fixed_camera_regression.py tests/test_cli.py -q` → PASS (camera and CLI tests may skip when the local clip or weights are missing; they ran before on the M2, so run them with the clip present), then the full suite.
- [ ] **Step 5: Commit:** `refactor(pipeline): run_frames over any frames with a per-frame pitch mapper`

---

### Task 2: Pipeline metrics against ground truth

**Files:**
- Modify: `footlytics/bench/metrics.py` (append)
- Test: `tests/test_bench_metrics.py` (new)

**Interfaces:**
- Consumes: Match State tracks (`frame_idx, track_id, role, team, x, y, bbox_x/y/w/h, interpolated`).
- Produces:
  - `match_people(gt, pred, frames, max_dist=2.0) -> pd.DataFrame[frame_idx, gt, pred, dist]`
    (one Hungarian match per frame on pitch distance, players and goalkeepers only).
  - `team_accuracy(gt, pred, matches, pred_sides) -> float | None`.
  - `ball_metrics(gt, pred, frames, hit_px=20.0) -> {"ball_recall", "ball_px_median"}`.
  - `shape_error(gt, pred, frames, pred_sides, min_players=6) -> dict`: `shape_error_m`, plus
    `centroid_x_err`, `centroid_y_err`, `width_err`, `depth_err` and `compactness_err`.
  - `seconds_per_match_minute(seconds, frames, fps=25.0) -> float`.
  - GT convention: GT team HOME means left and AWAY means right (`gt_to_state`).

- [ ] **Step 1: Write the failing tests**

```python
"""Pipeline metrics against GSR ground truth, on synthetic Match States."""
import numpy as np
import pandas as pd
import pytest

from footlytics.bench.metrics import (ball_metrics, match_people, seconds_per_match_minute,
                                      shape_error, team_accuracy)


def tracks(rows):
    cols = ["frame_idx", "track_id", "role", "team", "x", "y", "bbox_x", "bbox_y", "bbox_w",
            "bbox_h", "interpolated"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


def people(frame, xs, team, start_id, dx=0.0):
    return [(frame, start_id + i, "player", team, x + dx, 5.0 * i, 0, 0, 10, 20, False)
            for i, x in enumerate(xs)]


LEFT_X, RIGHT_X = [-30, -25, -20, -15, -10, -5], [5, 10, 15, 20, 25, 30]


def gt_frame(f):
    return people(f, LEFT_X, "home", 1) + people(f, RIGHT_X, "away", 101)


def test_match_people_pairs_within_the_distance():
    gt = tracks(gt_frame(0))
    pred = tracks(people(0, LEFT_X, "away", 1, dx=0.5) + people(0, RIGHT_X, "home", 101, dx=9))
    m = match_people(gt, pred, [0], max_dist=2.0)
    assert len(m) == 6 and (m["dist"] <= 2.0).all()


def test_team_accuracy_ignores_a_label_swap_but_not_a_wrong_side():
    gt = tracks(gt_frame(0))
    pred = tracks(people(0, LEFT_X, "away", 1) + people(0, RIGHT_X, "home", 101))
    m = match_people(gt, pred, [0])
    assert team_accuracy(gt, pred, m, {"away": "left", "home": "right"}) == 1.0
    assert team_accuracy(gt, pred, m, {"away": "right", "home": "left"}) == 0.0
    assert team_accuracy(gt, pred, m.iloc[:0], {}) is None


def test_ball_metrics_use_image_boxes_and_skip_interpolated_balls():
    gt = tracks([(f, 0, "ball", "unknown", 0, 0, 100, 100, 10, 10, False) for f in range(4)])
    pred = tracks([(0, 9, "ball", "unknown", 0, 0, 103, 104, 10, 10, False),   # 5 px: hit
                   (1, 9, "ball", "unknown", 0, 0, 160, 100, 10, 10, False),   # 60 px: miss
                   (2, 9, "ball", "unknown", 0, 0, np.nan, np.nan, np.nan, np.nan, True)])
    b = ball_metrics(gt, pred, range(4), hit_px=20.0)
    assert b["ball_recall"] == pytest.approx(1 / 4)
    assert b["ball_px_median"] == pytest.approx(np.median([5.0, 60.0]))


def test_shape_error_is_zero_for_identical_shapes_under_a_label_swap():
    gt = tracks(gt_frame(0) + gt_frame(1))
    pred = tracks(people(0, LEFT_X, "away", 1) + people(0, RIGHT_X, "home", 101)
                  + people(1, LEFT_X, "away", 1) + people(1, RIGHT_X, "home", 101))
    e = shape_error(gt, pred, [0, 1], {"away": "left", "home": "right"})
    assert e["shape_error_m"] == pytest.approx(0.0)
    shifted = pred.copy()
    shifted["x"] = shifted["x"] + 2.0
    assert shape_error(gt, shifted, [0, 1], {"away": "left", "home": "right"})["centroid_x_err"] \
        == pytest.approx(2.0)


def test_shape_error_without_enough_players_is_none():
    gt = tracks(people(0, LEFT_X[:3], "home", 1))
    assert shape_error(gt, gt, [0], {"home": "left"})["shape_error_m"] is None


def test_seconds_per_match_minute():
    assert seconds_per_match_minute(60.0, 750, 25.0) == pytest.approx(120.0)
```

- [ ] **Step 2: Run to verify failure:** `uv run --extra dev pytest tests/test_bench_metrics.py -q` → ImportError.

- [ ] **Step 3: Implement** (append to `footlytics/bench/metrics.py`; add `import pandas as pd`):

```python
# ------------------------------------------------------------ full pipeline vs ground truth

_PLAYING = ("player", "goalkeeper")
_GT_SIDE = {"home": "left", "away": "right"}       # gt_to_state: left -> HOME, right -> AWAY
_SHAPE = ("centroid_x", "centroid_y", "width", "depth", "compactness")


def _playing(t: pd.DataFrame) -> pd.DataFrame:
    p = t[t["role"].astype(str).isin(_PLAYING)]
    return p[np.isfinite(p["x"].astype(float)) & np.isfinite(p["y"].astype(float))]


def match_people(gt: pd.DataFrame, pred: pd.DataFrame, frames,
                 max_dist: float = 2.0) -> pd.DataFrame:
    """Per frame, the Hungarian match of GT and predicted players (and keepers) on pitch
    distance, keeping pairs within `max_dist` metres."""
    from scipy.optimize import linear_sum_assignment

    g, p = _playing(gt), _playing(pred)
    gb, pb = dict(tuple(g.groupby("frame_idx"))), dict(tuple(p.groupby("frame_idx")))
    rows = []
    for f in frames:
        a, b = gb.get(f), pb.get(f)
        if a is None or b is None:
            continue
        d = np.hypot(a["x"].to_numpy(float)[:, None] - b["x"].to_numpy(float)[None],
                     a["y"].to_numpy(float)[:, None] - b["y"].to_numpy(float)[None])
        r, c = linear_sum_assignment(d)
        for i, j in zip(r, c):
            if d[i, j] <= max_dist:
                rows.append((f, a.index[i], b.index[j], float(d[i, j])))
    return pd.DataFrame(rows, columns=["frame_idx", "gt", "pred", "dist"])


def team_accuracy(gt: pd.DataFrame, pred: pd.DataFrame, matches: pd.DataFrame,
                  pred_sides: dict) -> float | None:
    """Share of matched players whose predicted side (our team through `pred_sides`) is the
    GT side. A HOME/AWAY label swap is not an error; a wrong side is."""
    ok = n = 0
    for gi, pi in zip(matches["gt"], matches["pred"]):
        side = _GT_SIDE.get(str(gt.at[gi, "team"]))
        if side is None:
            continue
        n += 1
        ok += pred_sides.get(str(pred.at[pi, "team"])) == side
    return round(ok / n, 4) if n else None


def _ball_centres(t: pd.DataFrame) -> dict:
    b = t[(t["role"].astype(str) == "ball") & ~t["interpolated"].astype(bool)]
    b = b[np.isfinite(b["bbox_x"].astype(float))]
    c = np.column_stack([b["bbox_x"] + b["bbox_w"] / 2, b["bbox_y"] + b["bbox_h"] / 2])
    return dict(zip(b["frame_idx"], c))


def ball_metrics(gt: pd.DataFrame, pred: pd.DataFrame, frames,
                 hit_px: float = 20.0) -> dict:
    """In the image (detection, not calibration): recall = share of GT-ball frames where our
    ball box centre is within `hit_px`; px median over frames where both have a box.
    Interpolated balls have no box and do not count."""
    g, p = _ball_centres(gt), _ball_centres(pred)
    gt_frames = [f for f in frames if f in g]
    d = [float(np.linalg.norm(g[f] - p[f])) for f in gt_frames if f in p]
    return {"ball_recall": round(sum(x <= hit_px for x in d) / len(gt_frames), 4)
            if gt_frames else None,
            "ball_px_median": round(float(np.median(d)), 2) if d else None}


def _shape(xy: np.ndarray) -> dict:
    c = xy.mean(axis=0)
    return {"centroid_x": c[0], "centroid_y": c[1], "width": np.ptp(xy[:, 1]),
            "depth": np.ptp(xy[:, 0]),
            "compactness": float(np.linalg.norm(xy - c, axis=1).mean())}


def shape_error(gt: pd.DataFrame, pred: pd.DataFrame, frames, pred_sides: dict,
                min_players: int = 6) -> dict:
    """Mean absolute error (m) of each side's shape (centroid, width, depth, compactness),
    per frame where both GT and prediction see at least `min_players` of that side."""
    g, p = _playing(gt), _playing(pred)
    g = g.assign(side=g["team"].astype(str).map(_GT_SIDE))
    p = p.assign(side=p["team"].astype(str).map(pred_sides))
    errs = {k: [] for k in _SHAPE}
    gb = dict(tuple(g.groupby(["frame_idx", "side"])))
    pb = dict(tuple(p.groupby(["frame_idx", "side"])))
    for f in frames:
        for side in ("left", "right"):
            a, b = gb.get((f, side)), pb.get((f, side))
            if a is None or b is None or len(a) < min_players or len(b) < min_players:
                continue
            sa = _shape(a[["x", "y"]].to_numpy(float))
            sb = _shape(b[["x", "y"]].to_numpy(float))
            for k in _SHAPE:
                errs[k].append(abs(sa[k] - sb[k]))
    out = {f"{k}_err": round(float(np.mean(v)), 3) if v else None for k, v in errs.items()}
    vals = [v for v in out.values() if v is not None]
    out["shape_error_m"] = round(float(np.mean(vals)), 3) if vals else None
    return out


def seconds_per_match_minute(seconds: float, frames: int, fps: float = 25.0) -> float:
    return round(seconds / (frames / fps / 60.0), 2) if frames else float("nan")
```

- [ ] **Step 4: Run:** PASS.
- [ ] **Step 5: Commit:** `feat(bench): pipeline metrics vs ground truth (teams, ball, shape, s/min)`

---

### Task 3: Detector weights on the host

**Files:**
- Modify: `footlytics/bench/paths.py`, `footlytics/bench/datasets.py`
- Test: additions to `tests/test_bench_host.py`, `tests/test_bench_datasets.py`

**Interfaces:**
- Produces: `BenchPaths.detector_weights` (`root/weights/detector/yolo_v8x6_finetuned.pt`) and
  `.ball_weights` (`root/weights/detector/football-ball-detection.pt`);
  `FETCHABLE["detector-weights"]` (both files, pinned and checksummed).

- [ ] **Step 1: Tests**:
  - Append to `tests/test_bench_host.py`: `test_detector_paths_live_under_the_root`. Both paths
    sit under the root in `weights/detector/`, with the file names above.
  - Append to `tests/test_bench_datasets.py`: `test_detector_weights_come_from_their_two_model_repos`.
    It uses the `_HF` seam as in plan 2's weights test, and records
    `(repo, filename, repo_type, revision)` for each call. It asserts the two calls:
    - `("xleprime/SoccerMaster", "yolo_v8x6_finetuned.pt", "model", "0d94573662dbd678df22aa5c61ae77f474b939c9")`
    - `("martinjolif/yolo-football-ball-detection", "yolo-football-ball-detection.pt", "model", "63666b30e2a4973e089c982769f146af54530750")`

    It also asserts that both files land at the `BenchPaths` locations, under the local names
    above. The ball file is renamed from its HF name to `football-ball-detection.pt`.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**:
  - paths:
    ```python
    @property
    def detector_weights(self) -> Path:
        return self.root / "weights" / "detector" / "yolo_v8x6_finetuned.pt"

    @property
    def ball_weights(self) -> Path:
        """roboflow football-ball-detection (AGPL-3.0): internal benchmarking only."""
        return self.root / "weights" / "detector" / "football-ball-detection.pt"
    ```
  - datasets:
    - Add `BALL_REPO = "martinjolif/yolo-football-ball-detection"` and
      `BALL_REVISION = "63666b30e2a4973e089c982769f146af54530750"`.
    - `_detector(paths)`:
      - calls `fetch_hf(SOCCERMASTER_REPO, "yolo_v8x6_finetuned.pt", SOCCERMASTER_REVISION, paths.detector_weights, repo_type="model")`;
      - calls `fetch_hf(BALL_REPO, "yolo-football-ball-detection.pt", BALL_REVISION, paths.ball_weights, repo_type="model")`;
      - returns `paths.detector_weights.parent`.
    - `fetch_hf` already moves the downloaded file to `dest` when the names differ (plan 1:
      `got.replace(dest)`).
    - Register `FETCHABLE["detector-weights"] = _detector`.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit:** `feat(bench): detector and ball weights on the host, pinned`

---

### Task 4: `gsr_pipeline` task kind, clip-calibration cache, noise floor, suites

**Files:**
- Modify: `footlytics/bench/runner.py`, `footlytics/bench/suites/quick.yaml`, `footlytics/bench/suites/milestone.yaml`
- Create: `footlytics/bench/suites/nightly.yaml`
- Test: additions to `tests/test_bench_runner.py`

**Interfaces:**
- Consumes: Tasks 1-3; plan 1 `_score`, `state_to_predictions`, `gt_to_state`, `team_sides`; plan 2 clip calibration pieces.
- Produces:
  - `_clip_calibration(ctx, z, split, seq, cfg, every, gate, fresh=False) -> (list[FrameCalib], seconds, stats)`,
    cached in process under `(str(zip path), seq, every, gate, tuple(sorted(cfg.items())))`.
    `stats` = `{"keyframes", "keyframes_accepted", "solver_errors"}`; `seconds` is the
    calibration time paid when the entry was computed. `gsr_calibration` uses it too.
  - `_detector(paths, det_cfg: dict)`: a cached `Detector`.
  - `run_frames` is imported into runner's namespace (a test seam).
  - Kind `gsr_pipeline`. Per clip: `hota_jersey_off, deta_jersey_off, assa_jersey_off,
    hota_jersey_on, team_accuracy, ball_recall, ball_px_median, shape_error_m,
    calib_success_rate, seconds_per_match_minute`. The aggregate holds the combined GS-HOTA
    metrics plus the clip means of the others. `leakage` is unverified.
  - Task option `noise_of: <task id>` adds `res["noise"]`, the absolute difference of every
    numeric aggregate key the two tasks share. Option `fresh: true` bypasses the calibration
    cache.
  - `PIPELINE_VRAM_MIB = 14_000`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_bench_runner.py`). The tests use
  seams so that no GPU or model is needed:
  - monkeypatch `runner._clip_calibration` to return `([FrameCalib(i, H, None, "keyframe") for i in range(n)], 0.0, {"keyframes": 1, "keyframes_accepted": 1, "solver_errors": 0})`;
  - monkeypatch `runner._detector` to return `object()`;
  - monkeypatch `runner.run_frames` to return the GT Match State, i.e.
    `gt_to_state(labels)` for the clip, with its team labels swapped HOME↔AWAY to prove the
    side mapping. Report: `{"frames_processed": n, "frames_calibrated": n, "calib_success_rate": 1.0}`;
  - monkeypatch `runner.run_gsr` with the plan-1 `fake_scores`, which checks that the
    predictions file exists.

  Tests:
  - `test_gsr_pipeline_scores_a_clip_and_its_extras`:
    - status `complete`, with `hota_jersey_off`/`hota_jersey_on` in the per-clip and aggregate
      results;
    - `team_accuracy == 1.0`, despite the swap;
    - `shape_error_m is None` (the fixture has only 4 people per side, below `min_players=6`);
    - `ball_recall == 1.0` (make_labels' ball has a box);
    - `leakage == {"status": "unverified"}`;
    - the predictions file has null jerseys.
  - `test_gsr_pipeline_isolates_a_clip_with_no_detections`: the fake `run_frames` raises
    `RuntimeError("no detections at all")` for one of two clips. The result is `partial`, the bad
    clip is in `failed` with that message, and the other clip is scored.
  - `test_clip_calibration_is_shared_and_fresh_bypasses_it`: with the real `_clip_calibration`
    and a counting fake calibrator (reuse `FakeCal` and `every: 1` as in plan 2's gsr tests), the
    second call for the same clip and settings does not calibrate again. A call with
    `fresh=True` does.
  - `test_noise_of_reports_absolute_differences`: two fake tasks returning
    `{"aggregate": {"m": 1.0, "s": "x"}}` and `{"aggregate": {"m": 1.25}}`, the second with
    `noise_of` naming the first. The second gets `noise == {"m": 0.25}`.
  - `test_pipeline_suites_are_valid`: quick has `gsr-pipeline-valid-6` with the six pinned clips.
    Nightly has two `gsr_pipeline` tasks over `clips: all`; the second has `noise_of` naming the
    first and `fresh: true`. Milestone has `gsr-pipeline-test` (split test) and
    `gsr-baseline-test`.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** in `runner.py`:
  - `_CLIP_CALIBRATIONS: dict = {}` and `_clip_calibration(...)`:
    - Move the per-clip calibration from `gsr_calibration` into it: frame names from the zip,
      the overlay mask from 25 sampled frames, a `ClipCalibrator` whose fit function counts
      `cal.last_errors`, and `cc.run` over a generator of zip frames.
    - Time that work with `perf_counter`.
    - Store `(results, seconds, stats)` under the key unless `fresh`. Return the cached entry on
      a hit unless `fresh`.
    - `gsr_calibration` calls it and keeps everything else unchanged (its tests must still pass).
      Its `seconds_per_frame` uses the returned `seconds`.
  - `_DETECTORS: dict = {}` and `_detector(paths, det_cfg)`:
    - `Detector(paths.detector_weights, DetectorConfig(tile=det_cfg.get("tile", 0), imgsz=det_cfg.get("imgsz", 1280), conf=det_cfg.get("conf", 0.25), device="cuda", half=True, ball_weights=str(paths.ball_weights)))`,
      cached under `tuple(sorted(det_cfg.items()))`.
    - Import `Detector` and `DetectorConfig` lazily inside the function, because Ultralytics is
      heavy.
  - In `run_suite`, after a task's `res` is built and before it is stored:
    ```python
    ref = task.get("noise_of")
    if ref in row["tasks"]:
        a, b = res.get("aggregate") or {}, row["tasks"][ref].get("aggregate") or {}
        res["noise"] = {k: round(abs(a[k] - b[k]), 4) for k in a
                        if isinstance(a[k], (int, float)) and isinstance(b.get(k), (int, float))}
    ```
  - Kind:
    ```python
    PIPELINE_VRAM_MIB = 14_000


    @kind("gsr_pipeline")
    def gsr_pipeline(task: dict, ctx: TaskContext) -> dict:
        """The whole pipeline on GSR clips: calibrate, detect, track, assign teams; scored by
        GS-HOTA (jersey off, primary; on, official) plus our metrics against GT."""
        split, cfg = task["split"], task.get("calibrator", {})
        every, gate = task.get("every", 10), float(task.get("gate_px", 25.0))
        require_vram(task.get("vram_mib", PIPELINE_VRAM_MIB))
        det = _detector(ctx.paths, task.get("detector", {}))
        out_dir = ctx.work / "trackers" / f"SoccerNetGS-{split}" / "footlytics"
        out_dir.mkdir(parents=True, exist_ok=True)
        per, failed, done = {}, {}, []
        total_s, total_frames = 0.0, 0
        with zipfile.ZipFile(ctx.paths.gsr_zip(split)) as z:
            for seq in _resolve(ctx.paths, split, task.get("clips", "all")):
                try:
                    labels = _labels(ctx.paths, split, seq)
                    gt = gt_to_state(labels)
                    calib, calib_s, _ = _clip_calibration(ctx, z, split, seq, cfg, every, gate,
                                                          fresh=task.get("fresh", False))
                    names = sorted(n for n in z.namelist()
                                   if n.startswith(f"{seq}/img1/") and n.endswith(".jpg"))
                    t0 = time.perf_counter()
                    state, report = run_frames(
                        (_imread(z.read(n)) for n in names), 25.0,
                        homography_mapper([fc.H_img_to_pitch for fc in calib]), det,
                        MatchMeta(match_id=seq, fps=25.0, source_type="broadcast"),
                        PipelineConfig(verbose=False, montage=False))
                    run_s = time.perf_counter() - t0
                    (out_dir / f"{seq}.json").write_text(json.dumps(
                        state_to_predictions(state, split)))
                    sides = team_sides(state.tracks)
                    frames = sorted(set(gt.tracks["frame_idx"]))
                    m = match_people(gt.tracks, state.tracks, frames)
                    per[seq] = {
                        "team_accuracy": team_accuracy(gt.tracks, state.tracks, m, sides),
                        **ball_metrics(gt.tracks, state.tracks, frames),
                        "shape_error_m": shape_error(gt.tracks, state.tracks, frames,
                                                     sides)["shape_error_m"],
                        "calib_success_rate": report.get("calib_success_rate"),
                        "seconds_per_match_minute": seconds_per_match_minute(
                            calib_s + run_s, len(names)),
                    }
                    total_s += calib_s + run_s
                    total_frames += len(names)
                    done.append(seq)
                    ctx.log(f"{seq}: pipeline done ({len(names)} frames)")
                except Exception as e:  # noqa: BLE001
                    failed[seq] = f"{type(e).__name__}: {e}"
        if not done:
            raise RuntimeError(f"no clip ran: {failed}")
        scores = _score(ctx, split, "footlytics", done)
        for seq in done:
            per[seq].update(scores["clips"].get(seq, {}))

        def mean(k):
            v = [c[k] for c in per.values() if c.get(k) is not None]
            return round(float(np.mean(v)), 4) if v else None
        agg = {**scores["aggregate"],
               **{k: mean(k) for k in ("team_accuracy", "ball_recall", "ball_px_median",
                                       "shape_error_m", "calib_success_rate")},
               "seconds_per_match_minute": seconds_per_match_minute(total_s, total_frames)}
        return {"clips": per, "aggregate": agg, "failed": failed,
                "leakage": {"status": "unverified"}}
    ```
    Add the imports `from ..pipeline.radar import PipelineConfig, homography_mapper, run_frames`,
    `from ..state.schema import MatchMeta`,
    `from .metrics import match_people, team_accuracy, ball_metrics, shape_error, seconds_per_match_minute`,
    `from .adapters.gsr import team_sides`.
  - Suites:
    - `quick.yaml`: append
      ```yaml
        - id: gsr-pipeline-valid-6
          kind: gsr_pipeline
          split: valid
          clips: [SNGS-025, SNGS-028, SNGS-050, SNGS-056, SNGS-081, SNGS-086]
          every: 10
      ```
    - `nightly.yaml` (new):
      ```yaml
      # All 58 GSR valid clips, twice: the second run re-calibrates and its `noise` is the
      # run-to-run spread; a change smaller than that is no change (spec: Noise).
      name: nightly
      tasks:
        - id: gsr-pipeline-valid-all
          kind: gsr_pipeline
          split: valid
          clips: all
          every: 10
        - id: gsr-pipeline-valid-all-repeat
          kind: gsr_pipeline
          split: valid
          clips: all
          every: 10
          fresh: true
          noise_of: gsr-pipeline-valid-all
      ```
    - `milestone.yaml`: append
      ```yaml
        - id: gsr-pipeline-test
          kind: gsr_pipeline
          split: test
          clips: all
          every: 10
        - id: gsr-baseline-test
          kind: gsr_rescore
          split: test
          clips: all
          predictions: baseline/SoccerNetGS-test.zip
          tracker: tracklab
      ```
- [ ] **Step 4: Run:** `uv run --extra dev --extra calib pytest tests/test_bench_runner.py -q` → PASS; full suite green.
- [ ] **Step 5: Commit:** `feat(bench): gsr_pipeline task (GS-HOTA + teams, ball, shape, s/min), shared clip calibration, nightly noise floor`

---

### Task 5: Host runs (operational)

There is no new code unless a step fails. A failure is fixed with a test in the owning task's
file, through an implementer and a review.

- [ ] **Step 1:** `uv run footlytics bench fetch detector-weights --remote iec4090_2`, then wait for the tmux session to end and check the log.
- [ ] **Step 2:** `uv run footlytics bench run --suite quick --remote iec4090_2`. Expected:
  every task complete, and `gsr-pipeline-valid-6` gets GS-HOTA jersey-off and jersey-on per clip
  with the extras. Compare jersey-off with the re-scored baseline (38.34 on test, from plan 1's
  harness). Mind the split difference: quick is valid.
- [ ] **Step 3:** `uv run footlytics bench run --suite nightly --remote iec4090_2` (about 3-6 h; it runs in tmux; if ssh drops, ask the user to reconnect and `bench resume`).
- [ ] **Step 4:** `uv run footlytics bench run --suite milestone --remote iec4090_2`.
- [ ] **Step 5:** Commit the rows and scoreboard. Add a "Full pipeline, measured" table to the
  spec's Reference points: quick, nightly with its noise, milestone test GS-HOTA jersey off and
  on next to the baseline's 38.34 and 23.09, plus team accuracy, ball and shape. Commit with
  `bench: full pipeline measured on iec4090_2`.

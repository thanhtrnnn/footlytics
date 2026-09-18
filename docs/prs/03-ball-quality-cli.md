# PR 3: dedicated ball pass, officials split, quality report, calib-assist, CLI

Branch `feat/ball-quality-cli` (on top of `feat/moving-camera`).

> Landed in this repository with the migration to the Match State architecture.
> These four PRs were written against `scalliontor/Footlytics`; kept here as the
> record of what was bridged across from the V0 prototype and what each change measured.

## What
- `Detector`: `DetectorConfig.ball_weights` / `ball_imgsz` run a single-class ball model on the full frame; when it fires, its box replaces the ball row, the single-ball prior holds. Needed because `yolo_v8x6_finetuned.pt` exposes only a `person` class. Measured on a 720p tactical cam: ball in 30% of frames without, 95% with (roboflow football-ball-detection at 1920 px).
- `teams.split_officials`: for detectors with no referee class, a third-kit cluster is split off before the 2-means fit and the 11-a-side quota; guarded by size (<= 25% of tracks) and centroid separation (>= 0.6 of the two-team distance), so two kits stay a 2-means problem. Pipeline marks those tracklets `OFFICIAL` / `referee` and reports `officials_split`.
- `pipeline/quality.py`: `quality.json` + `quality.md` next to `tracks.parquet` (players per frame, % frames >= 18, ID-switch proxy per player per minute, ball coverage, calibration rate, seconds per match minute, `validate()` problems).
- `geometry/calib_assist.py`: grass-masked Hough line intersections -> numbered candidates image and a JSON template keyed by `Pitch.landmarks()` names.
- `cli.py`: `footlytics run | calib-assist | calibrate | drift-check`.
- `scripts/run_moving_cam.py`: the batch driver behind the numbers.

## Verification
`pytest -q`: 19 passed, 2 skipped on an M2. `footlytics run data/clips/sample_30s.mp4 --calib calib/brazil_france_30s_frame0.json --max-frames 12` writes tracks.parquet, meta.json, report.json, quality.json, quality.md, radar.mp4.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

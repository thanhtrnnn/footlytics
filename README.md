# FOOTLYTICS — Game-State Engine prototype

Match video -> continuous game state (22 players + ball) -> tactical data for analysts.

Docs: `docs/notion-summary.md` (idea), `docs/feasibility.md` (technical feasibility, blockers B8-B12),
`docs/tactical-query-vision.md` (tactical query MVP vision).

## Setup (macOS, Apple Silicon)

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
uv run pytest
```

## Run

```bash
bash scripts/fetch_sample.sh                       # one short public clip into data/clips/
uv run footlytics run data/clips/sample_30s.mp4 --out data/out/smoke
uv run footlytics run data/clips/sample_30s.mp4 --out data/out/calib --calib data/calib/sample_calib_frame0.json
```

`--calib` takes a landmark JSON for frame 0 (see `footlytics/calibration.py`); the camera motion tracker
propagates it through the clip. Without it, coordinates stay in image pixels.

Outputs: `gamestate.parquet`, `gamestate.csv`, `overlay.mp4`, `radar.mp4`, `quality.json`.

## Stages

0. COCO YOLO11 + ByteTrack + HSV team clustering, image-space coordinates. Done.
1. Landmark calibration (frame 0) + camera-motion propagation -> metre coordinates, 2D radar. Done on the 30 s clip.
2. Football-specific detector (player/goalkeeper/referee/ball), ball interpolation.
3. Cloud comparison vs sn-gamestate and SoccerMaster.

License note: `ultralytics` is AGPL-3.0. Research use only until licensed or swapped.

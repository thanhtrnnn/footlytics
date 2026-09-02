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
```

Outputs: `gamestate.parquet`, `gamestate.csv`, `overlay.mp4`, `radar.mp4`, `quality.json`.

## Stages

0. COCO YOLO11 + ByteTrack + HSV team clustering, image-space coordinates.
1. Pitch keypoints + homography -> metre coordinates, 2D radar.
2. Football-specific detector (player/goalkeeper/referee/ball), ball interpolation.
3. Cloud comparison vs sn-gamestate and SoccerMaster.

License note: `ultralytics` is AGPL-3.0. Research use only until licensed or swapped.

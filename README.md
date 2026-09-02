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

`--calib` takes a landmark JSON (see `footlytics/calibration.py`) annotated on one anchor frame (`"frame"` key,
default 0). The anchored camera tracker propagates it through the clip with KLT, re-anchors by SIFT matching every
50 frames and after shot cuts, and drops frames it cannot relate to the anchor. Without `--calib`, coordinates stay
in image pixels. `--ball-model data/weights/football-ball-detection.pt` adds a dedicated 1920 px ball pass.

To build a calibration JSON for a new clip:

```bash
uv run footlytics calib-assist data/clips/clip.mp4 --frame 0 --out data/out/assist
# open data/out/assist.jpg, pick the numbered intersections you recognise, copy them into
# data/out/assist.json -> template.points with their pitch vertex id (1-32) or landmark name
```

Weights: `data/weights/football-ball-detection.pt` and `football-pitch-detection.pt` come from the roboflow/sports
release (Google Drive) with a mirror on Hugging Face (`martinjolif/yolo-football-ball-detection`).

Outputs: `gamestate.parquet`, `gamestate.csv`, `overlay.mp4`, `radar.mp4`, `quality.json`.

## Stages

0. COCO YOLO11 + ByteTrack + HSV team clustering, image-space coordinates. Done.
1. Landmark calibration (frame 0) + camera-motion propagation -> metre coordinates, 2D radar. Done on the 30 s clip.
2. Dedicated ball detector, off-pitch filtering, shot-cut-robust camera anchoring. Done on the 3 min clip. ID switching still open.
3. Cloud comparison vs sn-gamestate and SoccerMaster.

License note: `ultralytics` is AGPL-3.0. Research use only until licensed or swapped.

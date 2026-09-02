"""Stage 2: dedicated ball detector inside the pipeline."""
from pathlib import Path

import pytest

from footlytics.pipeline import run_pipeline

CLIP = Path("data/clips/sample_3min.mp4")
CALIB = Path("data/calib/sample_calib_frame0.json")
BALL = Path("data/weights/football-ball-detection.pt")


@pytest.mark.skipif(not (CLIP.exists() and BALL.exists()), reason="clip or weights missing")
def test_dedicated_ball_model_raises_ball_coverage(tmp_path):
    base = run_pipeline(CLIP, tmp_path / "base", max_frames=40, model_name="yolo11n.pt")
    ded = run_pipeline(CLIP, tmp_path / "ded", max_frames=40, model_name="yolo11n.pt", ball_weights=BALL)
    assert ded["quality"]["ball_model"] == str(BALL)
    assert ded["quality"]["ball_coverage"] >= base["quality"]["ball_coverage"]
    assert ded["quality"]["ball_coverage"] >= 0.5

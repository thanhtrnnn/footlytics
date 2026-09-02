"""Stage 1 integration: manual calibration + camera motion -> metre coordinates and radar."""
from pathlib import Path

import pandas as pd
import pytest

from footlytics.pipeline import run_pipeline

CLIP = Path("data/clips/sample_30s.mp4")
CALIB = Path("data/clips/sample_calib_frame0.json")


@pytest.mark.skipif(not (CLIP.exists() and CALIB.exists()), reason="clip or calibration missing")
def test_calibrated_run_gives_metre_coordinates_and_radar(tmp_path):
    out = run_pipeline(CLIP, tmp_path, max_frames=20, model_name="yolo11n.pt", calib=CALIB)
    gs = pd.read_parquet(tmp_path / "gamestate.parquet")
    assert gs["calib_ok"].all()
    inside = gs["x_m"].between(-3, 108) & gs["y_m"].between(-3, 71)
    assert inside.mean() >= 0.9  # people on the pitch land on the pitch
    assert (tmp_path / "radar.mp4").exists()
    assert out["quality"]["calib_success_rate"] == 1.0
    assert "camera_motion_tracked_min" in out["quality"]

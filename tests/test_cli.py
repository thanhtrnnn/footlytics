"""End-to-end smoke on a few frames. Skipped when clip missing."""
import json
from pathlib import Path

import pandas as pd
import pytest

from footlytics.pipeline import run_pipeline

CLIP = Path("data/clips/sample_30s.mp4")


@pytest.mark.skipif(not CLIP.exists(), reason="sample clip not downloaded")
def test_run_pipeline_writes_all_artifacts(tmp_path):
    out = run_pipeline(CLIP, tmp_path, max_frames=15, model_name="yolo11n.pt")
    for name in ["gamestate.parquet", "gamestate.csv", "overlay.mp4", "quality.json", "quality.md"]:
        assert (tmp_path / name).exists(), name
    gs = pd.read_parquet(tmp_path / "gamestate.parquet")
    assert gs["frame"].max() <= 14
    assert set(gs["team"].unique()) <= {"0", "1", "ref", "unknown"}
    q = json.loads((tmp_path / "quality.json").read_text())
    assert q["frames"] >= 10
    assert out["quality"]["frames"] == q["frames"]

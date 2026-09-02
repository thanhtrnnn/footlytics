"""Integration smoke: detector + tracker on a few real frames. Skipped when clip missing."""
from pathlib import Path

import pytest

from footlytics.ingest import iter_frames, video_info
from footlytics.track import TRACK_COLUMNS, track_video

CLIP = Path("data/clips/sample_30s.mp4")


@pytest.mark.skipif(not CLIP.exists(), reason="sample clip not downloaded")
def test_ingest_reports_fps_and_frames():
    info = video_info(CLIP)
    assert info["fps"] == pytest.approx(25.0, abs=0.5)
    assert info["height"] == 720
    frames = list(iter_frames(CLIP, max_frames=3))
    assert len(frames) == 3
    assert frames[0][1].shape[0] == 720


@pytest.mark.skipif(not CLIP.exists(), reason="sample clip not downloaded")
def test_track_returns_schema_and_persistent_ids():
    tracks = track_video(CLIP, max_frames=10, model_name="yolo11n.pt")
    assert list(tracks.columns) == TRACK_COLUMNS
    assert tracks["frame"].max() <= 9
    assert (tracks["cls"].isin(["person", "sports ball"])).all()
    persons = tracks[tracks["cls"] == "person"]
    # at least a handful of players detected on a tactical-cam clip, ids persist across frames
    assert persons.groupby("frame").size().mean() >= 5
    assert persons.groupby("track_id")["frame"].nunique().max() >= 5

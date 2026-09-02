"""Dedicated ball detector on real frames. Skipped when clip or weights missing."""
from pathlib import Path

import pytest

from footlytics.ingest import iter_frames

CLIP = Path("data/clips/sample_3min.mp4")
WEIGHTS = Path("data/weights/football-ball-detection.pt")


@pytest.mark.skipif(not (CLIP.exists() and WEIGHTS.exists()), reason="clip or weights missing")
def test_ball_detector_finds_ball_in_most_sampled_frames():
    from footlytics.detect import BallDetector

    det = BallDetector(WEIGHTS, imgsz=1920)
    hits, n = 0, 0
    for idx, frame in iter_frames(CLIP, max_frames=1501):
        if idx % 150:
            continue
        n += 1
        res = det.detect(frame)
        if res is not None:
            x, y, conf = res
            assert 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0] and 0 < conf <= 1
            hits += 1
    assert n == 11
    assert hits / n >= 0.6

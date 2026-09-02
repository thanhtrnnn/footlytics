"""Calibration assist: detect pitch line intersections to label as landmarks. Skipped without frame."""
from pathlib import Path

import cv2
import numpy as np
import pytest

FRAME = Path("data/out/frame0.png")


@pytest.mark.skipif(not FRAME.exists(), reason="frame0.png missing")
def test_intersections_include_known_landmarks():
    from footlytics.calib_assist import detect_pitch_intersections

    frame = cv2.imread(str(FRAME))
    res = detect_pitch_intersections(frame, person_boxes=None)
    pts = np.array([[p["x"], p["y"]] for p in res["points"]])
    assert len(res["lines"]) >= 4
    assert len(pts) >= 4
    for known in [(13.4, 223.9), (370.4, 648.8), (1029.2, 364.4)]:  # halfway x far/near touchline, box corner
        d = np.linalg.norm(pts - np.array(known), axis=1).min()
        assert d < 8.0, (known, d)


@pytest.mark.skipif(not FRAME.exists(), reason="frame0.png missing")
def test_annotated_image_and_candidates_written(tmp_path):
    from footlytics.calib_assist import write_assist

    frame = cv2.imread(str(FRAME))
    out_img, out_json = write_assist(frame, tmp_path / "assist")
    assert out_img.exists() and out_json.exists()
    import json

    data = json.loads(out_json.read_text())
    assert "candidates" in data and all({"id", "x", "y"} <= set(c) for c in data["candidates"])
    assert data["template"]["points"] == [] and data["template"]["frame"] == 0

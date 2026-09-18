"""Moving-camera bridge end to end on the public tactical-cam clip. Skipped without clip/weights."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "data/clips/sample_30s.mp4"
CALIB = ROOT / "calib/brazil_france_30s_frame0.json"
WEIGHTS = ROOT / "weights/yolo_v8x6_finetuned.pt"


@pytest.mark.skipif(not (CLIP.exists() and CALIB.exists() and WEIGHTS.exists()), reason="clip, calibration or weights missing")
def test_run_with_anchored_camera_reports_calibrated_frames_and_on_pitch_players():
    import cv2

    from footlytics.geometry.camera_motion import AnchoredCamera
    from footlytics.geometry.homography import Calibration
    from footlytics.geometry.pitch import DEFAULT_PITCH
    from footlytics.perception.detect import Detector, DetectorConfig
    from footlytics.pipeline.radar import PipelineConfig, run
    from footlytics.state.schema import MatchMeta, TeamInfo

    cal = Calibration.load(CALIB, DEFAULT_PITCH.landmarks())
    ok, verdict = cal.verdict(DEFAULT_PITCH.landmarks())
    assert ok, verdict
    cap = cv2.VideoCapture(str(CLIP)); ok, anchor = cap.read(); cap.release()
    assert ok
    camera = AnchoredCamera(anchor)
    det = Detector(WEIGHTS, DetectorConfig(tile=0, imgsz=1280, device="mps", half=False, conf=0.25))
    meta = MatchMeta(match_id="BRA-FRA-30s", fps=25.0, source_type="tactical_cam",
                     home=TeamInfo("Brazil", "BRA"), away=TeamInfo("France", "FRA"))
    state, report = run(CLIP, cal, det, meta, PipelineConfig(max_frames=20, verbose=False), camera=camera)
    assert report["frames_calibrated"] == 20
    # the x6 weights have no referee class, so the pipeline must have tried to split officials off
    assert "officials_split" in report and report["officials_split"] >= 0
    assert report["frames_uncalibrated"] == 0
    assert report["median_players_per_frame"] >= 12
    p = state.players
    inside = (p.x.abs() <= 52.5 + 2) & (p.y.abs() <= 34 + 2)
    assert inside.mean() >= 0.95

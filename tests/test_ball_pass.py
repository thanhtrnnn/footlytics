"""Optional dedicated ball model inside Detector: replaces the ball row when it fires."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "data/clips/sample_3min.mp4"
W_MAIN = ROOT / "weights/yolo_v8x6_finetuned.pt"
W_BALL = ROOT / "weights/football-ball-detection.pt"


@pytest.mark.skipif(not (CLIP.exists() and W_MAIN.exists() and W_BALL.exists()), reason="clip or weights missing")
def test_ball_weights_raise_ball_hit_rate_on_sampled_frames():
    import cv2

    from footlytics.perception.detect import Detector, DetectorConfig
    from footlytics.state.schema import Role

    base = Detector(W_MAIN, DetectorConfig(tile=0, imgsz=1280, device="mps", half=False))
    ded = Detector(W_MAIN, DetectorConfig(tile=0, imgsz=1280, device="mps", half=False,
                                          ball_weights=str(W_BALL), ball_imgsz=1920))
    cap = cv2.VideoCapture(str(CLIP))
    hits_base = hits_ded = n = 0
    for idx in range(0, 1501, 150):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        assert ok
        frame = bgr[:, :, ::-1]
        n += 1
        for det, tally in ((base, "b"), (ded, "d")):
            out = det.detect(frame)
            balls = [r for r in out if det.role_of(r[5]) == Role.BALL.value]
            assert len(balls) <= det.cfg.max_balls      # candidates; the pipeline picks one
            if balls:
                if tally == "b":
                    hits_base += 1
                else:
                    hits_ded += 1
    cap.release()
    assert n == 11
    assert hits_ded >= hits_base
    assert hits_ded / n >= 0.6

"""Detectors beyond the COCO tracker: dedicated ball detector."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class BallDetector:
    """Single-class YOLO ball detector. Returns the most confident ball centre or None."""

    def __init__(self, weights: str | Path, device: str | None = None, imgsz: int = 1920, conf: float = 0.25):
        from ultralytics import YOLO

        from footlytics.track import default_device

        self._model = YOLO(str(weights))
        self._device = device or default_device()
        self._imgsz = imgsz
        self._conf = conf

    def detect(self, frame: np.ndarray) -> tuple[float, float, float] | None:
        r = self._model.predict(frame, imgsz=self._imgsz, conf=self._conf, device=self._device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        confs = r.boxes.conf.cpu().numpy()
        i = int(confs.argmax())
        x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
        return float((x1 + x2) / 2), float((y1 + y2) / 2), float(confs[i])

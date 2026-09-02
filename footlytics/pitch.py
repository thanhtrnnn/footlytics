"""Pitch keypoint detection and per-frame homography.

Keypoint layout follows roboflow/sports SoccerPitchConfiguration (32 vertices, 120x70 m pitch),
rescaled here to the FIFA standard 105x68 m used by footlytics.homography.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from footlytics.homography import PITCH_LENGTH_M, PITCH_WIDTH_M, fit_homography

_L, _W = 120.0, 70.0
_PB_W, _PB_L = 41.0, 20.15
_GB_W, _GB_L = 18.32, 5.5
_CC_R, _PS = 9.15, 11.0
_RAW_VERTICES = [
    (0, 0), (0, (_W - _PB_W) / 2), (0, (_W - _GB_W) / 2), (0, (_W + _GB_W) / 2), (0, (_W + _PB_W) / 2), (0, _W),
    (_GB_L, (_W - _GB_W) / 2), (_GB_L, (_W + _GB_W) / 2), (_PS, _W / 2),
    (_PB_L, (_W - _PB_W) / 2), (_PB_L, (_W - _GB_W) / 2), (_PB_L, (_W + _GB_W) / 2), (_PB_L, (_W + _PB_W) / 2),
    (_L / 2, 0), (_L / 2, _W / 2 - _CC_R), (_L / 2, _W / 2 + _CC_R), (_L / 2, _W),
    (_L - _PB_L, (_W - _PB_W) / 2), (_L - _PB_L, (_W - _GB_W) / 2), (_L - _PB_L, (_W + _GB_W) / 2), (_L - _PB_L, (_W + _PB_W) / 2),
    (_L - _PS, _W / 2), (_L - _GB_L, (_W - _GB_W) / 2), (_L - _GB_L, (_W + _GB_W) / 2),
    (_L, 0), (_L, (_W - _PB_W) / 2), (_L, (_W - _GB_W) / 2), (_L, (_W + _GB_W) / 2), (_L, (_W + _PB_W) / 2), (_L, _W),
    (_L / 2 - _CC_R, _W / 2), (_L / 2 + _CC_R, _W / 2),
]
# Rescale x by 105/120 and y by 68/70 so vertices land on the 105x68 frame.
PITCH_VERTICES_M: list[tuple[float, float]] = [
    (x * PITCH_LENGTH_M / _L, y * PITCH_WIDTH_M / _W) for x, y in _RAW_VERTICES
]


def homography_from_keypoints(
    xy: np.ndarray, conf: np.ndarray, min_conf: float = 0.5, ransac_thresh_m: float = 2.0
) -> tuple[np.ndarray | None, float, int]:
    """Fit image->pitch homography from the 32 detected keypoints, using confident ones only.

    Returns (H or None, median reprojection error in metres, number of keypoints used).
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    conf = np.asarray(conf, dtype=np.float64).ravel()
    keep = conf >= min_conf
    n_used = int(keep.sum())
    if n_used < 4:
        return None, float("nan"), n_used
    H, err = fit_homography(xy[keep], np.asarray(PITCH_VERTICES_M)[keep], ransac_thresh_m=ransac_thresh_m)
    return H, err, n_used


class PitchKeypointDetector:
    """YOLO-pose model predicting the 32 pitch keypoints."""

    def __init__(self, weights: str | Path, device: str | None = None, imgsz: int = 1280, conf: float = 0.3):
        from ultralytics import YOLO

        from footlytics.track import default_device

        self._model = YOLO(str(weights))
        self._device = device or default_device()
        self._imgsz = imgsz
        self._conf = conf

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (xy [32,2] in pixels, conf [32]). Zeros when no pitch box is found."""
        r = self._model.predict(frame, imgsz=self._imgsz, conf=self._conf, device=self._device, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return np.zeros((32, 2)), np.zeros(32)
        # pick the detection with the highest total keypoint confidence
        confs = r.keypoints.conf.cpu().numpy()
        best = int(confs.sum(axis=1).argmax())
        return r.keypoints.xy[best].cpu().numpy().astype(np.float64), confs[best].astype(np.float64)

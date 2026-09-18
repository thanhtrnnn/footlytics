"""Automatic calibration from a pitch-keypoint model.

An alternative to clicking landmarks by hand: a YOLO-pose model predicts the 32
roboflow/sports `SoccerPitchConfiguration` vertices, and those correspondences
are fitted into a `Calibration` like any other.

The 32-vertex ordering is the *model's output contract*, so it lives here beside
the detector rather than in `geometry.pitch` -- that module's 43 named landmarks
are derived from the Laws of the Game and are a different, human-facing set.

Carried over from the V0 prototype. Not wired into `pipeline.run`: `--calib` and
a hand-annotated anchor frame remain the documented path, because on the footage
measured so far the learned calibrators (see also `lines_nn.py`) are not reliable
enough to trust unattended.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from footlytics.geometry.homography import Calibration
from footlytics.geometry.pitch import DEFAULT_PITCH, Pitch

_L, _W = 120.0, 70.0                       # roboflow config's nominal pitch
_PB_W, _PB_L = 41.0, 20.15                 # penalty box
_GB_W, _GB_L = 18.32, 5.5                  # goal box
_CC_R, _PS = 9.15, 11.0                    # centre circle radius, penalty spot
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


def pitch_vertices(pitch: Pitch = DEFAULT_PITCH) -> np.ndarray:
    """The 32 vertices in Match State coordinates: metres, origin at the centre spot.

    The roboflow layout is defined on a 120x70 pitch with the origin in a corner,
    so it is rescaled to the real pitch and shifted to centre origin. Skipping the
    shift is the easy mistake here: it fits without complaint and puts every
    player half a pitch away from where they are.
    """
    v = np.asarray(_RAW_VERTICES, dtype=np.float64)
    return np.column_stack([
        v[:, 0] * pitch.length / _L - pitch.half_l,
        v[:, 1] * pitch.width / _W - pitch.half_w,
    ])


#: The 32 vertices on the default 105x68 pitch, centre origin.
PITCH_VERTICES_M: np.ndarray = pitch_vertices()


def homography_from_keypoints(
    xy: np.ndarray,
    conf: np.ndarray,
    image_size: tuple[int, int],
    min_conf: float = 0.5,
    ransac_thresh_m: float = 2.0,
    pitch: Pitch = DEFAULT_PITCH,
) -> tuple[Optional[Calibration], float, int]:
    """Fit image->pitch from the 32 keypoints, using confident ones only.

    Returns `(Calibration or None, median reprojection error in metres, n used)`.

    RANSAC is used rather than `homography.fit_homography`'s plain DLT: a keypoint
    model puts occasional vertices in completely the wrong place, and one such
    outlier drags a least-squares fit across the whole pitch.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    conf = np.asarray(conf, dtype=np.float64).ravel()
    target = pitch_vertices(pitch)
    if len(xy) != len(target):
        raise ValueError(f"expected {len(target)} keypoints, got {len(xy)}")

    keep = conf >= min_conf
    n_used = int(keep.sum())
    if n_used < 4:
        return None, float("nan"), n_used

    src, dst = xy[keep], target[keep]
    if n_used > 4:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh_m)
    else:
        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    if H is None:
        return None, float("nan"), n_used

    # Median over every keypoint used, not just the RANSAC inliers: an outlier the
    # fit ignored is exactly what a caller deciding whether to trust this needs to see.
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.median(np.linalg.norm(proj - dst, axis=1)))
    cal = Calibration(H=np.asarray(H, dtype=np.float64), image_size=tuple(image_size),
                      notes=f"keypoints: {n_used}/{len(target)} vertices, median {err:.2f} m")
    return cal, err, n_used


class PitchKeypointDetector:
    """YOLO-pose model predicting the 32 pitch keypoints."""

    def __init__(self, weights: str | Path, device: Optional[str] = None,
                 imgsz: int = 1280, conf: float = 0.3):
        from ultralytics import YOLO

        from footlytics.config import default_device

        self._model = YOLO(str(weights))
        self._device = device or default_device()
        self._imgsz = imgsz
        self._conf = conf

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return `(xy [32,2] in pixels, conf [32])`. Zeros when no pitch is found."""
        r = self._model.predict(frame, imgsz=self._imgsz, conf=self._conf,
                                device=self._device, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return np.zeros((32, 2)), np.zeros(32)
        # The frame holds one pitch: keep the detection whose keypoints are most confident.
        confs = r.keypoints.conf.cpu().numpy()
        best = int(confs.sum(axis=1).argmax())
        return r.keypoints.xy[best].cpu().numpy().astype(np.float64), confs[best].astype(np.float64)


def calibrate_frame(
    frame: np.ndarray,
    weights: str | Path,
    device: Optional[str] = None,
    imgsz: int = 1280,
    min_conf: float = 0.5,
    pitch: Pitch = DEFAULT_PITCH,
) -> tuple[Optional[Calibration], float, int]:
    """Detect keypoints on one frame and fit a `Calibration` from them."""
    det = PitchKeypointDetector(weights, device=device, imgsz=imgsz)
    xy, conf = det.detect(frame)
    h, w = frame.shape[:2]
    return homography_from_keypoints(xy, conf, image_size=(w, h),
                                     min_conf=min_conf, pitch=pitch)

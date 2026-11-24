"""Calibrate every frame of a broadcast clip: solve on keyframes, bridge between them by tracking.

A solve costs about a second (5.5 s with PnLCalib's full voting), so only every `every`-th
frame is solved, with one solver setting. A solve whose reprojection error exceeds `gate_px` is
rejected: on the V.League clip the good windows were at 3-24 px and the bad ones above 40.
Frames between accepted keyframes are related to the last one by `AnchoredCamera`. The
broadcast's static overlays (score bug, channel logo) are masked out of its feature search,
because features that never move would pin the motion estimate to zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

from .calib_nn import CameraFit
from .camera_motion import AnchoredCamera
from .homography import apply_homography
from .sn_camera import camera_from_homography


def _safe_inv(H: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Matrix inverse returning None on singular, non-finite, or ill-conditioned homographies."""
    if H is None or not np.isfinite(H).all():
        return None
    try:
        inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    return inv if np.isfinite(inv).all() else None


def _pitch_disagreement_m(H1: np.ndarray, H2: np.ndarray, w: int, h: int) -> float:
    """Median pitch-plane disagreement in metres across a 4x3 lower-image grid."""
    us = np.linspace(0.2 * w, 0.8 * w, 4)
    vs = np.linspace(0.5 * h, 0.9 * h, 3)
    pts = np.array([(u, v) for v in vs for u in us], float)
    p1 = apply_homography(H1, pts)
    p2 = apply_homography(H2, pts)
    ok = np.isfinite(p1).all(1) & np.isfinite(p2).all(1)
    if not ok.any():
        return float("inf")
    return float(np.median(np.linalg.norm(p1[ok] - p2[ok], axis=1)))


@dataclass
class FrameCalib:
    frame_idx: int
    H_img_to_pitch: Optional[np.ndarray]
    cam: Optional[dict]
    source: str                      # "keyframe" | "bridged" | "none"
    rep_err: Optional[float] = None


def static_overlay_mask(frames_bgr: list[np.ndarray], std_thresh: float = 3.0,
                        dilate_px: int = 9, max_share: float = 0.25,
                        band: float = 0.12) -> np.ndarray:
    """255 where features may be tracked, 0 on pixels that do not change while the picture
    does. If more than `max_share` of the frame is static the camera itself is still and the
    test says nothing; then only the top `band` (where broadcasters put overlays) is masked."""
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]).astype(np.float32)
    static = g.std(axis=0) < std_thresh
    h = g.shape[1]
    mask = np.full(g.shape[1:], 255, np.uint8)
    if static.mean() > max_share:
        mask[: int(round(band * h))] = 0
        return mask
    k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    mask[cv2.dilate(static.astype(np.uint8), k) > 0] = 0
    return mask


class ClipCalibrator:
    def __init__(self, fit_keyframes: Callable[[list, list], dict[int, Optional[CameraFit]]],
                 every: int = 25, gate_px: float = 25.0,
                 tracker_factory: Callable = AnchoredCamera, chunk: int = 30,
                 bridge_max_m: Optional[float] = 3.0):
        self.fit_keyframes, self.every, self.gate_px = fit_keyframes, every, gate_px
        self.tracker_factory, self.chunk = tracker_factory, chunk
        self.bridge_max_m = bridge_max_m
        self.keyframes = self.keyframes_accepted = 0

    def run(self, frames_bgr: Iterable[np.ndarray],
            mask: Optional[np.ndarray] = None) -> list[FrameCalib]:
        out: list[FrameCalib] = []
        anchor = None                                   # (tracker, H_img_to_pitch at anchor)
        pending_H: Optional[np.ndarray] = None
        buf: list[np.ndarray] = []

        def flush(start: int) -> None:
            nonlocal anchor, pending_H
            keys = [i for i in range(len(buf)) if (start + i) % self.every == 0]
            fits = self.fit_keyframes(buf, keys) if keys else {}
            self.keyframes += len(keys)
            for i, f in enumerate(buf):
                idx = start + i
                h, w = f.shape[:2]
                bridged_H: Optional[np.ndarray] = None
                if anchor is not None:
                    H_a_t = anchor[0].update(f, mask)
                    inv_a_t = _safe_inv(H_a_t)
                    if inv_a_t is not None:
                        H_b = anchor[1] @ inv_a_t
                        if np.isfinite(H_b).all() and abs(float(H_b[2, 2])) > 1e-12:
                            cand = H_b / H_b[2, 2]
                            if _safe_inv(cand) is not None:
                                bridged_H = cand
                fit = fits.get(i)
                if fit is not None and fit.rep_err <= self.gate_px:
                    accept = True
                    if self.bridge_max_m is not None and self.every > 1 and bridged_H is not None:
                        dis = _pitch_disagreement_m(fit.H_img_to_pitch, bridged_H, w, h)
                        if dis > self.bridge_max_m:
                            accept = (
                                pending_H is not None
                                and _pitch_disagreement_m(fit.H_img_to_pitch, pending_H, w, h)
                                <= self.bridge_max_m
                            )
                    if accept:
                        pending_H = None
                        self.keyframes_accepted += 1
                        anchor = (self.tracker_factory(f, mask), fit.H_img_to_pitch)
                        out.append(FrameCalib(idx, fit.H_img_to_pitch, fit.cam, "keyframe",
                                              fit.rep_err))
                        continue
                    pending_H = fit.H_img_to_pitch
                err = fit.rep_err if fit is not None else None
                if bridged_H is not None:
                    inv_b = _safe_inv(bridged_H)
                    if inv_b is not None:
                        cam = camera_from_homography(inv_b, w, h)
                        out.append(FrameCalib(idx, bridged_H, cam, "bridged", err))
                        continue
                out.append(FrameCalib(idx, None, None, "none", err))

        start = 0
        for f in frames_bgr:
            buf.append(f)
            if len(buf) == self.chunk:
                flush(start)
                start += len(buf)
                buf = []
        if buf:
            flush(start)
        return out

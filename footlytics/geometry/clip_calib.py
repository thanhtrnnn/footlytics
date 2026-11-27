"""Calibrate every frame of a broadcast clip: solve on keyframes, bridge between them by tracking.

A solve costs about a second (5.5 s with PnLCalib's full voting), so only every `every`-th
frame is solved, with one solver setting. A solve whose reprojection error exceeds `gate_px` is
rejected: on the V.League clip the good windows were at 3-24 px and the bad ones above 40.
Frames between accepted keyframes are related to the last one by `AnchoredCamera`. The
accepted cameras are then corrected across keyframes (`camera_smooth`): a keyframe replaces the
bridged homography outright, and the difference (median 0.22 m, up to metres) moved every player
on the pitch in one frame. The broadcast's static overlays (score bug, channel logo) are masked out of its feature search,
because features that never move would pin the motion estimate to zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

from .calib_nn import CameraFit
from .camera_motion import AnchoredCamera
from .camera_smooth import smooth_homographies
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


def _far_disagreement_m(H1: np.ndarray, H2: np.ndarray, w: int, h: int) -> float:
    """Worst pitch-plane disagreement over the far field (image rows 30-40%, full width).

    The lower-image grid of `_pitch_disagreement_m` is where players are, not where a fit goes
    wrong first: depth sensitivity is ~0.23 m/px at row 0.30H against 0.03 at 0.9H, so a fit
    that is a few pixels off at the far touchline is metres off there and invisible below.
    """
    us = np.linspace(0.1 * w, 0.9 * w, 5)
    pts = np.array([(u, v * h) for v in (0.30, 0.40) for u in us], float)
    p1, p2 = apply_homography(H1, pts), apply_homography(H2, pts)
    ok = np.isfinite(p1).all(1) & np.isfinite(p2).all(1)
    if not ok.any():
        return float("inf")
    return float(np.linalg.norm(p1[ok] - p2[ok], axis=1).max())


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
    """`bridge_max_m`: a keyframe may differ from the bridged camera by this much (median over
    the lower image) ... and by `bridge_far_max_m` at worst over the far field, where the
    lower-image grid cannot see an error. `min_keypoints`: a fit that reports how many keypoints
    it used (`n_keypoints`) needs at least this many; four make the homography exact, so the
    reprojection gate (`gate_px`) is blind to a wrong one. `smooth` removes the step at each
    accepted keyframe (the drift of the carried camera) and then smooths what is left with a
    Gaussian of `smooth_sigma` frames; cuts and gaps are never smoothed across. `camera_smooth`."""

    def __init__(self, fit_keyframes: Callable[[list, list], dict[int, Optional[CameraFit]]],
                 every: int = 25, gate_px: float = 25.0,
                 tracker_factory: Callable = AnchoredCamera, chunk: int = 30,
                 bridge_max_m: Optional[float] = 3.0, bridge_far_max_m: Optional[float] = 7.5,
                 min_keypoints: int = 5, smooth: bool = True, smooth_sigma: float = 3.0):
        self.fit_keyframes, self.every, self.gate_px = fit_keyframes, every, gate_px
        self.tracker_factory, self.chunk = tracker_factory, chunk
        self.bridge_max_m, self.bridge_far_max_m = bridge_max_m, bridge_far_max_m
        self.min_keypoints, self.smooth, self.smooth_sigma = min_keypoints, smooth, smooth_sigma
        self.keyframes = self.keyframes_accepted = 0
        self.keyframes_rejected_keypoints = 0
        self.smooth_info = None
        self._wh: Optional[tuple[int, int]] = None

    def _agrees(self, H1: np.ndarray, H2: np.ndarray, w: int, h: int) -> bool:
        """Does a keyframe fit `H1` agree with the camera `H2` (lower image and far field)?"""
        if self.bridge_max_m is None:
            return True
        if _pitch_disagreement_m(H1, H2, w, h) > self.bridge_max_m:
            return False
        return (self.bridge_far_max_m is None
                or _far_disagreement_m(H1, H2, w, h) <= self.bridge_far_max_m)

    def _redundant(self, fit: CameraFit) -> bool:
        n = getattr(fit, "n_keypoints", None)
        return n is None or n >= self.min_keypoints

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
                self._wh = (w, h)
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
                if fit is not None and not self._redundant(fit):
                    self.keyframes_rejected_keypoints += 1
                    fit = None
                if fit is not None and fit.rep_err <= self.gate_px:
                    accept = True
                    if self.bridge_max_m is not None and self.every > 1 and bridged_H is not None:
                        if not self._agrees(fit.H_img_to_pitch, bridged_H, w, h):
                            accept = (
                                pending_H is not None
                                and self._agrees(fit.H_img_to_pitch, pending_H, w, h)
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
        if self.smooth and self._wh is not None:
            out = self._smooth(out)
        return out

    def _smooth(self, out: list[FrameCalib]) -> list[FrameCalib]:
        w, h = self._wh
        Hs, self.smooth_info = smooth_homographies(
            [c.H_img_to_pitch for c in out], w, h, key=[c.source == "keyframe" for c in out],
            sigma=self.smooth_sigma)
        res: list[FrameCalib] = []
        for c, Hn in zip(out, Hs):
            if Hn is None or Hn is c.H_img_to_pitch:
                res.append(c)
                continue
            inv = _safe_inv(Hn)
            cam = camera_from_homography(inv, w, h) if inv is not None else None
            res.append(FrameCalib(c.frame_idx, Hn, cam if cam is not None else c.cam,
                                  c.source, c.rep_err))
        return res

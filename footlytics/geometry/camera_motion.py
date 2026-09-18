"""Frame-to-frame camera motion as an accumulated homography (KLT + RANSAC).

Use: H_t(image_t -> pitch) = H_0(image_0 -> pitch) @ inv(H_0_to_t).
Features are re-detected every frame on the previous image, restricted by an optional mask
(e.g. exclude player boxes). Falls back to the previous motion when tracking is weak.
"""
from __future__ import annotations

import cv2
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .homography import Calibration

_FEATURE_PARAMS = dict(maxCorners=600, qualityLevel=0.01, minDistance=8, blockSize=7)
_LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


class AnchoredCamera:
    """Camera motion relative to one anchor frame, robust to shot cuts.

    Per frame: KLT step from the previous frame (fast). Periodically, and whenever KLT fails
    (few inliers -> likely a cut or a blur), re-anchor by global SIFT matching to the anchor
    frame; if that fails too the frame is marked invalid. `H_anchor_to_t` maps anchor pixels
    to current pixels; use H_t = H_anchor @ inv(H_anchor_to_t).
    """

    def __init__(self, anchor_bgr: np.ndarray, mask: np.ndarray | None = None, reanchor_every: int = 50,
                 min_klt_inliers: int = 12, min_klt_ratio: float = 0.5, min_global_inliers: int = 30,
                 min_global_ratio: float = 0.5, start_at_anchor: bool = True, retry_every: int = 10,
                 sift_scale: float = 0.5, max_key_shift_px: float = 60.0):
        self._anchor_gray = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2GRAY) if anchor_bgr.ndim == 3 else anchor_bgr
        self._scale = sift_scale
        self._S = np.diag([sift_scale, sift_scale, 1.0])  # full-res px -> matching-res px
        self._Sinv = np.linalg.inv(self._S)
        self._sift = cv2.SIFT_create(nfeatures=2000)
        self._kp_a, self._des_a = self._sift.detectAndCompute(self._small(self._anchor_gray), self._small(mask))
        self._retry_every = retry_every
        self._scene_change_thresh = 20.0  # mean abs gray difference between consecutive (half-res) frames
        self._matcher = cv2.BFMatcher()
        self._reanchor_every = reanchor_every
        self._min_klt, self._min_klt_ratio = min_klt_inliers, min_klt_ratio
        self._min_glob, self._min_glob_ratio = min_global_inliers, min_global_ratio
        self.H_anchor_to_t = np.eye(3)
        self.valid = start_at_anchor
        self._prev_gray = self._anchor_gray if start_at_anchor else None
        # Keyframe: the last frame whose pose is known from a global match (or a chaining
        # hand-over). Tracking key->t instead of (t-1)->t means a fixed camera never
        # accumulates per-frame noise; chaining is only used once the camera has moved
        # further than one LK step can bridge.
        self._key_gray = self._anchor_gray if start_at_anchor else None
        self._H_key = np.eye(3)
        self.max_key_shift_px = max_key_shift_px
        self._count = 0
        self.n_reanchors = 0
        self.n_tracked = 0

    def _small(self, img: np.ndarray | None) -> np.ndarray | None:
        if img is None or self._scale == 1.0:
            return img
        return cv2.resize(img, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)

    def _klt_step(self, gray: np.ndarray, mask: np.ndarray | None, ref: np.ndarray | None = None) -> np.ndarray | None:
        ref = self._prev_gray if ref is None else ref
        p0 = cv2.goodFeaturesToTrack(ref, mask=mask, **_FEATURE_PARAMS)
        if p0 is None or len(p0) < self._min_klt:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(ref, gray, p0, None, **_LK_PARAMS)
        ok = st.ravel() == 1
        q0, q1 = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
        if len(q0) < self._min_klt:
            return None
        step, inl = cv2.findHomography(q0, q1, cv2.RANSAC, 2.0)
        n = int(inl.sum()) if inl is not None else 0
        self.n_tracked = n
        if step is None or n < self._min_klt or n / len(q0) < self._min_klt_ratio:
            return None
        return step

    def _global(self, gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
        kp, des = self._sift.detectAndCompute(self._small(gray), self._small(mask))
        if des is None or self._des_a is None or len(kp) < 8:
            return None
        matches = self._matcher.knnMatch(self._des_a, des, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < self._min_glob:
            return None
        src = np.float32([self._kp_a[g.queryIdx].pt for g in good])
        dst = np.float32([kp[g.trainIdx].pt for g in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0 * self._scale)
        n = int(inl.sum()) if inl is not None else 0
        if H is None or n < self._min_glob or n / len(good) < self._min_glob_ratio:
            return None
        return self._Sinv @ H @ self._S  # back to full-resolution pixels

    def _scene_changed(self, gray: np.ndarray) -> bool:
        a, b = self._small(self._prev_gray), self._small(gray)
        return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) > self._scene_change_thresh

    def _corner_shift(self, H: np.ndarray, shape: tuple[int, int]) -> float:
        h, w = shape
        c = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2)
        return float(np.abs(cv2.perspectiveTransform(c, H).reshape(-1, 2) - c.reshape(-1, 2)).max())

    def update(self, frame: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        self._count += 1
        self._prev_valid = self.valid
        scene_change = self._prev_gray is not None and self._scene_changed(gray)
        step = None
        if self.valid and not scene_change:
            # Prefer a single estimate from the keyframe: no accumulation on a fixed camera.
            if self._key_gray is not None:
                step_key = self._klt_step(gray, mask, ref=self._key_gray)
                if step_key is not None and self._corner_shift(step_key, gray.shape) <= self.max_key_shift_px:
                    self.H_anchor_to_t = step_key @ self._H_key
                    step = step_key
            if step is None and self._prev_gray is not None:
                # Camera moved beyond one LK step from the keyframe: chain, and hand the
                # keyframe over to the current frame so the next frames track from here.
                step_prev = self._klt_step(gray, mask)
                if step_prev is not None:
                    self.H_anchor_to_t = step_prev @ self.H_anchor_to_t
                    self._key_gray, self._H_key = gray, self.H_anchor_to_t.copy()
                    step = step_prev
        if step is None:
            self.valid = False
        need_global = (not self.valid and self._count % self._retry_every == 0) or (
            self.valid and self._count % self._reanchor_every == 0) or (not self.valid and self._count <= 1)
        if need_global or scene_change or (not self.valid and self._prev_valid):
            Hg = self._global(gray, mask)
            if Hg is not None:
                self.H_anchor_to_t = Hg
                self.valid = True
                self.n_reanchors += 1
                self._key_gray, self._H_key = gray, Hg.copy()
        self._prev_gray = gray
        return self.H_anchor_to_t if self.valid else None


class MovingCalibration:
    """Per-frame image->pitch mapping: anchor `Calibration` composed with camera motion."""

    def __init__(self, anchor: "Calibration"):
        self.anchor = anchor

    def feet_to_pitch(self, bboxes, H_anchor_to_t: np.ndarray) -> np.ndarray:
        """bboxes are (x, y, w, h) in frame-t pixels; H_anchor_to_t maps anchor px -> frame-t px."""
        b = np.asarray(bboxes, dtype=np.float64).reshape(-1, 4)
        feet_t = np.stack([b[:, 0] + b[:, 2] / 2.0, b[:, 1] + b[:, 3]], axis=1)
        feet_anchor = cv2.perspectiveTransform(feet_t.reshape(-1, 1, 2), np.linalg.inv(H_anchor_to_t)).reshape(-1, 2)
        return self.anchor.image_to_pitch(feet_anchor)

    def pitch_to_image(self, pts, H_anchor_to_t: np.ndarray) -> np.ndarray:
        px_anchor = self.anchor.pitch_to_image(pts)
        return cv2.perspectiveTransform(np.asarray(px_anchor, np.float64).reshape(-1, 1, 2), H_anchor_to_t).reshape(-1, 2)

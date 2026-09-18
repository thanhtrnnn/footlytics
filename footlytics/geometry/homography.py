"""Image <-> pitch mapping.

For a fixed camera this is solved *once per venue* and then reused for every
match at that ground, which is the single biggest reason a Bepro-style rig is
easier than broadcast: no per-frame calibration network, no drift, no failure
mode where the model loses the pitch during a goalmouth scramble.

Two mappings are supported:

`homography`
    A single 3x3 planar projective transform. Correct for one real pinhole
    camera looking at a flat pitch.

`homography + thin-plate spline`
    A stitched panorama is *not* a single pinhole view -- the stitch seams bend
    straight lines, so one homography leaves systematic residuals of a metre or
    more near the seams. The TPS soaks up that residual. Enable it only when
    the reprojection report says you need it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------- helpers

def _as_xy(pts) -> Array:
    a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if a.size == 0:
        raise ValueError("no points given")
    return a


def _normalise(pts: Array) -> tuple[Array, Array]:
    """Hartley normalisation: centre on the mean, scale mean distance to sqrt(2).

    Skipping this is the classic reason a hand-rolled DLT looks fine on toy data
    and falls apart on 4K pixel coordinates.
    """
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1).mean()
    s = np.sqrt(2) / d if d > 1e-12 else 1.0
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return (pts - c) * s, T


def fit_homography(src: Sequence, dst: Sequence) -> Array:
    """Least-squares DLT homography mapping `src` -> `dst`. Needs >= 4 points."""
    src, dst = _as_xy(src), _as_xy(dst)
    if len(src) != len(dst):
        raise ValueError(f"point count mismatch: {len(src)} vs {len(dst)}")
    if len(src) < 4:
        raise ValueError(f"need at least 4 correspondences, got {len(src)}")

    sn, Ts = _normalise(src)
    dn, Td = _normalise(dst)

    A = np.zeros((2 * len(sn), 9))
    for i, ((x, y), (u, v)) in enumerate(zip(sn, dn)):
        A[2 * i]     = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]

    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_homography(H: Array, pts) -> Array:
    pts = _as_xy(pts)
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    out = hom @ H.T
    w = out[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, np.nan, w)   # points on the horizon
    return out[:, :2] / w


class _TPS:
    """Thin-plate spline warp through control points. Hand-rolled rather than
    cv2's shape transformer, whose source/target argument order is a trap."""

    def __init__(self, src: Array, dst: Array, smooth: float = 0.0):
        self.src = src
        n = len(src)
        K = self._U(np.linalg.norm(src[:, None, :] - src[None, :, :], axis=-1))
        K[np.diag_indices(n)] = smooth
        P = np.hstack([np.ones((n, 1)), src])
        L = np.zeros((n + 3, n + 3))
        L[:n, :n], L[:n, n:], L[n:, :n] = K, P, P.T
        Y = np.vstack([dst, np.zeros((3, 2))])
        self.coef = np.linalg.solve(L + 1e-10 * np.eye(n + 3), Y)

    @staticmethod
    def _U(r: Array) -> Array:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r > 1e-12, r**2 * np.log(r**2), 0.0)

    def __call__(self, pts: Array) -> Array:
        pts = _as_xy(pts)
        K = self._U(np.linalg.norm(pts[:, None, :] - self.src[None, :, :], axis=-1))
        P = np.hstack([np.ones((len(pts), 1)), pts])
        return np.hstack([K, P]) @ self.coef


# --------------------------------------------------------------- calibration

@dataclass
class Calibration:
    """A frozen image->pitch mapping for one fixed camera view."""

    H: Array                                   # image px -> pitch metres
    image_size: tuple[int, int]                # (width, height)
    named_points: dict[str, tuple[float, float]] = field(default_factory=dict)
    use_tps: bool = False
    notes: str = ""
    _tps_fwd: Optional[_TPS] = field(default=None, repr=False, compare=False)
    _tps_inv: Optional[_TPS] = field(default=None, repr=False, compare=False)

    # ---------------------------------------------------------------- build

    @classmethod
    def from_correspondences(
        cls,
        image_points: dict[str, tuple[float, float]],
        pitch_landmarks: dict[str, tuple[float, float]],
        image_size: tuple[int, int],
        use_tps: bool = False,
        notes: str = "",
    ) -> "Calibration":
        """Fit from `{landmark_name: (px, py)}` clicked on one frame.

        `pitch_landmarks` is `Pitch.landmarks()`. Unknown names are rejected
        loudly rather than silently dropped -- a typo'd landmark name that gets
        ignored produces a calibration that is subtly, invisibly wrong.
        """
        unknown = set(image_points) - set(pitch_landmarks)
        if unknown:
            raise KeyError(f"unknown landmark name(s): {sorted(unknown)}")
        if len(image_points) < 4:
            raise ValueError(
                f"need >= 4 landmarks, got {len(image_points)}. Spread them across "
                "the whole frame -- 4 points clustered in one half give a "
                "homography that explodes at the far end."
            )

        names = sorted(image_points)
        src = np.array([image_points[n] for n in names], dtype=np.float64)
        dst = np.array([pitch_landmarks[n] for n in names], dtype=np.float64)

        H = fit_homography(src, dst)
        cal = cls(H=H, image_size=image_size, named_points=dict(image_points),
                  use_tps=use_tps, notes=notes)
        if use_tps:
            resid = apply_homography(H, src)
            cal._tps_fwd = _TPS(resid, dst)
            cal._tps_inv = _TPS(dst, resid)
        return cal

    # --------------------------------------------------------------- mapping

    def image_to_pitch(self, pts) -> Array:
        out = apply_homography(self.H, pts)
        return self._tps_fwd(out) if self._tps_fwd is not None else out

    def pitch_to_image(self, pts) -> Array:
        pts = _as_xy(pts)
        if self._tps_inv is not None:
            pts = self._tps_inv(pts)
        return apply_homography(np.linalg.inv(self.H), pts)

    def feet_to_pitch(self, bboxes) -> Array:
        """Map detections to the ground plane via the *bottom centre* of the box.

        A homography only maps the ground plane, and only a player's feet are on
        it. Using the box centre instead projects a point ~1 m in the air onto
        the grass, displacing the player *away from the camera* by an amount that
        grows with obliquity: on a synthetic 20 m-high camera 75 m back, that is
        6.7 m of error (see scripts/test_geometry.py). It is the single easiest
        way to corrupt every distance, line-height and offside number downstream.
        """
        b = np.asarray(bboxes, dtype=np.float64).reshape(-1, 4)
        feet = np.stack([b[:, 0] + b[:, 2] / 2.0, b[:, 1] + b[:, 3]], axis=1)
        return self.image_to_pitch(feet)

    # ------------------------------------------------------------ diagnostics

    def reprojection_error(self, pitch_landmarks: dict) -> dict:
        """Residuals in metres at the calibration points themselves.

        Optimistic by construction (these are the points we fitted to), so treat
        it as a floor: if it is already bad, nothing downstream can be trusted.
        """
        names = sorted(self.named_points)
        src = np.array([self.named_points[n] for n in names])
        truth = np.array([pitch_landmarks[n] for n in names])
        err = np.linalg.norm(self.image_to_pitch(src) - truth, axis=1)
        return {
            "n_points": len(names),
            "mean_m": float(err.mean()),
            "median_m": float(np.median(err)),
            "max_m": float(err.max()),
            "worst_landmark": names[int(err.argmax())],
            "per_point_m": {n: float(e) for n, e in zip(names, err)},
        }

    def verdict(self, pitch_landmarks: dict) -> tuple[bool, str]:
        """Is this calibration good enough to build analytics on?"""
        r = self.reprojection_error(pitch_landmarks)
        if r["max_m"] < 0.5:
            return True, f"good (max {r['max_m']:.2f} m at {r['worst_landmark']})"
        if r["max_m"] < 1.5:
            return True, (
                f"usable (max {r['max_m']:.2f} m at {r['worst_landmark']}). Fine for "
                "heatmaps and formation; too coarse for offside or line height."
            )
        return False, (
            f"BAD (max {r['max_m']:.2f} m at {r['worst_landmark']}). Re-click that "
            "landmark, add points near the far touchline, or enable use_tps if "
            "this is a stitched panorama."
        )

    # -------------------------------------------------------------------- i/o

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "H": self.H.tolist(),
            "image_size": list(self.image_size),
            "named_points": {k: list(v) for k, v in self.named_points.items()},
            "use_tps": self.use_tps,
            "notes": self.notes,
        }, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path, pitch_landmarks: dict | None = None) -> "Calibration":
        d = json.loads(Path(path).read_text())
        cal = cls(
            H=np.array(d["H"]),
            image_size=tuple(d["image_size"]),
            named_points={k: tuple(v) for k, v in d["named_points"].items()},
            use_tps=d.get("use_tps", False),
            notes=d.get("notes", ""),
        )
        if cal.use_tps:
            if pitch_landmarks is None:
                raise ValueError("this calibration uses TPS; pass pitch_landmarks to rebuild it")
            names = sorted(cal.named_points)
            src = np.array([cal.named_points[n] for n in names])
            dst = np.array([pitch_landmarks[n] for n in names])
            resid = apply_homography(cal.H, src)
            cal._tps_fwd, cal._tps_inv = _TPS(resid, dst), _TPS(dst, resid)
        return cal

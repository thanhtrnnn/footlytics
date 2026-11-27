"""Remove the camera's drift steps without touching its motion.

The camera of a clip is a chain: a solved keyframe fixes the pose, a tracker carries it frame to
frame (`AnchoredCamera`), and the next accepted keyframe replaces the carried pose outright. The
carried pose drifts, so at every keyframe the chain jumps by the accumulated drift (median 0.22 m,
23% of keyframes > 0.5 m, tail in metres), and every player on the pitch moves by that jump in one
frame. The tracker's own re-anchoring to the first frame (a global SIFT match) does the same
between keyframes.

What is *not* wrong is the chain's frame-to-frame motion: it follows the picture, boxes and
camera move together. Smoothing the camera in time therefore has to be gentle (measured on a fast
pan with repeated video frames, a Gaussian of 4 frames alone put 0.7 m of error into frames that
were right), and the jump has to be treated as what it is, an accumulated error that arrives all
at once. Here each keyframe's jump is measured (the step height between straight-line fits of the
frames either side, which discounts the pan itself) and handed back over the interval it
accumulated in, a linear ramp from the previous keyframe. The camera is then continuous at the
keyframe and still passes through it. What is left, the tracker's frame-to-frame noise, gets a
light local-linear Gaussian (`sigma`, 3 frames) that leaves a steady pan unchanged.

The parameterisation is the pitch position of four fixed image points (the corners of a quad from
the far to the near field). They are metres, so one threshold means the same thing everywhere; a
linear blend of them is always a valid homography; and the far corners carry the depth
amplification. `homography_from_params` solves the four-point H back exactly.

Segments break at frames with no camera, at a jump larger than `break_jump_m` between consecutive
frames (a cut), and at caller-supplied breaks (shot changes). A keyframe step larger than `cap_m`
is left as it is: metres handed back over ten frames is a worse motion than the step, and it is
not drift. Repeated video frames (camera unchanged to a centimetre) are one observation and keep
their first frame's value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

#: Image points (fractions of width, height): far-left, far-right, near-left, near-right.
ANCHOR_FRACS = np.array([(0.10, 0.40), (0.90, 0.40), (0.10, 0.95), (0.90, 0.95)], np.float64)
#: A projected anchor farther than this from the pitch centre means the quad reaches the horizon.
_MAX_ABS_M = 400.0


def anchor_points(w: float, h: float) -> np.ndarray:
    return ANCHOR_FRACS * np.array([w, h], np.float64)


def params_from_homography(H: Optional[np.ndarray], w: float, h: float) -> Optional[np.ndarray]:
    """(8,) pitch metres of the four anchors under `H` (image -> pitch), or None if degenerate."""
    if H is None:
        return None
    H = np.asarray(H, np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        return None
    p = np.c_[anchor_points(w, h), np.ones(4)] @ H.T
    # an anchor on or behind the camera plane (horizon): the quad is not on the ground
    if (np.abs(p[:, 2]) <= 1e-12).any() or (np.sign(p[:, 2]) != np.sign(p[0, 2])).any():
        return None
    xy = p[:, :2] / p[:, 2:3]
    if not np.isfinite(xy).all() or np.abs(xy).max() > _MAX_ABS_M:
        return None
    return xy.ravel()


def homography_from_params(P: np.ndarray, w: float, h: float) -> Optional[np.ndarray]:
    """The exact image -> pitch homography through the four anchors at `P`; None if degenerate."""
    dst = np.asarray(P, np.float64).reshape(4, 2)
    if not np.isfinite(dst).all():
        return None
    try:
        H, _ = cv2.findHomography(anchor_points(w, h), dst, 0)   # 4 points: exact, float64
    except cv2.error:
        return None
    if H is None or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        return None
    H = H / H[2, 2]
    return H if abs(np.linalg.det(H)) > 1e-15 else None


def corner_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Largest of the four anchors' pitch displacements between parameter rows, metres."""
    d = (np.asarray(a) - np.asarray(b)).reshape(-1, 4, 2)
    return np.hypot(d[..., 0], d[..., 1]).max(axis=1)


def local_linear(P: np.ndarray, wt: np.ndarray, sigma: float, chunk: int = 8192) -> np.ndarray:
    """Weighted Gaussian-kernel local-linear fit of every column of `P` (n, k), at every row.

    A straight line is fitted in each window, so a steady pan comes back unchanged (no lag, no
    bias), and at the end of a segment the fit is one-sided without being pulled inward.
    """
    n = len(P)
    R = max(int(np.ceil(3.0 * sigma)), 1)
    d = np.arange(-R, R + 1, dtype=np.float64)
    ker = np.exp(-0.5 * (d / sigma) ** 2)
    Pp = np.zeros((n + 2 * R, P.shape[1]))
    Pp[R:R + n] = P
    wp = np.zeros(n + 2 * R)
    wp[R:R + n] = wt
    out = np.empty_like(P, dtype=np.float64)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        idx = np.arange(a, b)[:, None] + np.arange(2 * R + 1)[None, :]    # padded coordinates
        W = wp[idx] * ker
        S0, S1, S2 = W.sum(1), (W * d).sum(1), (W * d * d).sum(1)
        X = Pp[idx]                                             # (m, 2R+1, k)
        T0 = np.einsum("nm,nmk->nk", W, X)
        T1 = np.einsum("nm,nmk->nk", W * d, X)
        det = S0 * S2 - S1 * S1
        good = det > 1e-9 * np.maximum(S0 * S2, 1e-30)
        with np.errstate(divide="ignore", invalid="ignore"):
            lin = (S2[:, None] * T0 - S1[:, None] * T1) / det[:, None]
            flat = T0 / S0[:, None]
        res = np.where(good[:, None], lin, flat)
        out[a:b] = np.where((S0 > 1e-12)[:, None], res, P[a:b])        # no weight anywhere: keep
    return out


def _mag(e: np.ndarray) -> float:
    return float(corner_dist(e[None], np.zeros((1, 8)))[0])


def _line_at(f: np.ndarray, X: np.ndarray, at: float) -> np.ndarray:
    """Least-squares straight line through rows `X` at times `f`, evaluated at `at`."""
    fm = f.mean()
    d = f - fm
    den = float((d * d).sum())
    slope = (d[:, None] * (X - X.mean(0))).sum(0) / den if den > 0 else np.zeros(X.shape[1])
    return X.mean(0) + slope * (at - fm)


def step_height(f: np.ndarray, X: np.ndarray, j: int, m: int = 5) -> Optional[np.ndarray]:
    """The jump at row `j` of the series (times `f`, rows `X`), net of the motion either side.

    Straight lines through up to `m` rows before and after `j` (the row itself excluded: a
    keyframe's own fit is the noisy part) are evaluated at `f[j]`; their difference is the step.
    None when a side has fewer than 2 rows.
    """
    lo, hi = max(j - m, 0), min(j + m + 1, len(f))
    if j - lo < 2 or hi - (j + 1) < 2:
        return None
    return _line_at(f[j + 1:hi], X[j + 1:hi], f[j]) - _line_at(f[lo:j], X[lo:j], f[j])


@dataclass
class SmoothInfo:
    frames: int = 0                 # frames with a camera
    segments: int = 0
    events: int = 0                 # keyframe steps handed back
    breaks_jump: int = 0            # cuts found from the camera itself
    breaks_cap: int = 0             # steps too large to be drift
    max_step_m: float = 0.0         # largest step distributed
    max_move_m: float = 0.0         # largest displacement any frame's camera received


def correct_drift(P: np.ndarray, valid: np.ndarray, key: Optional[np.ndarray] = None, *,
                  breaks: Optional[Sequence[bool]] = None, frames: Optional[np.ndarray] = None,
                  break_jump_m: float = 15.0, cap_m: float = 4.0,
                  dup_tol_m: float = 0.01, max_ramp: int = 60, m: int = 5,
                  sigma: float = 3.0
                  ) -> tuple[np.ndarray, SmoothInfo]:
    """Hand each drift step back over the interval it accumulated in. See the module docstring.

    `P` (T, 8) camera parameters, `valid` (T,) which rows have one, `key` (T,) which frames
    are (accepted) keyframes: the step at each is measured and handed back. `breaks[t]` starts a
    new segment at t. `frames[t]` is the video frame of row t (default t); a gap breaks.
    `max_ramp` frames is the longest interval a step is spread over. `sigma` (frames; 0 = off) then
    smooths what is left, the tracker's frame-to-frame noise, with a local-linear Gaussian that
    leaves a steady pan unchanged.
    """
    T = len(P)
    out = np.array(P, np.float64, copy=True)
    info = SmoothInfo(frames=int(np.sum(valid)))
    if T == 0 or not np.any(valid):
        return out, info
    fr = np.arange(T) if frames is None else np.asarray(frames)
    key = np.zeros(T, bool) if key is None else np.asarray(key, bool)
    brk = np.zeros(T, bool) if breaks is None else np.asarray(breaks, bool)
    vi = np.flatnonzero(valid)
    cut = np.zeros(len(vi), bool)
    cut[0] = True
    cut[1:] |= np.diff(fr[vi]) != 1
    cut |= brk[vi]
    if len(vi) > 1:
        jc = corner_dist(P[vi[1:]], P[vi[:-1]]) > break_jump_m
        info.breaks_jump = int((jc & ~cut[1:]).sum())
        cut[1:] |= jc

    def run_segment(rows: np.ndarray) -> None:
        info.segments += 1
        n = len(rows)
        X = P[rows]
        dup = np.zeros(n, bool)
        dup[1:] = corner_dist(X[1:], X[:-1]) < dup_tol_m
        first = np.maximum.accumulate(np.where(dup, 0, np.arange(n)))
        u = np.flatnonzero(~dup)                       # one row per distinct picture
        if len(u) < 2 * 2 + 1:
            return
        f, Xu, ku = fr[rows][u].astype(float), X[u], key[rows][u]
        steps: dict[int, np.ndarray] = {}
        for j in np.flatnonzero(ku):
            e = step_height(f, Xu, int(j), m)
            if e is not None:
                steps[int(j)] = e
        events = sorted(steps)
        corr = np.zeros((len(u), 8))
        prev = 0
        for j in events:
            e = steps[j]
            mag = _mag(e)
            if mag > cap_m:                            # not drift: spread nothing across it
                info.breaks_cap += 1
                prev = j
                continue
            s = max(prev, int(np.searchsorted(f, f[j] - max_ramp)))
            if f[j] > f[s]:
                w = (f[s + 1:j] - f[s]) / (f[j] - f[s])
                corr[s + 1:j] += w[:, None] * e[None, :]
            info.events += 1
            info.max_step_m = max(info.max_step_m, mag)
            prev = j
        info.max_move_m = max(info.max_move_m, float(corner_dist(corr, np.zeros_like(corr)).max()))
        full = np.empty_like(X)
        full[u] = Xu + corr
        full = full[first]
        if sigma > 0:                                   # then the noise the steps do not explain
            full = local_linear(full, (~dup).astype(float), sigma)[first]
        out[rows] = full

    starts = np.flatnonzero(cut)
    for s, e in zip(starts, np.r_[starts[1:], len(vi)]):
        run_segment(vi[int(s):int(e)])
    return out, info


def smooth_homographies(Hs: Sequence[Optional[np.ndarray]], w: float, h: float, *,
                        key: Optional[Sequence[bool]] = None,
                        breaks: Optional[Sequence[bool]] = None,
                        **kw) -> tuple[list[Optional[np.ndarray]], SmoothInfo]:
    """`correct_drift` on a per-frame list of image -> pitch homographies (None = no camera).

    A frame whose corrected camera cannot be solved keeps its own homography; frames that were
    not moved are returned as the very same arrays. `key[t]`: frame t is an accepted keyframe.
    """
    T = len(Hs)
    P = np.zeros((T, 8))
    valid = np.zeros(T, bool)
    for i, H in enumerate(Hs):
        p = params_from_homography(H, w, h)
        if p is not None:
            P[i], valid[i] = p, True
    Pc, info = correct_drift(P, valid, None if key is None else np.asarray(key, bool),
                             breaks=breaks, **kw)
    out: list[Optional[np.ndarray]] = []
    for i, H in enumerate(Hs):
        if H is None or not valid[i] or np.array_equal(Pc[i], P[i]):
            out.append(H)
            continue
        Hn = homography_from_params(Pc[i], w, h)
        out.append(H if Hn is None else Hn)
    return out, info

"""Finding the painted lines on the pitch.

Status: this does NOT work on hard footage, and the honest value of the module
is that it now says so.

The premise was that a fixed camera makes calibration easy enough for classical
vision -- no need for SoccerMaster's learned `LinesDetection` head and its 1.4 GB
backbone, which exist to handle broadcast cameras that pan and zoom. On a bright
match with clean markings that premise may hold. On the SoccerTrack v2 night
match it fails badly, and it failed in the most misleading way possible: tuned on
one frame it recovered the pitch width to within 0.0 m, and on four other frames
from the same clip it returned 87.5, 63.0, 46.0 and 47.5 m -- a 41 m spread --
while reporting high confidence every time. That first result was selection on a
single example, not a working method.

Agreement across frames does not rescue it either. The false detections are
static structures -- a perimeter fence, floodlight masts, a neighbouring pitch's
markings -- so they reproduce perfectly from frame to frame. On this clip the
consensus settled on a 47 m width with 5/8 agreement and a 119 m length with 8/8.
Stable false positives are indistinguishable from true ones by agreement alone.

Four approaches were tried and all four failed on this footage: raw peak height,
robust quantile statistics, peak prominence with a narrow-width constraint, and
multi-frame consensus. The third looked like a success because it was developed
and evaluated on the same single frame.

What does work is an INDEPENDENT check. Players cannot occupy ground outside the
pitch, so their positional extent is a hard lower bound on its size, derived from
completely different evidence than the pixels of a line. `cross_check` uses it to
reject impossible answers, and here it rejects them.

Use this module as a convenience that occasionally saves a manual measurement,
never as a dependency. For a fixed camera, measuring the pitch once per venue --
physically, or by clicking the corners with `geometry.annotate` -- takes minutes
and is reliable.

The useful trick here is not detecting lines in the abstract. It is scoring a
*known* line: we already have a homography, so we can project a candidate
touchline through the full fisheye chain and ask how much painted line lies
under that exact curve. Sweeping the candidate turns calibration checking into a
1-D search with a clear peak, and it handles lens distortion for free because
the projection does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


def line_mask(frame: np.ndarray, kernel: int = 11, blur: int = 3) -> np.ndarray:
    """Highlight thin bright structures: the painted lines.

    A morphological top-hat rather than a brightness threshold, because a night
    match under floodlights has a bright middle and dark corners -- any global
    threshold either loses the corners or floods the centre. Top-hat responds to
    *local* contrast, so a faint line in a dark corner scores like a bright one
    under a lamp.

    `kernel` should be comfortably wider than a painted line at its widest in the
    image; anything thinner than the kernel survives, anything broader is removed
    as background.
    """
    import cv2

    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur:
        g = cv2.GaussianBlur(g, (blur | 1, blur | 1), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel | 1, kernel | 1))
    top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
    return cv2.normalize(top, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def score_curve(mask: np.ndarray, pts: np.ndarray, half_width: int = 2) -> float:
    """Mean mask response along a projected curve, ignoring off-image points.

    Sampled in a small band rather than on the exact pixel, because a 1-pixel
    projection error would otherwise miss a line that is plainly there.
    """
    h, w = mask.shape[:2]
    pts = np.asarray(pts, float).reshape(-1, 2)
    vals = []
    for dx in range(-half_width, half_width + 1):
        for dy in range(-half_width, half_width + 1):
            x = np.round(pts[:, 0] + dx).astype(int)
            y = np.round(pts[:, 1] + dy).astype(int)
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            if ok.sum():
                vals.append(mask[y[ok], x[ok]].astype(float))
    if not vals:
        return 0.0
    stacked = np.concatenate(vals)
    # A line is bright along most of its length, so the mean of the upper half
    # is a better statistic than the plain mean: it tolerates the occasional
    # player standing on the line without rewarding a curve that clips one
    # bright blob.
    return float(np.mean(np.sort(stacked)[len(stacked) // 2:]))


@dataclass
class SweepResult:
    values: np.ndarray
    scores: np.ndarray
    best: Optional[float]
    prominence: float
    width_m: float
    contrast: float          # prominence of the chosen peak over the runner-up

    @property
    def confident(self) -> bool:
        return self.best is not None and self.prominence >= 3.0

    def describe(self, unit: str = "m") -> str:
        if self.best is None:
            return "no sharp line found -- do not trust this"
        v = ("clear" if self.confident else "WEAK -- verify by eye")
        return (f"{self.best:.2f} {unit} (prominence {self.prominence:.1f}, "
                f"peak width {self.width_m:.1f} {unit}) -- {v}")


def find_line_peaks(values: np.ndarray, scores: np.ndarray,
                    max_width_m: float = 6.0, min_prominence: float = 2.0):
    """Locate sharp, narrow peaks in a sweep -- the signature of a painted line.

    Peak HEIGHT is the wrong criterion and choosing it produced confidently wrong
    answers on real night footage: floodlights, fences, spectators and worn grass
    all out-score the actual touchline, so the raw maximum landed 20 m off the
    pitch while reporting high confidence.

    Peak SHAPE separates them. A painted line is a few centimetres wide, so it
    produces a narrow spike; clutter produces a broad plateau. Ranking by
    prominence among peaks narrower than `max_width_m` put the two touchlines
    first and second on exactly the footage where height ranked them nowhere.
    """
    from scipy.signal import find_peaks

    step = float(values[1] - values[0]) if len(values) > 1 else 1.0
    idx, props = find_peaks(scores, prominence=min_prominence,
                            width=(None, max_width_m / step))
    if not len(idx):
        return []
    order = np.argsort(props["prominences"])[::-1]
    return [(float(values[idx[i]]), float(props["prominences"][i]),
             float(props["widths"][i] * step)) for i in order]


def sweep_line(
    mask: np.ndarray,
    project: Callable[[np.ndarray], np.ndarray],
    values: Sequence[float],
    along: tuple[float, float],
    axis: str = "y",
    n_samples: int = 200,
    half_width: int = 2,
) -> SweepResult:
    """Score a family of straight pitch lines and return the best-matching one.

    `project` maps pitch coordinates to image pixels. `axis="y"` sweeps lines of
    constant y (touchlines) drawn between the `along` limits in x, and vice versa.
    """
    lo, hi = along
    t = np.linspace(lo, hi, n_samples)
    scores = []
    for v in values:
        pts = (np.column_stack([t, np.full(n_samples, v)]) if axis == "y"
               else np.column_stack([np.full(n_samples, v), t]))
        scores.append(score_curve(mask, project(pts), half_width))
    scores = np.asarray(scores)
    values = np.asarray(values, float)
    peaks = find_line_peaks(values, scores)
    if not peaks:
        return SweepResult(values, scores, None, 0.0, 0.0, 0.0)
    best, prom, wid = peaks[0]
    runner = peaks[1][1] if len(peaks) > 1 else 1e-6
    return SweepResult(values, scores, best, prom, wid, prom / max(runner, 1e-6))


def cross_check(measured: dict, player_x: np.ndarray, player_y: np.ndarray,
                trim_pct: float = 0.5) -> dict:
    """Reject line measurements that the tracked players contradict.

    Independent evidence, which is the point: players and painted lines share no
    failure mode. A pitch cannot be smaller than the ground its players occupy,
    so any measured dimension below the observed extent is impossible regardless
    of how convincing the line looked.
    """
    span_x = float(np.percentile(player_x, 100 - trim_pct) - np.percentile(player_x, trim_pct))
    span_y = float(np.percentile(player_y, 100 - trim_pct) - np.percentile(player_y, trim_pct))
    problems = []
    for name, meas, span in (("length", measured.get("length"), span_x),
                             ("width", measured.get("width"), span_y)):
        if meas is None:
            continue
        if meas < span:
            problems.append(
                f"measured {name} {meas:.1f} m is smaller than the {span:.1f} m "
                f"the players actually cover -- impossible")
        elif meas > span / 0.75:
            problems.append(
                f"measured {name} {meas:.1f} m implies players use only "
                f"{span / meas:.0%} of it, which is implausibly little")
    return {
        "player_span_x": span_x, "player_span_y": span_y,
        "problems": problems,
        "ok": not problems and measured.get("confident", False),
    }


def measure_pitch_multiframe(
    frames: Sequence[np.ndarray],
    project: Callable[[np.ndarray], np.ndarray],
    length_guess: float = 105.0,
    width_guess: float = 68.0,
    search: float = 25.0,
    tolerance: float = 2.0,
    min_agree: int = 3,
    **kw,
) -> dict:
    """Measure a pitch from several frames, agreeing or admitting failure.

    Single-frame measurement is not trustworthy: on real footage it produced
    answers spanning 41 m across five frames of one clip, each labelled
    confident. Painted lines do not move, so genuine detections agree across
    frames and spurious ones do not. Agreement is therefore the only usable
    confidence signal, and disagreement is a result, not an error -- it means
    "measure this pitch by hand".
    """
    widths, lengths, per_frame = [], [], []
    for f in frames:
        r = measure_pitch(f, project, length_guess, width_guess, search, **kw)
        per_frame.append(r)
        if r["width"] is not None:
            widths.append(r["width"])
        if r["length"] is not None:
            lengths.append(r["length"])

    def consensus(vals):
        if len(vals) < min_agree:
            return None, 0, 0.0
        v = np.asarray(vals)
        # Largest cluster of mutually-agreeing measurements.
        best, count = None, 0
        for c in v:
            n = int((np.abs(v - c) <= tolerance).sum())
            if n > count:
                best, count = float(np.median(v[np.abs(v - c) <= tolerance])), n
        return best, count, float(np.ptp(v))

    w, wn, wspread = consensus(widths)
    l, ln, lspread = consensus(lengths)
    ok_w = w is not None and wn >= min_agree
    ok_l = l is not None and ln >= min_agree

    return {
        "width": w if ok_w else None,
        "length": l if ok_l else None,
        "width_agreement": f"{wn}/{len(frames)}",
        "length_agreement": f"{ln}/{len(frames)}",
        "width_spread_m": wspread,
        "length_spread_m": lspread,
        "confident": ok_w and ok_l,
        "per_frame": per_frame,
        "advice": ("measurement agreed across frames" if (ok_w and ok_l) else
                   "NO agreement across frames -- line detection failed on this "
                   "footage. Measure the pitch physically, or click the corners "
                   "once with geometry.annotate; both are reliable and take "
                   "minutes, and for a fixed camera it is a one-off per venue."),
    }


def measure_pitch(
    frame: np.ndarray,
    project: Callable[[np.ndarray], np.ndarray],
    length_guess: float = 105.0,
    width_guess: float = 68.0,
    search: float = 25.0,
    step: float = 0.5,
    kernel: int = 11,
) -> dict:
    """Measure a pitch from one frame, given an approximate calibration.

    Written because assuming the IFAB default silently corrupted real data: a
    ground that was actually ~76 m wide, read as 68 m, put 2.14% of tracked
    positions outside the touchline. Pitch dimensions vary a lot outside elite
    stadiums, and V.League grounds are no exception -- so measure, do not assume.

    `project` takes pitch coordinates in a CORNER-origin frame (0..length,
    0..width) and returns image pixels.
    """
    mask = line_mask(frame, kernel=kernel)
    inset = 0.15                     # ignore the ends, where lines converge

    def opposing(values, along, axis, lo_sep, hi_sep):
        """Pick the best PAIR of parallel lines a plausible distance apart.

        Sweeping each side independently invites both to land on the same
        strong feature, or on clutter. Requiring a plausible separation uses the
        one thing we know for free -- a pitch has two touchlines, and they are
        between `lo_sep` and `hi_sep` apart.
        """
        t = np.linspace(*along, n_samples := 200)
        scores = []
        for v in values:
            pts = (np.column_stack([t, np.full(n_samples, v)]) if axis == "y"
                   else np.column_stack([np.full(n_samples, v), t]))
            scores.append(score_curve(mask, project(pts)))
        scores = np.asarray(scores)
        peaks = find_line_peaks(values, scores)
        best_pair, best_score = None, -np.inf
        for i, (a, pa, wa) in enumerate(peaks):
            for b, pb, wb in peaks[i + 1:]:
                lo, hi = (a, b) if a < b else (b, a)
                if lo_sep <= hi - lo <= hi_sep and pa + pb > best_score:
                    best_score, best_pair = pa + pb, ((lo, pa, wa), (hi, pb, wb))
        return scores, peaks, best_pair

    span = length_guess + 2 * search
    y_vals = np.arange(-search, width_guess + search + step, step)
    x_vals = np.arange(-search, length_guess + search + step, step)

    ys, y_peaks, y_pair = opposing(
        y_vals, (length_guess * inset, length_guess * (1 - inset)), "y", 45.0, 95.0)
    xs, x_peaks, x_pair = opposing(
        x_vals, (width_guess * inset, width_guess * (1 - inset)), "x", 85.0, 125.0)

    def as_result(vals, scores, entry):
        if entry is None:
            return SweepResult(vals, scores, None, 0.0, 0.0, 0.0)
        v, p, w = entry
        return SweepResult(vals, scores, v, p, w, p)

    far = as_result(y_vals, ys, y_pair[0] if y_pair else None)
    near = as_result(y_vals, ys, y_pair[1] if y_pair else None)
    left = as_result(x_vals, xs, x_pair[0] if x_pair else None)
    right = as_result(x_vals, xs, x_pair[1] if x_pair else None)

    return {
        "mask": mask,
        "far_touchline": far, "near_touchline": near,
        "left_goal_line": left, "right_goal_line": right,
        "width": (near.best - far.best) if y_pair else None,
        "length": (right.best - left.best) if x_pair else None,
        "confident": all(r.confident for r in (near, far, left, right)),
    }

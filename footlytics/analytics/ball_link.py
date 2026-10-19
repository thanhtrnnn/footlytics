"""Pick the ball by following a trajectory, not by choosing frame by frame.

`BallSelector` is greedy: it keeps the candidate nearest to where the ball was last
seen and, after a second without a sighting, grabs the most confident one. That has
two failure modes. A wrong lock (a boot, a head, a ball in an advert) persists for as
long as it stays reachable, and every re-acquisition is decided by one frame's
confidence, which for a 10 px ball is a poor judge. It also forces the detector to a
high confidence floor, because any false positive that out-scores the true ball can
capture the track.

Here the whole clip is decided at once. Each candidate scores its confidence; going
from one picked candidate to the next costs a little when the ball could have covered
the distance, and a fixed `break_cost` when it could not. The best-scoring path (at
most one candidate per frame, frames may be skipped) is found by dynamic programming.
A faint candidate that continues a track is then worth taking, and a bright one that
would force the track to jump and jump back is not.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def link_ball_candidates(
    frames: Sequence[int],
    xy: Sequence[np.ndarray],
    conf: Sequence[np.ndarray],
    fps: float,
    *,
    penalty: Optional[Sequence[np.ndarray]] = None,
    max_speed: float = 35.0,
    slack_m: float = 1.0,
    max_gap_s: float = 1.0,
    break_cost: float = 0.25,
    gap_cost: float = 0.005,
    speed_cost: float = 0.3,
) -> list[Optional[int]]:
    """For each entry of `frames` (ascending), the index of the candidate on the best path or None.

    `xy[k]` (n_k, 2) pitch metres and `conf[k]` (n_k,) are the candidates of frame
    `frames[k]`. `penalty[k]` (n_k,) in [0, 1] scales a candidate's score down (a ball that
    lies still); 1.0 means a candidate that cannot score at all, which is how decoys are
    excluded.

    Score of a path: the sum of its candidates' `conf * (1 - penalty)`, minus for each step to
    the next picked candidate `gap_cost` per skipped frame plus `speed_cost * (speed/max_speed)**2`
    when the ball could have got there (`max_speed * dt + slack_m`, within `max_gap_s`), else
    `break_cost`. So `break_cost` is what a lone detection must beat to be taken; a candidate
    that continues the track is taken at any confidence.
    """
    n = len(frames)
    picks: list[Optional[int]] = [None] * n
    if not n:
        return picks
    max_gap = max(int(round(max_gap_s * fps)), 1)
    unary = []
    for k in range(n):
        c = np.asarray(conf[k], float).ravel()
        if penalty is not None:
            c = c * (1.0 - np.clip(np.asarray(penalty[k], float).ravel(), 0.0, 1.0))
        unary.append(c)

    best: list[np.ndarray] = [np.empty(0)] * n
    pred: list[list[Optional[tuple[int, int]]]] = [[] for _ in range(n)]
    pref_val, pref_ref = -np.inf, None        # best path score ending at any earlier node
    for k in range(n):
        m = len(unary[k])
        if not m:
            continue
        pts = np.asarray(xy[k], float).reshape(-1, 2)
        val = np.zeros(m)
        prv: list[Optional[tuple[int, int]]] = [None] * m
        # a new path (free), or break from the best earlier node
        base = 0.0
        base_ref = None
        if pref_ref is not None and pref_val - break_cost > base:
            base, base_ref = pref_val - break_cost, pref_ref
        val[:] = base
        for i in range(m):
            prv[i] = base_ref
        # continue from a reachable earlier node
        j = k - 1
        while j >= 0 and frames[k] - frames[j] <= max_gap:
            if len(best[j]):
                dt = frames[k] - frames[j]
                d = np.linalg.norm(pts[:, None] - np.asarray(xy[j], float).reshape(-1, 2)[None], axis=-1)
                ok = d <= max_speed * dt / fps + slack_m
                step = gap_cost * (dt - 1) + speed_cost * ((d * fps / dt) / max_speed) ** 2
                cand = np.where(ok, best[j][None, :] - step, -np.inf)
                a = cand.argmax(axis=1)
                v = cand[np.arange(m), a]
                better = v > val
                for i in np.flatnonzero(better):
                    val[i], prv[i] = v[i], (j, int(a[i]))
            j -= 1
        val = np.where(unary[k] > 0, val + unary[k], -np.inf)   # no score, no node
        best[k], pred[k] = val, prv
        i = int(val.argmax())
        if np.isfinite(val[i]) and val[i] > pref_val:
            pref_val, pref_ref = float(val[i]), (k, i)

    if pref_ref is None:
        return picks
    node: Optional[tuple[int, int]] = pref_ref
    while node is not None:
        k, i = node
        picks[k] = i
        node = pred[k][i]
    return picks

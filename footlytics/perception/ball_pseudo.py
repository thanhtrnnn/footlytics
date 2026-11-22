"""Pseudo-labels for the ball from a detector's candidates plus temporal consistency.

No V.League ball labels exist, so the RT-DETR ball detector (trained on SoccerNet) is run over
unlabelled V.League footage and only the candidates that form a *ball track* are kept: a chain of
detections over consecutive frames that is confident, moves smoothly, keeps its size and stays on
the pitch. A single high-confidence detection is not enough (a head, a sock or a pitch marking can
score high for one frame), a smooth chain of them almost never is.

Pure numpy, no torch: candidates are `[x, y, w, h, score, on_pitch]` rows per frame (x, y the
top-left corner, in pixels), `on_pitch` 1.0 when the centre lies on the grass region.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackConfig:
    seed_score: float = 0.50      # a track starts from a candidate at least this confident
    extend_score: float = 0.20    # and is extended through weaker ones
    min_len: int = 15             # detections in a track (short chains are knee tape and boots)
    min_mean_score: float = 0.65
    min_first_share: float = 0.6  # share of its detections that are >= seed_score - 0.1
    max_gap: int = 2              # frames a track may skip (the box is interpolated there)
    max_speed: float = 90.0       # px per frame, hard gate on a link
    gate_floor: float = 18.0      # px, link gate around the prediction at rest
    max_size_ratio: float = 1.5   # w and h may change by this factor between linked frames
    max_accel_p90: float = 40.0   # px per frame^2, 90th percentile of the second difference
    min_path: float = 30.0        # px travelled; a still blob is not trusted ...
    still_score: float = 0.85     # ... unless it is this confident
    min_on_pitch: float = 0.8     # share of detections whose centre is on the pitch
    min_size: float = 5.0
    max_size: float = 60.0


@dataclass
class Track:
    frames: list[int]
    boxes: np.ndarray            # [n, 4] x, y, w, h
    scores: np.ndarray           # [n]
    interp: np.ndarray           # [n] bool, True where the box was interpolated over a gap
    id: int = -1
    stats: dict = field(default_factory=dict)


def _centre(b):
    return np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])


def _grow(dets, seed, used, cfg: TrackConfig, step: int):
    """Extend a track from `seed` = (frame, idx) forward (step=+1) or backward (-1)."""
    chain = []
    f, i = seed
    last = [(f, _centre(dets[f][i]), dets[f][i])]
    miss = 0
    t = f
    while True:
        t += step
        if t < 0 or t >= len(dets):
            break
        # velocity per frame from the last two points of the chain (in growth direction)
        if len(last) >= 2:
            (f1, c1, _), (f0, c0, _) = last[-1], last[-2]
            v = (c1 - c0) / (f1 - f0)          # signed, so it is right when growing backwards too
        else:
            v = np.zeros(2)
        fl, cl, bl = last[-1]
        dt = abs(t - fl)
        pred = cl + v * (t - fl)
        gate = max(cfg.gate_floor, 0.6 * np.linalg.norm(v) * dt + cfg.gate_floor)
        gate = min(gate, cfg.max_speed * dt)
        best, best_val = None, -1.0
        for j, row in enumerate(dets[t]):
            if (t, j) in used or row[4] < cfg.extend_score:
                continue
            d = np.linalg.norm(_centre(row) - pred)
            if d > gate:
                continue
            r = max(row[2] / bl[2], bl[2] / row[2], row[3] / bl[3], bl[3] / row[3])
            if r > cfg.max_size_ratio:
                continue
            val = row[4] - 0.5 * d / gate
            if val > best_val:
                best, best_val = j, val
        if best is None:
            miss += 1
            if miss > cfg.max_gap:
                break
            continue
        miss = 0
        row = dets[t][best]
        last.append((t, _centre(row), row))
        chain.append((t, best))
    return chain


def _interpolate(chain, dets):
    """chain = [(frame, idx)] sorted -> frames, boxes, scores, interp with gaps filled."""
    frames, boxes, scores, interp = [], [], [], []
    for k, (f, i) in enumerate(chain):
        if k:
            pf = chain[k - 1][0]
            prev = boxes[-1]
            for g in range(pf + 1, f):
                a = (g - pf) / (f - pf)
                frames.append(g)
                boxes.append((1 - a) * prev + a * dets[f][i][:4])
                scores.append(np.nan)
                interp.append(True)
        frames.append(f)
        boxes.append(np.asarray(dets[f][i][:4], np.float64))
        scores.append(float(dets[f][i][4]))
        interp.append(False)
    return frames, np.array(boxes), np.array(scores), np.array(interp)


def build_tracks(dets: list[np.ndarray], cfg: TrackConfig | None = None) -> list[Track]:
    """Greedy hysteresis tracking over `dets[frame]` = [n, 6] candidate rows.

    Seeds are taken in descending confidence; each track is grown both ways with a
    constant-velocity prediction and a gate, then kept only if it passes the track filters.
    Candidates of a rejected track stay free for other seeds.
    """
    cfg = cfg or TrackConfig()
    dets = [np.asarray(d, np.float64).reshape(-1, 6) for d in dets]
    seeds = sorted(((row[4], f, i) for f, d in enumerate(dets) for i, row in enumerate(d)
                    if row[4] >= cfg.seed_score), reverse=True)
    used: set = set()
    tracks: list[Track] = []
    for _, f, i in seeds:
        if (f, i) in used:
            continue
        fwd = _grow(dets, (f, i), used | {(f, i)}, cfg, +1)
        bwd = _grow(dets, (f, i), used | {(f, i)} | set(fwd), cfg, -1)
        chain = sorted(bwd + [(f, i)] + fwd)
        tr = _finish(chain, dets, cfg)
        if tr is None:
            continue
        used.update(chain)
        tr.id = len(tracks)
        tracks.append(tr)
    return tracks


def _finish(chain, dets, cfg: TrackConfig) -> Track | None:
    if len(chain) < cfg.min_len:
        return None
    frames, boxes, scores, interp = _interpolate(chain, dets)
    real = ~interp
    s = scores[real]
    rows = np.array([dets[f][i] for f, i in chain])
    c = boxes[:, :2] + boxes[:, 2:] / 2
    fr = np.array(frames)
    path = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
    # second difference per frame^2 over consecutive frames of the (interpolated) chain
    acc = np.linalg.norm(c[2:] - 2 * c[1:-1] + c[:-2], axis=1) if len(c) > 2 else np.zeros(1)
    size = float(np.median(boxes[:, 2:]))
    stats = {"n": int(len(chain)), "mean_score": float(s.mean()), "min_score": float(s.min()),
             "path_px": path, "accel_p90": float(np.percentile(acc, 90)),
             "on_pitch": float(rows[:, 5].mean()), "size_px": size,
             "first_share": float((s >= cfg.seed_score - 0.1).mean()),
             "span": int(fr[-1] - fr[0] + 1)}
    ok = (stats["mean_score"] >= cfg.min_mean_score and stats["first_share"] >= cfg.min_first_share
          and stats["accel_p90"] <= cfg.max_accel_p90 and stats["on_pitch"] >= cfg.min_on_pitch
          and cfg.min_size <= size <= cfg.max_size
          and (path >= cfg.min_path or stats["mean_score"] >= cfg.still_score))
    if not ok:
        return None
    return Track(frames=frames, boxes=boxes, scores=scores, interp=interp, stats=stats)


def label_rows(tracks: list[Track], dets: list[np.ndarray], conflict_score: float = 0.5,
               conflict_px: float = 40.0, min_frame_score: float = 0.30, bridge: int = 3,
               bridge_score: float = 0.5) -> dict[int, dict]:
    """frame -> {"box", "score", "track", "interp", "rivals"} for every frame of a track.

    `rivals` counts other candidates >= `conflict_score` farther than `conflict_px` from the track's
    box in the same frame: the track is probably the match ball, but a spare ball or a false
    positive is also in view, so callers may want to treat those frames more carefully.

    A frame is labelled only if its detection scores >= `min_frame_score`, or it is a weak frame
    or an interpolated one that sits between detections >= `bridge_score` at most `bridge` frames
    away on both sides (the ball is occluded or blurred for a moment inside a confident track; a
    track's low-score ends tend to drift onto a player).
    """
    out = {}
    for tr in tracks:
        strong = np.array([(not i) and s >= bridge_score for i, s in zip(tr.interp, np.nan_to_num(tr.scores))])
        for k, f in enumerate(tr.frames):
            if tr.interp[k] or tr.scores[k] < min_frame_score:
                if not (strong[max(0, k - bridge):k].any() and strong[k + 1:k + 1 + bridge].any()):
                    continue
            b = tr.boxes[k]
            cen = b[:2] + b[2:] / 2
            d = np.asarray(dets[f]).reshape(-1, 6)
            riv = 0
            for row in d:
                if row[4] >= conflict_score and np.linalg.norm(_centre(row) - cen) > conflict_px:
                    riv += 1
            out[f] = {"box": [float(v) for v in b], "score": None if tr.interp[k] else float(tr.scores[k]),
                      "track": tr.id, "interp": bool(tr.interp[k]), "rivals": riv}
    return out

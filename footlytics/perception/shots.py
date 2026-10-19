"""Which frames are live, wide, main-camera football: the only ones the tactical layer may read.

A broadcast cuts away constantly: bench and player close-ups, crowd shots, full-screen graphics and
replays. Radar positions from those frames are either impossible (no pitch in view) or from the
past (a replay), and the sequence detectors happily turn them into a "low block" or a "high press".
So each frame gets cheap signals from a 320 px thumbnail (CPU only, ~1 ms), and
`live_wide_segments` turns the signal stream into a `live` mask and a shot id.

Signals, per frame:
  grass     share of the frame that is pitch-green. Close-ups on a crowd, bench or ad board lose it.
  blob      height of the tallest non-grass object that is not a band across the frame. On the
            high side camera a player is under 0.1; a keeper close-up in front of grass is 0.5+,
            and grass share alone calls that one "wide".
  person_h  optional: the median detected person-box height as a share of the frame, which the
            pipeline already has for free. It overrules the picture when it says "close".
  stinger   the trophy-diamond transition every replay opens and closes with: a gold centre
            inside a red ring. Gold in the centre third only -- the score bug (top-left) and the
            FPT logo (top-right) are orange, static, and sit outside it. They are also masked out
            of grass and blob, so they are never evidence of anything.

A replay is the span between two stingers (the first opens it, the next closes it), because a
replay often looks wide: a lower angle, but a green pitch and small players. Judged frame by
frame it would pass as live. A stinger with no partner within `replay_max_s` is just a transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

THUMB = (320, 180)


@dataclass
class ShotConfig:
    grass_min: float = 0.55         # wide frames are 0.7-0.9 grass; crowd, bench and board shots are < 0.4
    blob_max: float = 0.25          # tallest foreground object, as a share of frame height
    person_h_max: float = 0.22      # median person box, same units
    stinger_gold: float = 0.08      # gold share of the centre, together with some red
    stinger_red: float = 0.04
    stinger_full_gold: float = 0.30  # the diamond at full screen is nearly all gold
    stinger_min_s: float = 0.5      # a stinger lasts ~1.3 s; the FPT logo in a close-up flickers for ~0.6 s
    stinger_gap_s: float = 0.4      # the diamond flashes; bridge gaps up to this
    replay_max_s: float = 45.0      # replays seen last 6-13 s
    bridge_s: float = 0.5           # a non-wide gap this short inside a wide shot (no cut) is bridged
    min_wide_s: float = 2.0         # a "wide" run shorter than this is a camera swing, not play
    cut_thr: float = 0.35           # half L1 distance between colour histograms; same-camera frames are < 0.05


@dataclass
class FrameSignals:
    grass: float
    blob: float
    stinger: bool
    hist: np.ndarray = field(repr=False)
    person_h: float = float("nan")

    def is_wide(self, cfg: ShotConfig) -> bool:
        if not np.isnan(self.person_h) and self.person_h > cfg.person_h_max:
            return False
        return self.grass >= cfg.grass_min and self.blob <= cfg.blob_max and not self.stinger


def frame_signals(frame_bgr: np.ndarray, person_h: float = float("nan")) -> FrameSignals:
    import cv2

    small = cv2.resize(frame_bgr, THUMB, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    valid = np.ones((h, w), bool)
    valid[:int(.15 * h), :int(.22 * w)] = False           # score bug
    valid[:int(.15 * h), int(.80 * w):] = False           # channel logo

    grass = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255)) > 0
    g = float(grass[valid].mean())
    closed = cv2.morphologyEx(grass.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    # Pitch lines and distant players close over; a near person or a board does not.
    # Rows from 0.35 down: above that are the stands and boards, which a person in front of would
    # merge into one band and hide.
    lower = (~closed) & valid
    lower[:int(.35 * h)] = False
    n, _, stats, _ = cv2.connectedComponentsWithStats(lower.astype(np.uint8), connectivity=8)
    tall = [stats[i, cv2.CC_STAT_HEIGHT] / h for i in range(1, n)
            if stats[i, cv2.CC_STAT_WIDTH] < .5 * w and stats[i, cv2.CC_STAT_AREA] > .002 * h * w]
    blob = max(tall, default=0.0)

    c = hsv[h // 4:3 * h // 4, w // 3:2 * w // 3].astype(np.int16)
    hue, sat, val = c[..., 0], c[..., 1], c[..., 2]
    red = (((hue <= 7) | (hue >= 172)) & (sat > 170) & (val > 110)).mean()
    gold = ((hue >= 14) & (hue <= 24) & (sat > 90) & (sat < 200) & (val > 140)).mean()
    stinger = bool((gold > ShotConfig.stinger_gold and red > ShotConfig.stinger_red)
                   or gold > ShotConfig.stinger_full_gold)

    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
    return FrameSignals(g, float(blob), stinger, hist / max(hist.sum(), 1.0), person_h)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end] (inclusive) of each run of True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def _fill_gaps(mask: np.ndarray, max_gap: int, blocked: np.ndarray | None = None) -> np.ndarray:
    """Set False gaps of at most `max_gap` frames between True runs, unless a `blocked` frame is inside."""
    out = mask.copy()
    runs = _runs(~mask)
    for s, e in runs:
        if s == 0 or e == len(mask) - 1 or e - s + 1 > max_gap:
            continue
        if blocked is None or not blocked[s:e + 2].any():
            out[s:e + 1] = True
    return out


def stinger_and_replay(signals: list[FrameSignals], fps: float,
                       cfg: ShotConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(stinger, replay) masks. A replay runs from one stinger to the next, stingers included."""
    cfg = cfg or ShotConfig()
    n = len(signals)
    stinger = _fill_gaps(np.array([s.stinger for s in signals]), round(cfg.stinger_gap_s * fps))
    runs = [(s, e) for s, e in _runs(stinger) if e - s + 1 >= round(cfg.stinger_min_s * fps)]
    stinger = np.zeros(n, bool)
    for s, e in runs:
        stinger[s:e + 1] = True
    replay = np.zeros(n, bool)
    i = 0
    while i < len(runs) - 1:
        if runs[i + 1][0] - runs[i][1] <= cfg.replay_max_s * fps:
            replay[runs[i][0]:runs[i + 1][1] + 1] = True
            i += 2
        else:
            i += 1
    return stinger, replay


def live_wide_segments(signals: list[FrameSignals], fps: float,
                       cfg: ShotConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(live, shot): per frame, whether it is live wide play and which shot it belongs to.

    A shot ends at a cut (colour-histogram jump) and at every change of `live`, so an interval of
    consecutive frames with the same shot id is something one camera saw. Analytics must not
    join frames across ids.
    """
    cfg = cfg or ShotConfig()
    n = len(signals)
    if n == 0:
        return np.zeros(0, bool), np.zeros(0, np.int32)

    stinger, replay = stinger_and_replay(signals, fps, cfg)

    hists = np.stack([s.hist for s in signals])
    cut = np.r_[False, 0.5 * np.abs(np.diff(hists, axis=0)).sum(axis=1) > cfg.cut_thr]

    wide = np.array([s.is_wide(cfg) for s in signals]) & ~stinger & ~replay
    wide = _fill_gaps(wide, round(cfg.bridge_s * fps), blocked=cut) & ~stinger & ~replay
    live = wide.copy()
    for s, e in _runs(wide):
        if e - s + 1 < cfg.min_wide_s * fps:
            live[s:e + 1] = False

    boundary = cut | np.r_[False, live[1:] != live[:-1]]
    return live, np.cumsum(boundary).astype(np.int32)

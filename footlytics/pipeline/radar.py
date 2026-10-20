"""Video in, MatchState out.

Single pass over the frames, in this order, because each step needs the last:

    detect -> map feet to pitch -> kit descriptor -> track (motion + appearance)

Team assignment happens *after* the pass, not during it, because it is a
per-track decision and the vote is only trustworthy once the whole track exists.
Same for role: a goalkeeper is a goalkeeper for the whole clip, so we decide
once rather than 1,500 times.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from ..analytics.ball import interpolate_ball
from ..analytics.ball_link import link_ball_candidates
from ..analytics.kinematics import add_kinematics
from ..geometry.homography import Calibration, apply_homography
from ..geometry.pitch import Pitch
from ..perception.detect import Detector, DetectorConfig
from ..perception.teams import (split_officials, TeamClassifier, kit_descriptor,
                                team_scores_by_track)
from ..perception.identity import (apply_to_state, assign_teams_with_roster,
                                   build_tracklets, goalkeeper_teams, goalkeeper_tracklets,
                                   merge_same_role, stitch_tracklets, touchline_people)
from ..perception.shots import (ShotConfig, frame_signals, live_wide_segments,
                                stinger_and_replay)
from ..perception.track import PitchTracker, TrackerConfig
from ..geometry.camera_motion import AnchoredCamera, MovingCalibration
from ..state.schema import MatchMeta, MatchState, Role, Team, TeamInfo


#: (source frame index, BGR frame, detections [N, >=6]) -> feet in pitch metres [N, 2], or
#: None when this frame cannot be related to the pitch: its detections are dropped and it
#: counts as uncalibrated. Called once per processed frame, also when N == 0, so a camera
#: tracker behind it sees every frame.
FeetMapper = Callable[[int, np.ndarray, np.ndarray], Optional[np.ndarray]]


def homography_mapper(Hs: Sequence[Optional[np.ndarray]]) -> FeetMapper:
    """Per-frame image -> pitch homographies (e.g. from `ClipCalibrator`)."""
    def mapper(idx: int, frame_bgr, dets: np.ndarray) -> Optional[np.ndarray]:
        H = Hs[idx] if 0 <= idx < len(Hs) else None
        if H is None:
            return None
        if not len(dets):
            return np.empty((0, 2))
        b = np.asarray(dets[:, :4], float)
        feet = np.column_stack([b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3]])
        return apply_homography(H, feet)
    return mapper


@dataclass
class PipelineConfig:
    stride: int = 1                 # process every Nth frame
    max_frames: Optional[int] = None
    #: Discard anything the homography places outside the pitch. These are
    #: nearly always substitutes, staff and spectators, and they wreck both the
    #: team clustering and the "22 players" sanity check. 6 m was too generous --
    #: the spurious boxes on a sampled frame sat at y = 40 m on a 38 m half-width,
    #: i.e. just off the touchline, and sailed through.
    pitch_margin_m: float = 2.0
    #: The same test for ball candidates. The spare balls lying beyond the touchline
    #: (ball-boy stations) are real balls to the detector and sit 3.5-4 m out on the
    #: sample clips; this margin is what removes them. A wider margin let them through
    #: and they became "the ball" in most frames.
    ball_margin_m: float = 2.0
    #: Ball continuity: of several ball candidates, keep the one the ball could have
    #: reached from where it was last seen, at up to this speed (+ `ball_slack_m`).
    #: After `ball_reacquire_s` without a sighting, take the most confident again.
    ball_max_speed: float = 35.0
    ball_slack_m: float = 1.0
    ball_reacquire_s: float = 1.0
    #: A ball candidate that sat within 0.5 m of one spot for 80% of this long is a
    #: still decoy unless nothing else moves (`BallSelector`).
    ball_still_s: float = 2.0
    #: Choose the ball by following a trajectory across the whole clip instead of frame by
    #: frame (`analytics.ball_link`). Off by default until the benchmark says it is better.
    #: The ball then bypasses the tracker: its rows are written after the pass, one track id
    #: per unbroken stretch.
    ball_link: bool = False
    #: What a lone ball detection must beat to be taken when the ball could not have got
    #: there from the previous pick. A candidate that continues the track needs nothing.
    ball_link_break_cost: float = 0.25
    #: Keep every raw ball candidate (box, confidence, whether it survived calibration and
    #: the on-pitch filter) in `report['_ball_candidates']`, so a benchmark can tell a
    #: detector miss from a selection loss. Cheap, but not for a long match.
    keep_ball_candidates: bool = False
    use_appearance: bool = True
    #: Drop person tracks seen in fewer frames than this (0.2 s at 25 fps). 1 keeps everything.
    min_track_frames: int = 5
    team_fit_sample: int = 4000     # descriptors sampled to fit the kit clusters
    #: Join track fragments into whole-player tracklets before deciding identity.
    #: On real football, this collapsed 129 fragments back to 22 correct players
    #: with zero welded identities (scripts/test_identity.py).
    stitch: bool = True
    #: Keyword overrides for `identity.stitch_tracklets`.
    stitch_args: dict = field(default_factory=dict)
    #: Players a side. Drives the team quota; set lower after a red card.
    roster_size: int = 11
    #: Take people who live on the boundary lines out of the teams before the quota:
    #: moving ones become assistant referees, still ones `Role.OTHER` (ball boys,
    #: staff). See `identity.touchline_people`.
    touchline_people: bool = True
    #: Find goalkeepers by position (inside a penalty area, deepest towards the goal)
    #: and give them the team that defends that end, whatever their kit clustered as.
    #: See `identity.goalkeeper_tracklets`.
    goalkeepers: bool = True
    #: Give each goalkeeper slot (end of the pitch x team) one id across the whole clip, cuts
    #: included: a side has one keeper, so this is role continuity, not forced identity.
    merge_keepers: bool = True
    #: Fill ball gaps of at most this many frames by interpolating between the
    #: two sightings either side. 0 disables. The tracker already coasts the ball
    #: for a few frames, so this only reaches the gaps where the track died.
    ball_max_gap: int = 5
    #: Keep up to 600 frames for the team montage (`report['_montage']`). Off for batch
    #: runs: at 1080p that is 3.7 GB of RAM.
    montage: bool = True
    #: Mark replays, close-ups, crowd shots and graphics (`live_wide` / `shot` columns), so analytics
    #: skip them. None = on for broadcast footage only: on a fixed rig every frame is live and the
    #: grass-share test would only add ways to be wrong. Rows are never dropped, only flagged.
    shot_filter: Optional[bool] = None
    shot: ShotConfig = field(default_factory=ShotConfig)
    verbose: bool = True


def on_pitch(xy: np.ndarray, roles: list[str], pitch: Pitch,
             margin_m: float, ball_margin_m: float) -> np.ndarray:
    """Which pitch positions to keep: players within `margin_m` of the lines, the ball
    within `ball_margin_m`. Non-finite positions are never kept."""
    return np.array([
        bool(np.isfinite(p).all()) and pitch.contains(
            p[0], p[1], ball_margin_m if r == Role.BALL.value else margin_m)
        for p, r in zip(xy, roles)
    ], dtype=bool)


def choose_ball(xy: np.ndarray, conf: np.ndarray, last_xy: Optional[np.ndarray],
                frames_since: int, fps: float, max_speed: float = 35.0,
                slack_m: float = 1.0, reacquire_s: float = 1.0) -> Optional[int]:
    """Index of the ball candidate to keep this frame, or None.

    With a recent sighting, only candidates the ball could have reached are
    eligible and the nearest wins; if none is reachable the frame gets no ball
    rather than a teleport. Without one (start, or `reacquire_s` since the last
    sighting) the most confident candidate is taken. The reach is the tracker's
    own ball gate (`max_speed * (frames_since + 1) / fps + slack_m`), so a kept
    candidate is never one the tracker would then refuse and re-id.
    """
    if not len(xy):
        return None
    if last_xy is None or frames_since > reacquire_s * fps:
        return int(np.argmax(conf))
    reach = max_speed * (frames_since + 1) / fps + slack_m
    d = np.linalg.norm(xy - last_xy, axis=1)
    ok = np.flatnonzero(d <= reach)
    return int(ok[np.argmin(d[ok])]) if len(ok) else None


class BallSelector:
    """One ball per frame from several candidates: continuity, and no still decoys.

    Continuity alone (`choose_ball`) locks onto a ball that lies still: it is
    always at distance 0 from itself. On the 3-minute clip a second ball lay in
    the box for 57 s and was "the ball" in 45% of sightings. A candidate that has
    sat within `still_m` of one spot for `still_share` of the last `still_s`
    seconds is *still*; a still one with nobody within `alone_m` is a *decoy*.
    A decoy is never taken, and a held still ball is let go as soon as some other
    candidate is not still. A match ball waiting for a restart has the taker
    beside it, so it is not a decoy and stays held while nothing else moves.
    `people_xy` (this frame's people, pitch metres) enables the decoy test;
    without it every still ball is treated as a decoy.
    """

    def __init__(self, fps: float, max_speed: float = 35.0, slack_m: float = 1.0,
                 reacquire_s: float = 1.0, still_m: float = 0.5, still_s: float = 2.0,
                 still_share: float = 0.8, alone_m: float = 3.0):
        self.fps, self.max_speed, self.slack_m = fps, max_speed, slack_m
        self.reacquire_s, self.still_m, self.still_share = reacquire_s, still_m, still_share
        self.alone_m = alone_m
        self.window = max(int(round(still_s * fps)), 1)
        self._hist: deque = deque()
        self.last_xy: Optional[np.ndarray] = None
        self.last_frame = -10**9
        self.n_unreachable = 0
        self.n_released = 0

    def _still(self, frame: int, xy: np.ndarray) -> np.ndarray:
        while self._hist and self._hist[0][0] < frame - self.window:
            self._hist.popleft()
        if not len(xy) or len(self._hist) < self.still_share * self.window:
            return np.zeros(len(xy), bool)
        hits = np.zeros(len(xy))
        for _, h in self._hist:
            if len(h):
                hits += np.linalg.norm(xy[:, None] - h[None], axis=-1).min(axis=1) <= self.still_m
        return hits >= self.still_share * self.window

    def flags(self, frame: int, xy: np.ndarray, people_xy: Optional[np.ndarray] = None
              ) -> tuple[np.ndarray, np.ndarray]:
        """(still, decoy) per candidate. Call once per calibrated frame, with that frame's
        candidates (maybe none): it also advances the still-ball history."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        still = self._still(frame, xy)
        self._hist.append((frame, xy.copy()))
        decoy = still.copy()
        if len(xy) and people_xy is not None and len(people_xy) and still.any():
            near = np.linalg.norm(xy[:, None] - np.asarray(people_xy, float)[None], axis=-1).min(axis=1)
            decoy &= near > self.alone_m
        return still, decoy

    def pick(self, frame: int, xy: np.ndarray, conf: np.ndarray,
             people_xy: Optional[np.ndarray] = None) -> Optional[int]:
        """Call on every calibrated frame, with that frame's ball candidates (maybe none)."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        conf = np.asarray(conf, float).ravel()
        still, decoy = self.flags(frame, xy, people_xy)
        if not len(xy):
            return None
        moving = np.flatnonzero(~still)
        if self.last_xy is None or frame - self.last_frame > self.reacquire_s * self.fps:
            ok = np.flatnonzero(~decoy)
            p = int(ok[np.argmax(conf[ok])]) if len(ok) else None
        else:
            p = choose_ball(xy, conf, self.last_xy, frame - self.last_frame, self.fps,
                            self.max_speed, self.slack_m, self.reacquire_s)
            if p is None:
                self.n_unreachable += 1
            elif still[p] and len(moving):
                p = int(moving[np.argmax(conf[moving])])
                self.n_released += 1
            elif decoy[p]:
                p = None
                self.n_released += 1
        if p is not None:
            self.last_xy, self.last_frame = xy[p].copy(), frame
        return p


def link_ball_rows(log: list, fps: float, first_track_id: int,
                   cfg: PipelineConfig) -> tuple[list[dict], int]:
    """Observation rows for the ball from the clip's candidates, by `link_ball_candidates`.

    `log` is (frame, pitch xy, detections, still, decoy) per frame that had ball candidates.
    Decoys (a still ball with nobody near it) cannot be picked; a still ball that someone
    stands beside scores half. One track id per unbroken stretch, as the tracker would give.
    """
    frames = [e[0] for e in log]
    xy = [e[1] for e in log]
    conf = [e[2][:, 4] for e in log]
    penalty = [np.where(e[4], 1.0, np.where(e[3], 0.5, 0.0)) for e in log]
    picks = link_ball_candidates(
        frames, xy, conf, fps, penalty=penalty, max_speed=cfg.ball_max_speed,
        slack_m=cfg.ball_slack_m, max_gap_s=cfg.ball_reacquire_s,
        break_cost=cfg.ball_link_break_cost)
    rows: list[dict] = []
    tid, prev = first_track_id - 1, None
    for (frame, pts, dets, _, _), p in zip(log, picks):
        if p is None:
            continue
        if (prev is None or frame - prev[0] > cfg.ball_reacquire_s * fps
                or np.linalg.norm(pts[p] - prev[1]) > cfg.ball_max_speed * (frame - prev[0]) / fps
                + cfg.ball_slack_m):
            tid += 1
        prev = (frame, pts[p])
        d = dets[p]
        rows.append(dict(
            frame_idx=frame, period=1, timestamp=frame / fps, track_id=tid,
            role=Role.BALL.value, team=Team.UNKNOWN.value, jersey=np.nan,
            bbox_x=d[0], bbox_y=d[1], bbox_w=d[2], bbox_h=d[3], det_conf=d[4],
            x=pts[p][0], y=pts[p][1], z=0.0))
    return rows, tid - first_track_id + 1 if rows else 0


def _log(on: bool, *a):
    if on:
        print(*a, flush=True)


@dataclass
class Observation:
    """Everything one processed frame gives the tracker, before any tracking decision.

    This is the seam between the expensive half of the pipeline (decode, detect, calibrate,
    describe kit colour: GPU) and the cheap half (track, stitch, decide teams, roles and
    identity: CPU, seconds). Pickle a clip's observations once (`save_observations`) and
    `run_observations` replays everything after it on a laptop, so a tracking or stitching
    change can be measured without a GPU.
    """
    dets: np.ndarray                    # (N, >=6) raw detector output, boxes in pixels
    roles: list[str]                    # per detection
    xy: Optional[np.ndarray]            # (N, 2) feet in pitch metres; None = frame not calibrated
    desc: Optional[np.ndarray] = None   # (N, D) kit descriptor per detection; zeros for the ball
    signal: Optional[object] = None     # shots.FrameSignals, when the shot filter is on
    frame: Optional[np.ndarray] = field(default=None, repr=False)   # RGB, only for the montage


def save_observations(path: str | Path, obs: Sequence[Observation], fps: float,
                      has_referee_class: bool) -> None:
    import pickle

    slim = [Observation(o.dets.astype(np.float32), list(o.roles), o.xy,
                        None if o.desc is None else o.desc.astype(np.float16), o.signal)
            for o in obs]
    with open(path, "wb") as f:
        pickle.dump({"fps": fps, "has_referee_class": has_referee_class, "obs": slim}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)


def load_observations(path: str | Path) -> tuple[list[Observation], float, bool]:
    import pickle

    with open(path, "rb") as f:
        d = pickle.load(f)
    obs = d["obs"]
    for o in obs:
        if o.desc is not None:
            o.desc = o.desc.astype(np.float32)
    return obs, d["fps"], d["has_referee_class"]


def observe_frames(
    frames: Iterable[np.ndarray],
    fps: float,
    mapper: FeetMapper,
    detector: Detector,
    cfg: PipelineConfig,
    source: str = "frames",
) -> Iterator[Observation]:
    """Decode-side half of the pipeline: one `Observation` per processed frame (see `stride`)."""
    n_total = len(frames) if hasattr(frames, "__len__") else None
    eff_fps = fps / cfg.stride
    shots_on = cfg.shot_filter
    _log(cfg.verbose, f"[pipeline] {source}")
    src_idx = out_idx = 0
    for frame_bgr in frames:
        if src_idx == 0:
            h, w = frame_bgr.shape[:2]
            _log(cfg.verbose, f"[pipeline] {w}x{h}, "
                              + (f"{n_total} frames @ " if n_total is not None else "")
                              + f"{fps:.2f} fps, stride {cfg.stride} -> "
                                f"effective {eff_fps:.2f} fps")
        if src_idx % cfg.stride != 0:
            src_idx += 1
            continue
        if cfg.max_frames is not None and out_idx >= cfg.max_frames:
            break

        frame = frame_bgr[:, :, ::-1]                       # BGR -> RGB
        dets = detector.detect(frame)
        roles = [detector.role_of(c) for c in dets[:, 5]]
        signal = None
        if shots_on:
            people_h = [b[3] for b, r in zip(dets[:, :4], roles) if r != Role.BALL.value]
            signal = frame_signals(frame_bgr, float(np.median(people_h)) / frame_bgr.shape[0]
                                   if people_h else float("nan"))
        xy = mapper(src_idx, frame_bgr, dets)
        desc = None
        if cfg.use_appearance and len(dets) and xy is not None:
            rows = [None if r == Role.BALL.value else kit_descriptor(frame, b)
                    for b, r in zip(dets[:, :4], roles)]
            dim = next((d.shape for d in rows if d is not None), None) \
                or kit_descriptor(frame, dets[0, :4]).shape
            desc = np.stack([np.zeros(dim, np.float32) if d is None else d for d in rows])
        yield Observation(dets, roles, xy, desc, signal, frame if cfg.montage else None)
        src_idx += 1
        out_idx += 1


def _tee(obs: Iterable[Observation], sink: list) -> Iterator[Observation]:
    for o in obs:
        sink.append(Observation(o.dets, o.roles, o.xy, o.desc, o.signal))
        yield o


def run_frames(
    frames: Iterable[np.ndarray],
    fps: float,
    mapper: FeetMapper,
    detector: Detector,
    meta: MatchMeta,
    cfg: PipelineConfig | None = None,
    tracker_cfg: TrackerConfig | None = None,
    source: str = "frames",
    capture: Optional[list] = None,
) -> tuple[MatchState, dict]:
    """Process BGR frames into a MatchState. Returns (state, report).

    `capture`, if given, collects every frame's `Observation` (see `save_observations`).

    `mapper(src_idx, frame_bgr, dets)` places each frame's detections on the pitch (see
    `FeetMapper`): a fixed calibration, a camera tracker, or per-frame homographies
    (`homography_mapper`). `fps` is the source frame rate; `cfg.stride` thins it.
    """
    cfg = cfg or PipelineConfig()
    shots = (meta.source_type == "broadcast") if cfg.shot_filter is None else cfg.shot_filter
    cfg = replace(cfg, shot_filter=shots)
    obs = observe_frames(frames, fps, mapper, detector, cfg, source)
    if capture is not None:
        obs = _tee(obs, capture)
    return run_observations(obs, fps, meta, cfg, tracker_cfg,
                            has_referee_class=Role.REFEREE.value in set(detector.roles.values()))


def run_observations(
    observations: Iterable[Observation],
    fps: float,
    meta: MatchMeta,
    cfg: PipelineConfig | None = None,
    tracker_cfg: TrackerConfig | None = None,
    has_referee_class: bool = False,
) -> tuple[MatchState, dict]:
    """Track, stitch and decide identity over pre-computed `Observation`s. `fps` is the
    *source* rate; observations are already strided."""
    cfg = cfg or PipelineConfig()
    pitch = Pitch(meta.pitch_length, meta.pitch_width)
    src_fps = fps

    # The effective frame rate after striding is what the tracker and every
    # speed downstream must use. Getting this wrong silently scales every
    # distance and speed in the match.
    eff_fps = src_fps / cfg.stride
    meta.fps = eff_fps
    tracker = PitchTracker(tracker_cfg or TrackerConfig(fps=eff_fps))
    n_uncalibrated = 0
    shots_on = (meta.source_type == "broadcast") if cfg.shot_filter is None else cfg.shot_filter
    signals = []

    rows: list[dict] = []
    desc_rows: list[np.ndarray] = []
    desc_track: list[int] = []
    desc_role: list[str] = []
    montage_frames, montage_boxes = [], []

    out_idx = 0
    n_det_total = 0
    n_off_pitch = 0
    n_ball_off_pitch = 0
    ball_sel = BallSelector(eff_fps, cfg.ball_max_speed, cfg.ball_slack_m,
                            cfg.ball_reacquire_s, still_s=cfg.ball_still_s)
    ball_cands: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    link_log: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    t0 = time.time()

    for ob in observations:
        dets, xy_all, roles_all, desc_all = ob.dets, ob.xy, ob.roles, ob.desc
        n_det_total += len(dets)
        if shots_on:
            signals.append(ob.signal)

        if cfg.keep_ball_candidates and len(dets):
            is_b = np.array([r == Role.BALL.value for r in roles_all], bool)
            if is_b.any():
                ok = (np.zeros(int(is_b.sum()), bool) if xy_all is None else
                      on_pitch(xy_all[is_b], [Role.BALL.value] * int(is_b.sum()), pitch,
                               cfg.pitch_margin_m, cfg.ball_margin_m))
                ball_cands[out_idx] = (dets[is_b, :5].copy(), ok)
        if xy_all is None:
            n_uncalibrated += 1
            dets = dets[:0]                                  # skip the frame, tracks coast

        if len(dets):
            roles = list(roles_all)
            xy = xy_all                                      # same length and order as dets

            keep = on_pitch(xy, roles, pitch, cfg.pitch_margin_m, cfg.ball_margin_m)
            n_off_pitch += int((~keep).sum())
            n_ball_off_pitch += sum(1 for r, k in zip(roles, keep) if r == Role.BALL.value and not k)
            dets, xy = dets[keep], xy[keep]
            roles = [r for r, k in zip(roles, keep) if k]
            desc = None if desc_all is None else desc_all[keep]

            # One ball per frame, chosen for continuity and against still decoys.
            bi = np.flatnonzero([r == Role.BALL.value for r in roles])
            people = [i for i, r in enumerate(roles) if r != Role.BALL.value]
            if cfg.ball_link:
                # decided after the pass, over the whole clip; none goes to the tracker
                if len(bi):
                    still, decoy = ball_sel.flags(out_idx, xy[bi], xy[people])
                    link_log.append((out_idx, xy[bi].copy(), dets[bi].copy(), still, decoy))
                drop = set(bi.tolist())
            else:
                pick = ball_sel.pick(out_idx, xy[bi], dets[bi, 4], people_xy=xy[people])
                drop = (set(bi.tolist()) - ({int(bi[pick])} if pick is not None else set())
                        if len(bi) else set())
            if drop:
                sel = np.array([i not in drop for i in range(len(dets))], bool)
                dets, xy = dets[sel], xy[sel]
                roles = [r for r, k in zip(roles, sel) if k]
                if desc is not None:
                    desc = desc[sel]

            embeddings = desc if (len(dets) and cfg.use_appearance) else None
            if len(dets):
                matches = tracker.update(
                    xy, roles, bboxes=dets[:, :4], confs=dets[:, 4],
                    embeddings=embeddings,
                )
                for di, tid in matches:
                    rows.append(dict(
                        frame_idx=out_idx, period=1, timestamp=out_idx / eff_fps,
                        track_id=tid, role=roles[di], team=Team.UNKNOWN.value,
                        jersey=np.nan,
                        bbox_x=dets[di, 0], bbox_y=dets[di, 1],
                        bbox_w=dets[di, 2], bbox_h=dets[di, 3],
                        det_conf=dets[di, 4],
                        x=xy[di, 0], y=xy[di, 1],
                        z=0.0 if roles[di] == Role.BALL.value else np.nan,
                    ))
                    if embeddings is not None and roles[di] != Role.BALL.value:
                        desc_rows.append(embeddings[di])
                        desc_track.append(tid)
                        desc_role.append(roles[di])
                        if cfg.montage and ob.frame is not None and len(montage_frames) < 600:
                            montage_frames.append(ob.frame)
                            montage_boxes.append(dets[di, :4])

        if cfg.verbose and out_idx % 50 == 0:
            el = time.time() - t0
            _log(True, f"  frame {out_idx}  ({out_idx / max(el, 1e-6):.1f} fps proc, "
                       f"{len(rows):,} observations)")
        out_idx += 1

    elapsed = time.time() - t0

    n_link_segments = 0
    if cfg.ball_link and link_log:
        link_rows, n_link_segments = link_ball_rows(
            link_log, eff_fps, first_track_id=tracker._next_id, cfg=cfg)
        rows.extend(link_rows)
        rows.sort(key=lambda r: r["frame_idx"])

    if not rows:
        raise RuntimeError("no detections at all -- wrong weights, or the "
                           "calibration is placing everything off the pitch")

    df = pd.DataFrame(rows)
    if cfg.min_track_frames > 1:
        # The tracker reports a detection from its first frame, so a one-frame false alarm
        # (a doubled box, a spectator at the touchline) becomes a track id. Real people last.
        n = df.groupby("track_id")["frame_idx"].transform("size")
        ball = df["role"] == Role.BALL.value
        df = df[(n >= cfg.min_track_frames) | ball].reset_index(drop=True)
        alive = set(df["track_id"].unique().tolist())
        sel = [i for i, t in enumerate(desc_track) if t in alive]
        desc_rows = [desc_rows[i] for i in sel]
        desc_role = [desc_role[i] for i in sel]
        desc_track = [desc_track[i] for i in sel]
    shot_report: dict = {}
    if shots_on:
        live, shot = live_wide_segments(signals, eff_fps, cfg.shot)
        _, replay = stinger_and_replay(signals, eff_fps, cfg.shot)
        df["live_wide"] = live[df["frame_idx"].to_numpy()]
        df["shot"] = shot[df["frame_idx"].to_numpy()]
        shot_report = {
            "live_wide_share": float(live.mean()),
            "replay_seconds": float(replay.sum() / eff_fps),
            "replays": int(np.sum(np.diff(np.r_[0, replay.astype(int)]) == 1)),
            "shots": int(shot.max()) + 1,
        }
        _log(cfg.verbose, f"[pipeline] shots: {shot_report['live_wide_share']:.0%} live wide, "
                          f"{shot_report['replays']} replays ({shot_report['replay_seconds']:.0f} s), "
                          f"{shot_report['shots']} shots")
    report: dict = {
        **shot_report,
        "frames_processed": out_idx,
        "seconds": elapsed,
        "fps_processed": out_idx / max(elapsed, 1e-6),
        "detections": n_det_total,
        "dropped_off_pitch": n_off_pitch,
        "ball_dropped_off_pitch": n_ball_off_pitch,
        "ball_frames_unreachable": ball_sel.n_unreachable,
        "ball_still_released": ball_sel.n_released,
        "ball_link_segments": n_link_segments,
        "ball_link_frames": sum(r["role"] == Role.BALL.value for r in rows) if cfg.ball_link else 0,
        "tracks_created": tracker._next_id - 1,
        "scene_jumps_compensated": tracker.n_comp,
        "frames_uncalibrated": n_uncalibrated,
        "frames_calibrated": out_idx - n_uncalibrated,
        "calib_success_rate": (out_idx - n_uncalibrated) / max(out_idx, 1),
    }

    if cfg.keep_ball_candidates:
        report["_ball_candidates"] = ball_cands

    # ---- role, once per track ---------------------------------------------
    role_vote = (df.groupby(["track_id", "role"], observed=True).size()
                   .unstack(fill_value=0).idxmax(axis=1))
    df["role"] = df["track_id"].map(role_vote)
    state = MatchState(meta, df)

    # ---- identity, once per tracklet ---------------------------------------
    # Deliberately after the pass, not during it. Identity is a property of a
    # person across the whole match, so it is decided once, with the benefit of
    # everything each tracklet turned out to be -- and constrained by the fact
    # that football is eleven a side.
    clf = None
    scores: dict[int, float] = {}
    if desc_rows:
        D = np.stack(desc_rows); T = np.array(desc_track)
        idx = np.arange(len(D))
        if len(idx) > cfg.team_fit_sample:
            idx = np.random.default_rng(0).choice(idx, cfg.team_fit_sample, replace=False)
        clf = TeamClassifier().fit(D[idx])
        diag = clf.diagnose(T, D)
        scores = team_scores_by_track(T, D)
        report["team_diagnosis"] = diag
        report["_team_clf"] = clf
        if cfg.montage:
            report["_montage"] = (montage_frames, montage_boxes,
                                  T[:len(montage_frames)], D[:len(montage_frames)])
        _log(cfg.verbose, f"[pipeline] teams: {diag['summary']}")

    track_emb: dict[int, np.ndarray] = {}
    if desc_rows:
        Dm = np.stack(desc_rows); Tm = np.array(desc_track)
        for tid in np.unique(Tm):
            track_emb[int(tid)] = Dm[Tm == tid].mean(axis=0)

    # Detectors without a referee class (the SoccerMaster x6 weights expose only "person")
    # would put the officials into one team and skew the eleven-a-side quota. Split a third
    # kit off first; with two kits the helper returns nothing and the 2-means path is untouched.
    officials: set[int] = set()
    if desc_rows and not has_referee_class:
        info: dict = {}
        officials = split_officials(np.array(desc_track), np.stack(desc_rows), info=info)
        report["officials_check"] = info
        _log(cfg.verbose, f"[pipeline] officials: {info.get('reason')} {info}")
        if officials:
            _log(cfg.verbose, f"[pipeline] officials split off (third kit): {sorted(officials)}")
    report["officials_split"] = len(officials)

    tracklets = build_tracklets(state, embeddings=track_emb or None, team_scores=scores or None)
    n_before = len(tracklets)
    if cfg.stitch:
        tracklets = stitch_tracklets(tracklets, eff_fps, **cfg.stitch_args)
    for t in tracklets:
        if scores:
            w = [(scores.get(x, 0.0), 1.0) for x in t.track_ids]
            t.team_score = float(np.mean([v for v, _ in w])) if w else 0.0
    # People who live on the boundary lines -- assistant referees, ball boys, staff --
    # are not players, whatever kit cluster they fall into. Decide before the quota.
    on_line = (touchline_people(state.tracks, tracklets, pitch.half_l, pitch.half_w)
               if cfg.touchline_people else {})
    report["touchline_people"] = {
        "assistant_referees": sum(r == Role.REFEREE.value for r in on_line.values()),
        "other": sum(r == Role.OTHER.value for r in on_line.values()),
    }
    # A stitched tracklet is an official when most of its frames are, not when one of its
    # fragments is: a single mis-clustered fragment must not turn a player into a referee.
    frames_of = state.tracks.groupby("track_id").size().to_dict()

    def official_share(t) -> float:
        return (sum(frames_of.get(i, 0) for i in t.track_ids if i in officials)
                / max(sum(frames_of.get(i, 0) for i in t.track_ids), 1))

    # Goalkeepers first, among everyone who is not on a touchline: a keeper wears a third kit,
    # which is exactly what the official split looks for, and position says who he is better.
    keepers: dict[int, int] = {}
    if cfg.goalkeepers and scores:
        keepers = goalkeeper_tracklets(
            state.tracks, [t for t in tracklets if t.id not in on_line], pitch.half_l, pitch.half_w)
    official_tracklets = [t for t in tracklets if t.id not in keepers
                          and (official_share(t) >= 0.5 or on_line.get(t.id) == Role.REFEREE.value)]
    other_tracklets = [t for t in tracklets
                       if on_line.get(t.id) == Role.OTHER.value and t.id not in keepers
                       and official_share(t) < 0.5]
    side = {id(t) for t in official_tracklets} | {id(t) for t in other_tracklets}
    player_tracklets = [t for t in tracklets if id(t) not in side]
    if scores:
        player_tracklets = assign_teams_with_roster(player_tracklets, team_size=cfg.roster_size)
    if keepers:
        keeper_team = goalkeeper_teams(state.tracks, player_tracklets, keepers)
        for t in player_tracklets:
            if t.id in keeper_team:
                t.role, t.team = Role.GOALKEEPER.value, keeper_team[t.id]
        keepers = {k: v for k, v in keepers.items() if k in keeper_team}
        if cfg.merge_keepers and keepers:
            # One keeper a side: every sighting at the same end for the same team is him.
            slot = {k: (end, keeper_team[k]) for k, end in keepers.items()}
            player_tracklets = merge_same_role(player_tracklets, slot)
            keepers = {t.id: slot[t.id][0] for t in player_tracklets if t.id in slot}
    report["goalkeepers"] = len(keepers)
    for t in official_tracklets:
        t.team = Team.OFFICIAL.value
        t.role = Role.REFEREE.value
    for t in other_tracklets:
        t.team = Team.UNKNOWN.value
        t.role = Role.OTHER.value
    tracklets = player_tracklets + official_tracklets + other_tracklets
    state = apply_to_state(state, tracklets)
    report["tracklets_before_stitch"] = n_before
    report["tracklets_after_stitch"] = len(tracklets)
    _log(cfg.verbose, f"[pipeline] tracklets: {n_before} -> {len(tracklets)} after stitching")

    if cfg.ball_max_gap:
        before = int((state.tracks["role"].astype(str) == Role.BALL.value).sum())
        state.tracks = interpolate_ball(state.tracks, eff_fps, max_gap=cfg.ball_max_gap,
                                        last_frame=out_idx - 1)
        after = int((state.tracks["role"].astype(str) == Role.BALL.value).sum())
        report["ball_frames_detected"] = before
        report["ball_frames_interpolated"] = after - before
        _log(cfg.verbose, f"[pipeline] ball: {before} frames seen, "
                          f"{after - before} interpolated (gaps <= {cfg.ball_max_gap})")

    state.tracks = add_kinematics(state.tracks, eff_fps)
    state = MatchState(meta, state.tracks)

    report["validation"] = state.validate()
    players_per_frame = state.players.groupby("frame_idx").size()
    report["median_players_per_frame"] = (
        float(players_per_frame.median()) if len(players_per_frame) else 0.0)

    _log(cfg.verbose, f"[pipeline] {state}")
    _log(cfg.verbose, f"[pipeline] median players/frame: "
                      f"{report['median_players_per_frame']:.1f} (expect ~22 full-pitch)")
    for p in report["validation"]:
        _log(cfg.verbose, f"[pipeline] WARNING: {p}")
    return state, report


def run(
    video_path: str | Path,
    calibration: Calibration,
    detector: Detector,
    meta: MatchMeta,
    cfg: PipelineConfig | None = None,
    tracker_cfg: TrackerConfig | None = None,
    camera: AnchoredCamera | None = None,
) -> tuple[MatchState, dict]:
    """Process a clip into a MatchState. Returns (state, report).

    `camera`: pass an AnchoredCamera (anchored on the frame the calibration was clicked on)
    when the camera pans, zooms or cuts. Each frame is related to the anchor before mapping;
    frames that cannot be related (a bench close-up) are skipped and counted in the report.
    With `camera=None` the behaviour is the fixed-camera path, unchanged.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or meta.fps
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (W, H) != tuple(calibration.image_size):
        cap.release()
        raise ValueError(
            f"calibration was made for {calibration.image_size} but this video is "
            f"{(W, H)}. A homography is resolution-specific -- recalibrate, or "
            f"scale the clicked points."
        )

    if camera is not None:
        moving = MovingCalibration(calibration)

        def mapper(idx: int, frame_bgr: np.ndarray, dets: np.ndarray) -> Optional[np.ndarray]:
            mask = np.full(frame_bgr.shape[:2], 255, np.uint8)
            for b in dets[:, :4]:
                x, y, w, h = (int(v) for v in b)
                cv2.rectangle(mask, (x - 6, y - 6), (x + w + 6, y + h + 6), 0, -1)
            H_a_t = camera.update(frame_bgr, mask=mask)
            if H_a_t is None:
                return None
            return (moving.feet_to_pitch(dets[:, :4], H_a_t) if len(dets) else np.empty((0, 2)))
    else:
        def mapper(idx: int, frame_bgr: np.ndarray, dets: np.ndarray) -> Optional[np.ndarray]:
            return calibration.feet_to_pitch(dets[:, :4]) if len(dets) else np.empty((0, 2))

    def read_frames():
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                yield frame_bgr
        finally:
            cap.release()

    frames = read_frames()
    try:
        state, report = run_frames(frames, src_fps, mapper, detector, meta, cfg, tracker_cfg,
                                   source=str(video_path))
    finally:
        frames.close()                                       # release the capture on early exit too
    if camera is not None:
        report["reanchors"] = camera.n_reanchors
    return state, report

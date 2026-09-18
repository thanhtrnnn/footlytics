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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..analytics.ball import interpolate_ball
from ..analytics.kinematics import add_kinematics
from ..geometry.homography import Calibration
from ..geometry.pitch import Pitch
from ..perception.detect import Detector, DetectorConfig
from ..perception.teams import split_officials, TeamClassifier, kit_descriptor
from ..perception.identity import (apply_to_state, assign_teams_with_roster,
                                   build_tracklets, stitch_tracklets)
from ..perception.track import PitchTracker, TrackerConfig
from ..geometry.camera_motion import AnchoredCamera, MovingCalibration
from ..state.schema import MatchMeta, MatchState, Role, Team, TeamInfo


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
    use_appearance: bool = True
    team_fit_sample: int = 4000     # descriptors sampled to fit the kit clusters
    #: Join track fragments into whole-player tracklets before deciding identity.
    #: On real football, this collapsed 129 fragments back to 22 correct players
    #: with zero welded identities (scripts/test_identity.py).
    stitch: bool = True
    #: Players a side. Drives the team quota; set lower after a red card.
    roster_size: int = 11
    #: Fill ball gaps of at most this many frames by interpolating between the
    #: two sightings either side. 0 disables. The tracker already coasts the ball
    #: for a few frames, so this only reaches the gaps where the track died.
    ball_max_gap: int = 5
    verbose: bool = True


def _log(on: bool, *a):
    if on:
        print(*a, flush=True)


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

    cfg = cfg or PipelineConfig()
    pitch = Pitch(meta.pitch_length, meta.pitch_width)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or meta.fps
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (W, H) != tuple(calibration.image_size):
        raise ValueError(
            f"calibration was made for {calibration.image_size} but this video is "
            f"{(W, H)}. A homography is resolution-specific -- recalibrate, or "
            f"scale the clicked points."
        )

    # The effective frame rate after striding is what the tracker and every
    # speed downstream must use. Getting this wrong silently scales every
    # distance and speed in the match.
    eff_fps = src_fps / cfg.stride
    meta.fps = eff_fps
    tracker = PitchTracker(tracker_cfg or TrackerConfig(fps=eff_fps))
    moving = MovingCalibration(calibration) if camera is not None else None
    n_uncalibrated = 0

    _log(cfg.verbose, f"[pipeline] {video_path}")
    _log(cfg.verbose, f"[pipeline] {W}x{H}, {n_total} frames @ {src_fps:.2f} fps, "
                      f"stride {cfg.stride} -> effective {eff_fps:.2f} fps")

    rows: list[dict] = []
    desc_rows: list[np.ndarray] = []
    desc_track: list[int] = []
    desc_role: list[str] = []
    montage_frames, montage_boxes = [], []

    src_idx = out_idx = 0
    n_det_total = 0
    n_off_pitch = 0
    t0 = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if src_idx % cfg.stride != 0:
            src_idx += 1
            continue
        if cfg.max_frames is not None and out_idx >= cfg.max_frames:
            break

        frame = frame_bgr[:, :, ::-1]                       # BGR -> RGB
        dets = detector.detect(frame)
        n_det_total += len(dets)

        H_a_t = None
        if camera is not None:
            mask = np.full(frame_bgr.shape[:2], 255, np.uint8)
            for b in dets[:, :4]:
                x, y, w, h = (int(v) for v in b)
                cv2.rectangle(mask, (x - 6, y - 6), (x + w + 6, y + h + 6), 0, -1)
            H_a_t = camera.update(frame_bgr, mask=mask)
            if H_a_t is None:
                n_uncalibrated += 1
                dets = dets[:0]                              # skip the frame, tracks coast

        if len(dets):
            roles = [detector.role_of(c) for c in dets[:, 5]]
            xy = (moving.feet_to_pitch(dets[:, :4], H_a_t) if moving is not None
                  else calibration.feet_to_pitch(dets[:, :4]))

            keep = np.array([
                np.isfinite(p).all() and pitch.contains(p[0], p[1], cfg.pitch_margin_m)
                for p in xy
            ])
            n_off_pitch += int((~keep).sum())
            dets, xy = dets[keep], xy[keep]
            roles = [r for r, k in zip(roles, keep) if k]

            embeddings = None
            if len(dets):
                if cfg.use_appearance:
                    embeddings = np.stack([
                        kit_descriptor(frame, b) if r != Role.BALL.value
                        else np.zeros(kit_descriptor(frame, dets[0, :4]).shape, np.float32)
                        for b, r in zip(dets[:, :4], roles)
                    ])
                matches = tracker.update(
                    xy, roles, bboxes=dets[:, :4], confs=dets[:, 4],
                    embeddings=embeddings,
                )
                live = {t.id: t for t in tracker.tracks}
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
                        if len(montage_frames) < 600:
                            montage_frames.append(frame)
                            montage_boxes.append(dets[di, :4])

        if cfg.verbose and out_idx % 50 == 0:
            el = time.time() - t0
            _log(True, f"  frame {out_idx}  ({out_idx / max(el, 1e-6):.1f} fps proc, "
                       f"{len(rows):,} observations)")
        src_idx += 1
        out_idx += 1

    cap.release()
    elapsed = time.time() - t0

    if not rows:
        raise RuntimeError("no detections at all -- wrong weights, or the "
                           "calibration is placing everything off the pitch")

    df = pd.DataFrame(rows)
    report: dict = {
        "frames_processed": out_idx,
        "seconds": elapsed,
        "fps_processed": out_idx / max(elapsed, 1e-6),
        "detections": n_det_total,
        "dropped_off_pitch": n_off_pitch,
        "tracks_created": tracker._next_id - 1,
    }
    if camera is not None:
        report["frames_uncalibrated"] = n_uncalibrated
        report["frames_calibrated"] = out_idx - n_uncalibrated
        report["calib_success_rate"] = (out_idx - n_uncalibrated) / max(out_idx, 1)
        report["reanchors"] = camera.n_reanchors

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
        scores = clf.track_scores(T, D)
        report["team_diagnosis"] = diag
        report["_team_clf"] = clf
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
    if desc_rows and Role.REFEREE.value not in set(detector.roles.values()):
        officials = split_officials(np.array(desc_track), np.stack(desc_rows))
        if officials:
            _log(cfg.verbose, f"[pipeline] officials split off (third kit): {sorted(officials)}")
    report["officials_split"] = len(officials)

    tracklets = build_tracklets(state, embeddings=track_emb or None)
    n_before = len(tracklets)
    if cfg.stitch:
        tracklets = stitch_tracklets(tracklets, eff_fps)
    for t in tracklets:
        if scores:
            w = [(scores.get(x, 0.0), 1.0) for x in t.track_ids]
            t.team_score = float(np.mean([v for v, _ in w])) if w else 0.0
    official_tracklets = [t for t in tracklets if set(t.track_ids) & officials]
    player_tracklets = [t for t in tracklets if not (set(t.track_ids) & officials)]
    if scores:
        player_tracklets = assign_teams_with_roster(player_tracklets, team_size=cfg.roster_size)
    for t in official_tracklets:
        t.team = Team.OFFICIAL.value
        t.role = Role.REFEREE.value
    tracklets = player_tracklets + official_tracklets
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

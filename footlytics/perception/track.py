"""Multi-object tracking, done in pitch coordinates rather than pixels.

Most trackers associate detections in image space. That is a workaround for not
knowing the geometry. Once the pitch is calibrated -- which for a fixed camera
happens once, before the season -- tracking in metres is strictly better:

* Motion is genuinely near-constant-velocity in metres. In pixels, a player
  running at a steady 7 m/s covers wildly different pixel distances near the
  camera versus at the far touchline, so any image-space motion model is wrong
  somewhere in the frame.
* The gate becomes physical. "No human moves more than 12 m/s" is a hard,
  interpretable constraint; "no more than 80 px/frame" is a guess that is
  simultaneously too tight in the foreground and too loose at distance.
* Occlusion handling improves, because a coasting Kalman prediction in metres
  stays plausible for the ~1 s a player spends behind another.

Deliberately not a re-implementation of ByteTrack: the second association pass
over low-confidence detections buys much less here than on broadcast footage,
because a fixed full-pitch camera rarely loses a player to frame exit.

What motion alone cannot do
---------------------------
Measured on a synthetic 22-player match with *perfect* detections
(scripts/test_tracker.py), the residual identity switches are not spread
randomly: 100% of them occur while another player is within 2 m, median 1.05 m,
against a 9 m baseline separation. Geometry and a motion model simply do not
contain the information needed to tell two players apart in a duel.

That is why `update()` accepts optional appearance embeddings. Kit colour alone
resolves most of it (opponents differ); a ReID embedding resolves teammates; and
jersey numbers settle the remainder. Without at least one of them, expect
identity to degrade steadily through every set piece and every tackle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..state.schema import Role

# Ball and people never associate with each other.
_BALL = Role.BALL.value
_PERSON_ROLES = (Role.PLAYER.value, Role.GOALKEEPER.value, Role.REFEREE.value, Role.OTHER.value)


@dataclass
class TrackerConfig:
    fps: float = 25.0
    #: Hard association gate. Usain Bolt peaked at ~12.4 m/s; the ball goes far
    #: faster, hence a separate gate.
    max_speed: float = 12.0
    max_ball_speed: float = 35.0
    #: Frames a track survives unmatched before it is retired. At 25 fps, 30
    #: frames = 1.2 s, about how long a player stays fully occluded.
    max_age: int = 30
    #: Detections needed before a track is considered real and reported.
    min_hits: int = 3
    #: sigma of the acceleration driving the process noise, m/s^2.
    accel_noise: float = 3.5
    #: sigma of a single position measurement, metres. Dominated by
    #: bounding-box jitter, not by the homography.
    meas_noise: float = 0.35
    #: Metres of association cost charged per unit of cosine appearance distance.
    #: 6.0 means "a completely different-looking crop is as bad as being 6 m away",
    #: which comfortably dominates the sub-2 m ambiguity of a duel.
    appearance_weight: float = 6.0
    #: Hard veto: never associate across an appearance gap wider than this.
    max_appearance_dist: float = 0.5
    #: EMA factor for a track's running appearance template. Low = long memory,
    #: which is what we want, since a track's kit does not change.
    appearance_ema: float = 0.15


class _KF:
    """Constant-velocity Kalman filter on [x, y, vx, vy] in metres."""

    __slots__ = ("x", "P", "F", "Q", "R", "H")

    def __init__(self, xy: np.ndarray, dt: float, accel_noise: float, meas_noise: float):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], float)
        self.P = np.diag([meas_noise**2, meas_noise**2, 25.0, 25.0])
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        G = np.array([dt * dt / 2, dt * dt / 2, dt, dt])
        self.Q = np.outer(G, G) * accel_noise**2
        self.R = np.eye(2) * meas_noise**2
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, z: np.ndarray) -> None:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def pos(self) -> np.ndarray:
        return self.x[:2]

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))


@dataclass
class Track:
    id: int
    kf: _KF
    kind: str                       # "ball" or "person"
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    role_votes: dict[str, float] = field(default_factory=dict)
    last_bbox: Optional[np.ndarray] = None
    min_hits: int = 3               # set from TrackerConfig.min_hits at creation
    last_conf: float = 0.0
    #: Last *observed* position. The gate is anchored here rather than on the
    #: filter's prediction -- see PitchTracker.update.
    last_meas: Optional[np.ndarray] = None
    #: Running L2-normalised appearance template, or None if not in use.
    embedding: Optional[np.ndarray] = None

    def blend_embedding(self, e: np.ndarray, ema: float) -> None:
        e = e / max(float(np.linalg.norm(e)), 1e-9)
        self.embedding = e if self.embedding is None else (
            (1 - ema) * self.embedding + ema * e)
        self.embedding /= max(float(np.linalg.norm(self.embedding)), 1e-9)

    @property
    def role(self) -> str:
        if self.kind == "ball":
            return _BALL
        if not self.role_votes:
            return Role.PLAYER.value
        return max(self.role_votes.items(), key=lambda kv: kv[1])[0]

    @property
    def confirmed(self) -> bool:
        return self.hits >= self.min_hits


class PitchTracker:
    """Associates per-frame pitch-space detections into persistent tracks."""

    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.dt = 1.0 / self.cfg.fps
        self.tracks: list[Track] = []
        self._next_id = 1

    # ------------------------------------------------------------------ step

    def update(
        self,
        xy: np.ndarray,
        roles: Sequence[str],
        bboxes: Optional[np.ndarray] = None,
        confs: Optional[Sequence[float]] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> list[tuple[int, int]]:
        """Advance one frame.

        `xy` is (N, 2) in pitch metres, `roles` the per-detection role strings.
        `embeddings` is an optional (N, D) array of appearance features -- kit
        colour histograms or ReID vectors; they need not be normalised.
        Returns (detection_index, track_id) for every matched detection.
        """
        xy = np.asarray(xy, float).reshape(-1, 2)
        roles = list(roles)
        confs = list(confs) if confs is not None else [1.0] * len(xy)
        kinds = np.array(["ball" if r == _BALL else "person" for r in roles])
        if embeddings is not None:
            embeddings = np.asarray(embeddings, float).reshape(len(xy), -1)
            embeddings = embeddings / np.maximum(
                np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)

        for t in self.tracks:
            t.kf.predict()
            t.age += 1
            t.time_since_update += 1

        matches: list[tuple[int, int]] = []
        used_det: set[int] = set()

        for kind in ("person", "ball"):
            det_idx = np.flatnonzero(kinds == kind)
            trk = [t for t in self.tracks if t.kind == kind]
            gate = (self.cfg.max_speed if kind == "person" else self.cfg.max_ball_speed) * self.dt
            # Coasting tracks are allowed a proportionally wider gate.
            if det_idx.size and trk:
                # Two different distances, doing two different jobs.
                #
                # The GATE is physical: an object cannot be further from where it
                # was last *seen* than its top speed allows. Anchoring the gate on
                # the filter's prediction instead is wrong, and fails exactly when
                # it matters -- a kicked ball reverses direction, the constant-
                # velocity filter confidently predicts several metres the wrong
                # way, and the true detection falls outside a gate that was sized
                # for physical displacement. The track then dies and is reborn
                # with a new id on every single kick.
                #
                # The COST is predictive: among candidates that survive the gate,
                # the one nearest the filter's prediction is the best match.
                pred = np.array([t.kf.pos for t in trk])
                anchor = np.array([
                    t.last_meas if t.last_meas is not None else t.kf.pos for t in trk
                ])
                d_cost = np.linalg.norm(xy[det_idx][:, None, :] - pred[None, :, :], axis=-1)
                d_gate = np.linalg.norm(xy[det_idx][:, None, :] - anchor[None, :, :], axis=-1)
                limit = np.array([gate * (1 + t.time_since_update) + 1.0 for t in trk])
                ok = d_gate <= limit[None, :]

                # Appearance turns a 1 m geometric ambiguity into an easy call.
                if embeddings is not None and kind == "person":
                    have = np.array([t.embedding is not None for t in trk])
                    if have.any():
                        tmpl = np.stack([
                            t.embedding if t.embedding is not None
                            else np.zeros(embeddings.shape[1]) for t in trk
                        ])
                        cos_d = 1.0 - embeddings[det_idx] @ tmpl.T
                        cos_d[:, ~have] = 0.0            # no template yet: no opinion
                        ok &= (cos_d <= self.cfg.max_appearance_dist) | (~have)[None, :]
                        d_cost = d_cost + self.cfg.appearance_weight * cos_d

                cost = np.where(ok, d_cost, 1e6)
                r_i, c_i = linear_sum_assignment(cost)
                for ri, ci in zip(r_i, c_i):
                    if cost[ri, ci] >= 1e6:
                        continue
                    di = int(det_idx[ri]); t = trk[ci]
                    t.kf.update(xy[di])
                    t.last_meas = xy[di].copy()
                    t.hits += 1
                    t.time_since_update = 0
                    t.last_conf = float(confs[di])
                    if bboxes is not None:
                        t.last_bbox = np.asarray(bboxes[di], float)
                    t.role_votes[roles[di]] = t.role_votes.get(roles[di], 0.0) + float(confs[di])
                    if embeddings is not None:
                        t.blend_embedding(embeddings[di], self.cfg.appearance_ema)
                    matches.append((di, t.id))
                    used_det.add(di)

            for di in det_idx:
                di = int(di)
                if di in used_det:
                    continue
                t = Track(
                    id=self._next_id,
                    kf=_KF(xy[di], self.dt, self.cfg.accel_noise, self.cfg.meas_noise),
                    kind=kind,
                    role_votes={roles[di]: float(confs[di])}, min_hits=self.cfg.min_hits,
                    last_bbox=None if bboxes is None else np.asarray(bboxes[di], float),
                    last_conf=float(confs[di]),
                    last_meas=xy[di].copy(),
                    embedding=None if embeddings is None else embeddings[di].copy(),
                )
                self._next_id += 1
                self.tracks.append(t)
                matches.append((di, t.id))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.cfg.max_age]
        return matches

    # ----------------------------------------------------------------- views

    def active(self, include_coasting: bool = True) -> list[Track]:
        """Tracks worth reporting this frame."""
        return [
            t for t in self.tracks
            if t.hits >= self.cfg.min_hits and (include_coasting or t.time_since_update == 0)
        ]


def resolve_roles(tracks_df, min_frames: int = 10):
    """Collapse noisy per-frame roles to one role per track, by majority vote.

    Tracks shorter than `min_frames` keep their per-frame roles: a 4-frame track is
    not evidence of anything, and collapsing it would only hide the flicker.

    Per-frame role classification flickers -- a goalkeeper is called a player
    for a handful of frames whenever they step off their line. Role is a
    property of the *person*, not of the frame, so decide it once per track.
    """
    import pandas as pd

    df = tracks_df
    counts = df.groupby(["track_id", "role"], observed=True).size().unstack(fill_value=0)
    if counts.empty:
        return df
    winner = counts.idxmax(axis=1)
    # Not enough evidence to overrule the per-frame classifier: keep those frames as they are.
    winner = winner[counts.sum(axis=1) >= min_frames]
    out = df.copy()
    voted = out["track_id"].map(winner)
    out["role"] = voted.where(voted.notna(), out["role"]).astype("category")
    return out

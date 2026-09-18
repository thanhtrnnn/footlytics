"""The Match State: the single contract every other module talks through.

Perception writes it. Analytics, visualisation and the LLM layer only ever read
it. That boundary is the whole architecture -- it means the tracking model can
be swapped for a better one (or for a commercial feed such as Bepro's own
FIFA-certified tracking) without a single analytics function changing.

Coordinate convention
---------------------
Pitch coordinates are metres with the origin at the centre spot:

        x in [-L/2, +L/2]   along the length, positive towards the away goal
        y in [-W/2, +W/2]   along the width,  positive towards the far touchline
        z in [0, inf)       height, only ever populated for the ball

This matches kloppy's metric pitch convention, so `to_kloppy()` is a relabel
rather than a transform. Attacking direction is *not* normalised here; that is
an analytics-time concern (see `analytics.orientation`), because throwing away
the raw direction early makes second-half bugs invisible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class Team(str, Enum):
    HOME = "home"
    AWAY = "away"
    OFFICIAL = "official"   # referees and assistants
    UNKNOWN = "unknown"


class Role(str, Enum):
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"
    OTHER = "other"


#: Columns of the observation table, in order, with their dtypes.
#: One row = one tracked object in one frame.
TRACK_SCHEMA: dict[str, str] = {
    "frame_idx":  "int32",      # 0-based index into the source video
    "period":     "int8",       # 1 = first half, 2 = second half, 3/4 = ET
    "timestamp":  "float32",    # seconds since kickoff of that period
    "track_id":   "int32",      # stable within a period; -1 = untracked detection
    "role":       "category",
    "team":       "category",
    "jersey":     "float32",    # nullable on purpose: NaN = not yet read
    # image space (pixels, in the coordinate frame of the stitched/source view)
    "bbox_x":     "float32",    # left
    "bbox_y":     "float32",    # top
    "bbox_w":     "float32",
    "bbox_h":     "float32",
    "det_conf":   "float32",
    # pitch space (metres, origin = centre spot)
    "x":          "float32",
    "y":          "float32",
    "z":          "float32",    # ball height; NaN for people
    # derived kinematics, filled by analytics.kinematics
    "speed":      "float32",    # m/s
    "accel":      "float32",    # m/s^2
    # provenance: True where the position was filled in rather than observed,
    # by analytics.ball.interpolate_ball. Never True for people.
    "interpolated": "bool",
}

_REQUIRED = ("frame_idx", "timestamp", "track_id", "role", "team", "x", "y")


@dataclass
class TeamInfo:
    name: str
    short_name: str = ""
    #: BGR colour sampled from the kit, used for the radar and for sanity checks
    kit_color: tuple[int, int, int] = (255, 255, 255)
    #: True if this team attacks +x during period 1
    attacks_positive_x_first_half: bool = True


@dataclass
class MatchMeta:
    match_id: str
    fps: float
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    home: TeamInfo = field(default_factory=lambda: TeamInfo("Home", "HOM"))
    away: TeamInfo = field(default_factory=lambda: TeamInfo("Away", "AWY"))
    competition: str = ""
    date: str = ""                     # ISO 8601
    venue: str = ""
    #: Where the pixels came from -- "broadcast", "fixed_panoramic", "tactical_cam".
    #: Analytics that are only valid on full-pitch footage check this.
    source_type: str = "unknown"
    #: Free-form provenance: model versions, calibration file, git sha.
    provenance: dict = field(default_factory=dict)

    @property
    def full_pitch_visible(self) -> bool:
        return self.source_type in ("fixed_panoramic", "tactical_cam")


class MatchState:
    """Metadata plus a tidy table of per-frame observations.

    Deliberately a thin wrapper over a DataFrame rather than a graph of objects:
    a 90-minute match is ~1.5M rows, and every analytic we care about is a
    groupby.
    """

    def __init__(self, meta: MatchMeta, tracks: pd.DataFrame):
        self.meta = meta
        self.tracks = self._coerce(tracks)

    # ------------------------------------------------------------------ build

    @staticmethod
    def _coerce(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in _REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"MatchState is missing required columns: {missing}")

        df = df.copy()
        for col, dtype in TRACK_SCHEMA.items():
            if col not in df.columns:
                df[col] = pd.Series(np.nan, index=df.index)
            if dtype == "category":
                df[col] = df[col].astype("object").fillna("unknown").astype("category")
            elif dtype == "bool":
                # Must fill before the cast: astype(bool) turns NaN into True,
                # which would mark every row of a fresh table as interpolated.
                df[col] = df[col].fillna(False).astype(dtype)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

        df = df[list(TRACK_SCHEMA)]
        return df.sort_values(["frame_idx", "track_id"], kind="stable").reset_index(drop=True)

    @classmethod
    def empty(cls, meta: MatchMeta) -> "MatchState":
        return cls(meta, pd.DataFrame({c: pd.Series(dtype=object) for c in _REQUIRED}))

    # ------------------------------------------------------------------ views

    @property
    def players(self) -> pd.DataFrame:
        """Outfielders and keepers only -- no ball, no officials."""
        return self.tracks[self.tracks["role"].isin([Role.PLAYER.value, Role.GOALKEEPER.value])]

    @property
    def ball(self) -> pd.DataFrame:
        return self.tracks[self.tracks["role"] == Role.BALL.value]

    def frame(self, frame_idx: int) -> pd.DataFrame:
        return self.tracks[self.tracks["frame_idx"] == frame_idx]

    def team(self, team: Team | str) -> pd.DataFrame:
        return self.players[self.players["team"] == str(getattr(team, "value", team))]

    @property
    def n_frames(self) -> int:
        return 0 if self.tracks.empty else int(self.tracks["frame_idx"].max()) + 1

    @property
    def duration_s(self) -> float:
        return 0.0 if self.tracks.empty else float(self.tracks["timestamp"].max())

    # ------------------------------------------------------------------- i/o

    def save(self, path: str | Path) -> Path:
        """Parquet for the table, JSON sidecar for the metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tracks.to_parquet(path / "tracks.parquet", index=False)
        (path / "meta.json").write_text(json.dumps(asdict(self.meta), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MatchState":
        path = Path(path)
        raw = json.loads((path / "meta.json").read_text())
        raw["home"] = TeamInfo(**raw["home"])
        raw["away"] = TeamInfo(**raw["away"])
        return cls(MatchMeta(**raw), pd.read_parquet(path / "tracks.parquet"))

    # ------------------------------------------------------------------ checks

    def validate(self) -> list[str]:
        """Cheap sanity checks. Returns human-readable problems, worst first.

        These exist because a tracking bug looks exactly like a tactical
        insight until someone checks the units.
        """
        problems: list[str] = []
        t = self.tracks
        if t.empty:
            return ["match state is empty"]

        half_l, half_w = self.meta.pitch_length / 2, self.meta.pitch_width / 2
        margin = 5.0  # players legitimately step off the pitch
        off = t[(t["x"].abs() > half_l + margin) | (t["y"].abs() > half_w + margin)]
        if len(off) / len(t) > 0.01:
            problems.append(
                f"{len(off) / len(t):.1%} of observations fall outside the pitch "
                f"(+{margin}m tolerance) -- homography is probably wrong"
            )

        per_frame = self.players.groupby("frame_idx").size()
        if not per_frame.empty and self.meta.full_pitch_visible:
            typical = per_frame.median()
            if typical < 18:
                problems.append(
                    f"median {typical:.0f} players per frame on full-pitch footage "
                    "(expected ~22) -- detector is missing players"
                )

        speeds = self.players["speed"].dropna()
        if not speeds.empty:
            impossible = (speeds > 12.5).mean()   # 12.5 m/s ~ faster than Mbappe
            if impossible > 0.001:
                problems.append(
                    f"{impossible:.2%} of player speeds exceed 12.5 m/s -- "
                    "identity switches or a bad frame rate"
                )

        unknown = (self.players["team"] == Team.UNKNOWN.value).mean()
        if unknown > 0.1:
            problems.append(f"{unknown:.1%} of player observations have no team assigned")

        # Are the configured pitch dimensions even right?
        #
        # This is the check that would have caught a real error on SoccerTrack
        # v2: the loader assumed the IFAB default 68 m width, players spanned
        # 106% of it, and 2.14% of positions sat outside the touchline -- up to
        # 11 m out, systematically, for whole passages of play. The fixed
        # off-pitch margin below missed it because 5 m of tolerance is more than
        # the error on most frames.
        #
        # The trick is to need no ground truth. Players fill a pitch similarly
        # along both axes, so if the configured length is right, the fraction of
        # it they span calibrates what to expect across the width. Spanning MORE
        # than 100% of the configured width is impossible and proves the number
        # wrong on its own.
        if len(self.players) > 1000 and self.meta.full_pitch_visible:
            px = self.players["x"].to_numpy(); py = self.players["y"].to_numpy()
            span_x = float(np.percentile(px, 99.5) - np.percentile(px, 0.5))
            span_y = float(np.percentile(py, 99.5) - np.percentile(py, 0.5))
            cov_x = span_x / self.meta.pitch_length
            cov_y = span_y / self.meta.pitch_width
            if cov_y > 1.0:
                implied = span_y / max(cov_x, 1e-6)
                problems.append(
                    f"players span {cov_y:.0%} of the configured {self.meta.pitch_width:.0f} m "
                    f"width, which is impossible. Length coverage is {cov_x:.0%}, implying a "
                    f"true width nearer {implied:.0f} m. Measure the pitch -- the IFAB default "
                    f"is an assumption, not a fact."
                )
            elif cov_x > 1.0:
                implied = span_x / max(cov_y, 1e-6)
                problems.append(
                    f"players span {cov_x:.0%} of the configured {self.meta.pitch_length:.0f} m "
                    f"length, which is impossible; true length is nearer {implied:.0f} m."
                )

        # Eleven a side. A lopsided split does not just look wrong on a radar --
        # it silently corrupts every team-level statistic downstream, so it is
        # checked here rather than left to whoever happens to glance at a plot.
        named = self.players[self.players["team"].isin(
            [Team.HOME.value, Team.AWAY.value])]
        if len(named) and self.meta.full_pitch_visible:
            per = (named.groupby(["frame_idx", "team"], observed=True).size()
                        .unstack(fill_value=0))
            for col in (Team.HOME.value, Team.AWAY.value):
                if col not in per:
                    per[col] = 0
            h = per[Team.HOME.value].median()
            a = per[Team.AWAY.value].median()
            if abs(h - a) > 3:
                problems.append(
                    f"team split is {h:.0f} v {a:.0f} per frame, not ~11 v 11 -- "
                    "team assignment is wrong and every team-level statistic "
                    "built on it will be too"
                )

        return problems

    def __repr__(self) -> str:
        return (
            f"<MatchState {self.meta.match_id!r} "
            f"{self.meta.home.short_name}-{self.meta.away.short_name} "
            f"{self.n_frames} frames / {self.duration_s / 60:.1f} min, "
            f"{len(self.tracks):,} observations>"
        )

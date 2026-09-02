"""Assemble the game-state table from tracks, team labels, ball and homographies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footlytics.homography import PITCH_LENGTH_M, PITCH_WIDTH_M, image_to_pitch

GAMESTATE_COLUMNS = [
    "timestamp_s",
    "frame",
    "player_id",
    "team",
    "role",
    "x_m",
    "y_m",
    "ball_x_m",
    "ball_y_m",
    "det_conf",
    "calib_ok",
    "interpolated",
]
REQUIRED_NON_NULL = ["timestamp_s", "frame", "player_id", "team", "x_m", "y_m", "det_conf", "calib_ok"]


def build_gamestate(
    tracks: pd.DataFrame,
    teams: dict[int, int | str],
    ball: pd.DataFrame | None,
    fps: float,
    homographies: dict[int, np.ndarray] | None,
) -> pd.DataFrame:
    """tracks: frame, track_id, x1, y1, x2, y2, conf, cls. teams: track_id -> label.

    Positions use the bbox foot point (centre x, bottom y). If a homography exists for the
    frame, positions are mapped to pitch metres and calib_ok=True; otherwise image pixels.
    """
    t = tracks.copy()
    t["fx"] = (t["x1"] + t["x2"]) / 2.0
    t["fy"] = t["y2"].astype(float)
    xs = np.full(len(t), np.nan)
    ys = np.full(len(t), np.nan)
    calib = np.zeros(len(t), dtype=bool)
    for frame, idx in t.groupby("frame").indices.items():
        H = homographies.get(int(frame)) if homographies else None
        pts = t.iloc[idx][["fx", "fy"]].to_numpy()
        if H is not None:
            mapped = image_to_pitch(H, pts)
            xs[idx], ys[idx] = mapped[:, 0], mapped[:, 1]
            calib[idx] = True
        else:
            xs[idx], ys[idx] = pts[:, 0], pts[:, 1]
    gs = pd.DataFrame(
        {
            "timestamp_s": t["frame"].astype(float) / fps,
            "frame": t["frame"].astype(int),
            "player_id": t["track_id"].astype(int),
            "team": t["track_id"].map(lambda i: str(teams.get(int(i), "unknown"))),
            "role": t["cls"].astype(str) if "cls" in t else "player",
            "x_m": xs,
            "y_m": ys,
            "det_conf": t["conf"].astype(float),
            "calib_ok": calib,
        }
    )
    if ball is not None and len(ball):
        b = ball[["frame", "ball_x_m", "ball_y_m", "interpolated"]]
        gs = gs.merge(b, on="frame", how="left")
        gs["interpolated"] = gs["interpolated"].fillna(False).astype(bool)
    else:
        gs["ball_x_m"] = np.nan
        gs["ball_y_m"] = np.nan
        gs["interpolated"] = False
    gs = gs.sort_values(["frame", "player_id"]).reset_index(drop=True)
    return gs[GAMESTATE_COLUMNS]


def validate_gamestate(gs: pd.DataFrame) -> None:
    missing = [c for c in GAMESTATE_COLUMNS if c not in gs.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    bad = [c for c in REQUIRED_NON_NULL if gs[c].isna().any()]
    if bad:
        raise ValueError(f"NaN in required columns: {bad}")
    if not gs["timestamp_s"].is_monotonic_increasing:
        raise ValueError("timestamp_s not monotonic")


def filter_off_pitch(gs: pd.DataFrame, margin_m: float = 2.0) -> pd.DataFrame:
    """Drop tracks whose median calibrated position lies outside the pitch plus margin.

    Rows without calibration are kept untouched (coordinates are pixels there).
    """
    if gs.empty or not gs["calib_ok"].any():
        return gs
    med = gs[gs["calib_ok"]].groupby("player_id")[["x_m", "y_m"]].median()
    inside = med["x_m"].between(-margin_m, PITCH_LENGTH_M + margin_m) & med["y_m"].between(-margin_m, PITCH_WIDTH_M + margin_m)
    keep = set(med.index[inside]) | set(gs.loc[~gs["calib_ok"], "player_id"])
    return gs[gs["player_id"].isin(keep)].reset_index(drop=True)

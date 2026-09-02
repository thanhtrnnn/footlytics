"""Ball trajectory post-processing."""
from __future__ import annotations

import pandas as pd


def interpolate_ball(df: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """Linearly fill NaN gaps in ball_x_m/ball_y_m of length <= max_gap frames.

    Adds boolean column `interpolated`. Longer gaps stay NaN. Assumes one row per frame,
    sorted by frame.
    """
    out = df.sort_values("frame").reset_index(drop=True).copy()
    missing = out["ball_x_m"].isna()
    filled = out[["ball_x_m", "ball_y_m"]].interpolate(method="linear", limit_area="inside")
    # identify run lengths of missing values
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    ok = missing & (run_len <= max_gap) & filled["ball_x_m"].notna()
    out.loc[ok, ["ball_x_m", "ball_y_m"]] = filled.loc[ok, ["ball_x_m", "ball_y_m"]]
    out["interpolated"] = ok.astype(bool)
    return out

import numpy as np
import pandas as pd

from footlytics.ball import interpolate_ball


def test_short_gaps_are_filled_long_gaps_left_nan():
    frames = np.arange(12)
    x = np.array([0, 1, np.nan, np.nan, 4, 5, np.nan, np.nan, np.nan, np.nan, np.nan, 11], dtype=float)
    y = x.copy()
    df = pd.DataFrame({"frame": frames, "ball_x_m": x, "ball_y_m": y})
    out = interpolate_ball(df, max_gap=3)
    assert np.allclose(out.loc[2:3, "ball_x_m"].to_numpy(), [2, 3])
    assert out.loc[6:10, "ball_x_m"].isna().all()
    assert out.loc[2, "interpolated"]
    assert not out.loc[1, "interpolated"]
    assert not out.loc[6, "interpolated"]


def test_no_nan_input_unchanged():
    df = pd.DataFrame({"frame": [0, 1, 2], "ball_x_m": [1.0, 2.0, 3.0], "ball_y_m": [1.0, 1.0, 1.0]})
    out = interpolate_ball(df, max_gap=3)
    assert out["ball_x_m"].tolist() == [1.0, 2.0, 3.0]
    assert not out["interpolated"].any()

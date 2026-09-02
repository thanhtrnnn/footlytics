import numpy as np
import pandas as pd
import pytest

from footlytics.gamestate import GAMESTATE_COLUMNS, build_gamestate, validate_gamestate


def _tracks():
    # frame, track_id, x1,y1,x2,y2, conf
    rows = []
    for f in range(3):
        for tid in (1, 2):
            rows.append({"frame": f, "track_id": tid, "x1": 10 * tid, "y1": 20, "x2": 10 * tid + 5, "y2": 40, "conf": 0.9, "cls": "person"})
    return pd.DataFrame(rows)


def test_columns_and_types():
    tracks = _tracks()
    teams = {1: 0, 2: 1}
    ball = pd.DataFrame({"frame": [0, 1, 2], "ball_x_m": [np.nan] * 3, "ball_y_m": [np.nan] * 3, "interpolated": [False] * 3})
    gs = build_gamestate(tracks, teams, ball, fps=25.0, homographies=None)
    assert list(gs.columns) == GAMESTATE_COLUMNS
    assert gs["timestamp_s"].is_monotonic_increasing
    assert gs.loc[gs["frame"] == 1, "timestamp_s"].iloc[0] == pytest.approx(1 / 25.0)
    assert not gs["player_id"].isna().any()
    assert not gs["team"].isna().any()
    assert (~gs["calib_ok"]).all()  # no homographies -> image space only
    validate_gamestate(gs)


def test_foot_point_used_for_position_in_image_space():
    tracks = _tracks()
    gs = build_gamestate(tracks, {1: 0, 2: 1}, None, fps=25.0, homographies=None)
    row = gs[(gs["frame"] == 0) & (gs["player_id"] == 1)].iloc[0]
    assert row["x_m"] == pytest.approx(12.5)  # bbox centre x
    assert row["y_m"] == pytest.approx(40.0)  # bbox bottom y


def test_validate_rejects_nan_required():
    tracks = _tracks()
    gs = build_gamestate(tracks, {1: 0, 2: 1}, None, fps=25.0, homographies=None)
    gs.loc[0, "player_id"] = np.nan
    with pytest.raises(ValueError):
        validate_gamestate(gs)

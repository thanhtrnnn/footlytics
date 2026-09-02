import numpy as np
import pandas as pd

from footlytics.quality import compute_quality


def _gs():
    rows = []
    # frame 0: 20 players, frame 1: 22 players, frame 2: 10 players; one new track appears in frame 2
    counts = {0: 20, 1: 22, 2: 10}
    for f, n in counts.items():
        for pid in range(1, n + 1):
            rows.append({"timestamp_s": f / 25, "frame": f, "player_id": pid, "team": pid % 2, "role": "person",
                         "x_m": 1.0, "y_m": 1.0, "ball_x_m": np.nan if f == 2 else 5.0, "ball_y_m": np.nan if f == 2 else 5.0,
                         "det_conf": 0.9, "calib_ok": f != 2, "interpolated": False})
    rows.append({"timestamp_s": 2 / 25, "frame": 2, "player_id": 99, "team": 1, "role": "person", "x_m": 1.0, "y_m": 1.0,
                 "ball_x_m": np.nan, "ball_y_m": np.nan, "det_conf": 0.9, "calib_ok": False, "interpolated": False})
    return pd.DataFrame(rows)


def test_quality_metrics():
    q = compute_quality(_gs(), fps=25.0, wall_seconds=6.0, expected_players=22)
    assert q["frames"] == 3
    assert q["mean_players_per_frame"] == (20 + 22 + 11) / 3
    assert q["pct_frames_ge_20_players"] == 2 / 3
    assert q["pct_frames_ge_18_players"] == 2 / 3
    assert q["track_births_after_first_frame"] == 3  # ids 21, 22 born at frame 1; id 99 at frame 2
    assert q["ball_coverage"] == 2 / 3
    assert q["calib_success_rate"] == 2 / 3
    assert q["seconds_per_match_minute"] == 6.0 / (3 / 25 / 60)


def test_id_switch_rate_per_player_per_minute():
    q = compute_quality(_gs(), fps=25.0, wall_seconds=6.0, expected_players=22)
    minutes = 3 / 25 / 60
    assert q["id_switch_rate_per_player_per_minute"] == 3 / 22 / minutes

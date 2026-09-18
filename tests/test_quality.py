"""quality.py: one machine-readable report per run, from the MatchState and the pipeline report."""
import json

import numpy as np
import pandas as pd

from footlytics.pipeline.quality import compute_quality, write_quality
from footlytics.state.schema import MatchMeta, MatchState, Role, Team, TeamInfo


def _state():
    rows = []
    fps = 25.0
    counts = {0: 20, 1: 22, 2: 10}
    for f, n in counts.items():
        for pid in range(1, n + 1):
            rows.append(dict(frame_idx=f, period=1, timestamp=f / fps, track_id=pid, role=Role.PLAYER.value,
                             team=Team.HOME.value if pid % 2 else Team.AWAY.value, jersey=np.nan,
                             bbox_x=0, bbox_y=0, bbox_w=10, bbox_h=30, det_conf=0.9, x=1.0, y=1.0, z=np.nan))
        if f < 2:
            rows.append(dict(frame_idx=f, period=1, timestamp=f / fps, track_id=-1, role=Role.BALL.value,
                             team=Team.UNKNOWN.value, jersey=np.nan, bbox_x=0, bbox_y=0, bbox_w=4, bbox_h=4,
                             det_conf=0.5, x=5.0, y=5.0, z=0.0))
    rows.append(dict(frame_idx=2, period=1, timestamp=2 / fps, track_id=99, role=Role.PLAYER.value,
                     team=Team.AWAY.value, jersey=np.nan, bbox_x=0, bbox_y=0, bbox_w=10, bbox_h=30,
                     det_conf=0.9, x=1.0, y=1.0, z=np.nan))
    meta = MatchMeta(match_id="Q", fps=fps, source_type="tactical_cam", home=TeamInfo("H", "H"), away=TeamInfo("A", "A"))
    return MatchState(meta, pd.DataFrame(rows))


def test_quality_metrics_from_state_and_report():
    st = _state()
    report = {"frames_processed": 3, "seconds": 6.0, "frames_calibrated": 2, "frames_uncalibrated": 1,
              "calib_success_rate": 2 / 3, "tracklets_after_stitch": 23, "validation": []}
    q = compute_quality(st, report, stride=1)
    assert q["frames"] == 3
    assert q["mean_players_per_frame"] == (20 + 22 + 11) / 3
    assert q["pct_frames_ge_18_players"] == 2 / 3
    assert q["ball_coverage"] == 2 / 3
    assert q["track_births_after_first_frame"] == 3          # 21, 22 at frame 1; 99 at frame 2
    minutes = 3 / 25 / 60
    assert q["id_switch_rate_per_player_per_minute"] == 3 / 22 / minutes
    assert q["seconds_per_match_minute"] == 6.0 / minutes
    assert q["calib_success_rate"] == 2 / 3
    assert q["tracklets_after_stitch"] == 23
    assert q["validation"] == []


def test_write_quality_files(tmp_path):
    st = _state()
    q = compute_quality(st, {"frames_processed": 3, "seconds": 1.0, "validation": ["x"]}, stride=2)
    write_quality(q, tmp_path)
    assert (tmp_path / "quality.json").exists() and (tmp_path / "quality.md").exists()
    assert json.loads((tmp_path / "quality.json").read_text())["frames"] == 3
    assert q["stride"] == 2
    assert q["seconds_per_match_minute"] == 1.0 / (3 / 25 / 60)   # meta.fps is already the effective rate

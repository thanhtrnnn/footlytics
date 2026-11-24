"""Tests for chunked, resumable 90-minute full-match processing (`footlytics.pipeline.match`)."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from footlytics.cli import app
from footlytics.geometry.calib_nn import CameraFit
from footlytics.pipeline.match import (
    ChunkSpec,
    assemble_match_states,
    choose_stride,
    parse_time_s,
    parse_window,
    plan_match_chunks,
    run_match,
)
from footlytics.state.schema import MatchMeta, MatchState, Role, Team
from tests.test_pipeline_frames import H


def _write_synthetic_video(path: Path, n_frames: int = 60, fps: float = 50.0) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    for i in range(n_frames):
        frame = np.full((240, 320, 3), (50, 140, 60), dtype=np.uint8)
        frame[100:160, 75:115] = (200, 40, 30)
        frame[105:165, 135:170] = (200, 45, 35)
        frame[100:160, 195:235] = (30, 40, 200)
        frame[115:175, 235:270] = (35, 45, 200)
        frame[0, 0] = i % 250
        writer.write(frame)
    writer.release()
    return path


class StubCalibrator:
    def __init__(self):
        self.calls = 0
        self.last_errors: dict = {}

    def fit(self, frames, which=None, **kw):
        self.calls += 1
        keys = list(range(len(frames))) if which is None else list(which)
        fit = CameraFit({"x_focal_length": 1500.0}, 1.0, H.copy())
        return {k: fit for k in keys}


class StubDetector:
    roles: ClassVar[dict[int, str]] = {0: Role.PLAYER.value, 1: Role.BALL.value}

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        people = [
            [80, 100, 20, 50, 0.9, 0],
            [140, 110, 20, 50, 0.9, 0],
            [200, 100, 20, 50, 0.9, 0],
            [240, 120, 20, 50, 0.9, 0],
        ]
        ball = [[150, 130, 6, 6, 0.7, 1]]
        return np.array(people + ball, dtype=float)

    def role_of(self, c):
        return self.roles[int(c)]


def test_parse_time_and_window():
    assert parse_time_s("90") == pytest.approx(90.0)
    assert parse_time_s("04:30") == pytest.approx(270.0)
    assert parse_time_s("1:07:15.5") == pytest.approx(4035.5)
    with pytest.raises(ValueError):
        parse_time_s("")
    with pytest.raises(ValueError):
        parse_time_s("-5")

    assert parse_window("05:00-50:00") == pytest.approx((300.0, 3000.0))
    assert parse_window("100..250") == pytest.approx((100.0, 250.0))
    assert parse_window("60:120") == pytest.approx((60.0, 120.0))
    with pytest.raises(ValueError):
        parse_window("300-100")


def test_choose_stride_normalises_high_fps_broadcasts():
    assert choose_stride(60.0, 25.0) == 2
    assert choose_stride(50.0, 25.0) == 2
    assert choose_stride(25.0, 25.0) == 1
    assert choose_stride(29.97, 25.0) == 1


def test_plan_match_chunks_splits_periods_and_absorbs_sliver():
    # 110 min video @ 25 fps = 165,000 frames; 1st half 0..2460s (41.0m -> 4 chunks, 1.0m sliver absorbed)
    specs = plan_match_chunks(
        total_frames=165_000,
        src_fps=25.0,
        period_windows=[(1, 0.0, 2460.0), (2, 3600.0, 6300.0)],
        chunk_minutes=10.0,
    )
    p1 = [s for s in specs if s.period == 1]
    p2 = [s for s in specs if s.period == 2]
    assert len(p1) == 4
    assert p1[0].start_frame == 0 and p1[-1].end_frame == 2460 * 25
    assert len(p2) == 5
    assert p2[0].start_frame == 3600 * 25 and p2[-1].end_frame == 6300 * 25


def test_assemble_match_states_offsets_ids_frames_and_aligns_teams():
    def _chunk_state(home_x: float, away_x: float) -> MatchState:
        rows = []
        for f in range(4):
            rows.append({
                "frame_idx": f, "period": 1, "timestamp": f / 25.0,
                "track_id": 1, "role": Role.GOALKEEPER.value,
                "team": Team.HOME.value, "x": home_x, "y": 0.0,
                "live_wide": True, "shot": 0,
            })
            rows.append({
                "frame_idx": f, "period": 1, "timestamp": f / 25.0,
                "track_id": 2, "role": Role.GOALKEEPER.value,
                "team": Team.AWAY.value, "x": away_x, "y": 0.0,
                "live_wide": True, "shot": 0,
            })
        return MatchState(MatchMeta(match_id="c", fps=25.0), pd.DataFrame(rows))

    # Period 1 chunk 0: HOME defends left (-50), AWAY defends right (+50)
    # Period 1 chunk 1: k-means flipped labels (called right +50 HOME) -> must flip back!
    # Period 2 chunk 0: teams swapped ends at halftime (real HOME now defends +50, but k-means
    # called left -50 HOME) -> must flip so real HOME at +50 stays HOME!
    s1a = ChunkSpec("p1_c00", period=1, start_frame=0, end_frame=100, start_s=0.0, end_s=4.0)
    s1b = ChunkSpec("p1_c01", period=1, start_frame=100, end_frame=200, start_s=4.0, end_s=8.0)
    s2a = ChunkSpec("p2_c00", period=2, start_frame=250, end_frame=350, start_s=10.0, end_s=14.0)
    merged = assemble_match_states(
        [
            (s1a, _chunk_state(-50.0, 50.0)),
            (s1b, _chunk_state(50.0, -50.0)),
            (s2a, _chunk_state(-50.0, 50.0)),
        ],
        25.0, "M", "HN", "CA",
    )

    p1 = merged.tracks[merged.tracks["period"] == 1]
    p2 = merged.tracks[merged.tracks["period"] == 2]
    # All period-1 HOME keepers stand at -50; all period-2 HOME keepers stand at +50
    assert (p1[p1["team"] == Team.HOME.value]["x"] == -50.0).all()
    assert (p2[p2["team"] == Team.HOME.value]["x"] == 50.0).all()
    assert set(p1["track_id"]).isdisjoint(set(p2["track_id"]))
    assert int(p1["shot"].iloc[0]) != int(p2["shot"].iloc[0])


def test_run_match_checkpoints_resumes_and_replays_cpu(tmp_path: Path, monkeypatch):
    from footlytics.bench import runner

    vid = _write_synthetic_video(tmp_path / "match.mp4", n_frames=60, fps=50.0)
    stub_cal = StubCalibrator()
    stub_det = StubDetector()
    monkeypatch.setattr(runner, "_calibrator", lambda paths: stub_cal)
    monkeypatch.setattr(runner, "_detector", lambda paths, cfg: stub_det)

    out = tmp_path / "out_match"
    # Two short halves on a 50 fps video -> auto stride=2 (25 Hz), 2 chunks of 14 frames each
    q1 = run_match(
        vid,
        out,
        period_windows=[(1, 0.0, 0.56), (2, 0.60, 1.16)],
        chunk_minutes=0.02,
        target_fps=25.0,
        every=5,
        home="Ha Noi",
        away="CAHCM",
        save_obs=True,
    )
    assert q1["chunks_total"] == 2 and q1["chunks_reused"] == 0
    assert (out / "tracks.parquet").is_file()
    assert (out / "sequences.csv").is_file()
    assert (out / "possession.csv").is_file()
    cal_calls_first = stub_cal.calls
    det_calls_first = stub_det.calls
    assert cal_calls_first > 0 and det_calls_first > 0

    # Resume: all chunks already have done.json -> zero new GPU calibration or detector calls
    q2 = run_match(
        vid,
        out,
        period_windows=[(1, 0.0, 0.56), (2, 0.60, 1.16)],
        chunk_minutes=0.02,
        target_fps=25.0,
        every=5,
        home="Ha Noi",
        away="CAHCM",
        save_obs=True,
    )
    assert q2["chunks_reused"] == 2
    assert stub_cal.calls == cal_calls_first and stub_det.calls == det_calls_first

    # CPU replay_only: rebuilds MatchState + sequences from obs.pkl without touching GPU models
    q3 = run_match(
        vid,
        out,
        period_windows=[(1, 0.0, 0.56), (2, 0.60, 1.16)],
        chunk_minutes=0.02,
        target_fps=25.0,
        replay_only=True,
    )
    assert q3["chunks_total"] == 2
    assert stub_cal.calls == cal_calls_first and stub_det.calls == det_calls_first


def test_cli_match_run_help():
    res = CliRunner().invoke(app, ["match-run", "--help"])
    assert res.exit_code == 0
    assert "--first-half" in res.output and "--replay-only" in res.output

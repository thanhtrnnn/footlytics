"""Unit and pipeline tests for track-level jersey number recognition."""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from footlytics.bench.adapters.gsr import state_to_predictions
from footlytics.bench.runner import load_suite
from footlytics.perception.jersey import (
    crop_sharpness,
    extract_jersey_crop,
    parse_jersey_number,
    read_track_jerseys,
    select_track_samples,
)
from footlytics.pipeline.radar import PipelineConfig, homography_mapper, run_frames
from footlytics.state.schema import MatchMeta, Role
from tests.test_pipeline_frames import H


class FakeJerseyDetector:
    """Three steady outfield players (tall boxes) with distinct shirt regions plus a ball."""
    roles: ClassVar[dict[int, str]] = {0: Role.PLAYER.value, 1: Role.BALL.value}

    def detect(self, frame):
        t = int(frame[0, 0, 0])
        people = [
            [60, 120, 40, 80, 0.9, 0],
            [260, 120, 40, 80, 0.9, 0],
            [460, 120, 40, 80, 0.9, 0],
        ]
        ball = [[200 + 2 * t, 340, 6, 6, 0.5, 1]]
        return np.array(people + ball, float)

    def role_of(self, c):
        return self.roles[int(c)]


class FakeJerseyReader:
    """Deterministic stub encoding predicted numbers into crop pixel values."""

    def __init__(self):
        self.calls: list[int] = []

    def read_batch(self, crops):
        self.calls.append(len(crops))
        out = []
        for c in crops:
            tag = int(c[0, 0, 0])
            if tag == 10:
                out.append((10, 0.95))
            elif tag == 23:
                out.append((23, 0.90))
            elif tag == 250:
                # Alternate between conflicting numbers on alternating crop rows
                alt = int(c[0, 0, 1])
                out.append((7 if alt % 2 == 0 else 8, 0.85))
            else:
                out.append((None, 0.0))
        return out


def _tagged_frames(n: int, h: int = 480, w: int = 640) -> list[np.ndarray]:
    out = []
    for t in range(n):
        f = np.full((h, w, 3), 90, np.uint8)
        f[0, 0] = t
        # run_frames treats inputs as BGR and flips to RGB before cropping
        # Player 0 (x=60..100, y=120..200) -> RGB (10, 40, 200) -> consistent #10
        f[120:200, 60:100] = (200, 40, 10)
        # Player 1 (x=260..300, y=120..200) -> RGB (23, 200, 40) -> consistent #23
        f[120:200, 260:300] = (40, 200, 23)
        # Player 2 (x=460..500, y=120..200) -> RGB (250, t%2, 120) -> conflicting reads 7 / 8
        f[120:200, 460:500] = (120, t % 2, 250)
        out.append(f)
    return out


def test_parse_jersey_number():
    assert parse_jersey_number("7") == 7
    assert parse_jersey_number("  23 ") == 23
    assert parse_jersey_number("#10.") == 10
    assert parse_jersey_number("09") == 9
    assert parse_jersey_number("0") is None
    assert parse_jersey_number("00") is None
    assert parse_jersey_number("100") is None
    assert parse_jersey_number("10a") is None
    assert parse_jersey_number("") is None
    assert parse_jersey_number("?!") is None


def test_extract_jersey_crop_and_sharpness():
    frame = np.full((100, 100, 3), 120, np.uint8)
    # Too short -> rejected
    assert extract_jersey_crop(frame, (10, 10, 20, 30), min_box_h=36) is None
    # Valid box partially out of bounds gets clamped safely
    crop = extract_jersey_crop(frame, (-5, 10, 50, 60), min_box_h=36, min_crop_px=12)
    assert crop is not None and crop.shape[0] >= 12 and crop.shape[1] >= 12

    flat = np.full((32, 32, 3), 100, np.uint8)
    sharp = flat.copy()
    sharp[8:24, 12:20] = 240
    assert crop_sharpness(sharp) > crop_sharpness(flat)
    assert crop_sharpness(np.empty((0, 0, 3), np.uint8)) == 0.0


def test_select_track_samples_caps_quality_and_spacing():
    c = np.zeros((16, 16, 3), np.uint8)
    samples = [
        (1, 0, c, 10.0),
        (1, 1, c, 9.5),   # within min_frame_gap=3 of frame 0 -> skipped
        (1, 2, c, 9.0),   # within min_frame_gap=3 of frame 0 -> skipped
        (1, 3, c, 8.0),   # kept
        (1, 6, c, 7.0),   # kept (cap hit at 3)
        (1, 10, c, 6.0),  # over max_per_track
        (2, 4, c, 5.0),
    ]
    selected = select_track_samples(samples, max_per_track=3, min_frame_gap=3)
    assert len(selected) == 4
    assert sum(1 for tid, _ in selected if tid == 1) == 3
    assert sum(1 for tid, _ in selected if tid == 2) == 1

    # read_track_jerseys integrates cleanly with a fake batch reader
    class StubReader:
        def read_batch(self, crops):
            return [(10, 0.9) if idx == 0 else (None, 0.0) for idx, _ in enumerate(crops)]

    df = read_track_jerseys(samples, StubReader(), max_per_track=3, min_frame_gap=3)
    assert list(df.columns) == ["track_id", "number", "confidence"]
    assert len(df) == 1 and int(df.iloc[0]["track_id"]) == 1 and int(df.iloc[0]["number"]) == 10


def test_run_frames_attaches_jerseys_and_emits_gsr_predictions():
    n = 12
    frames = _tagged_frames(n)
    reader = FakeJerseyReader()
    cfg = PipelineConfig(
        verbose=False,
        montage=False,
        stitch=False,
        touchline_people=False,
        goalkeepers=False,
        jersey_ocr=True,
        jersey_stride=1,
        jersey_min_box_h=36,
        jersey_min_reads=2,
        jersey_min_share=0.6,
    )
    state, report = run_frames(
        frames, 25.0, homography_mapper([H] * n), FakeJerseyDetector(),
        MatchMeta(match_id="SNGS-990", fps=25.0), cfg, jersey_reader=reader,
    )
    assert report["jersey_reads"] > 0
    assert report["jersey_tracklets_labelled"] == 2
    assert len(reader.calls) == 1

    players = state.tracks[state.tracks["role"] == Role.PLAYER.value]
    by_x = players.groupby("track_id")["bbox_x"].first().to_dict()
    j_by_tid = players.groupby("track_id")["jersey"].first().to_dict()
    for tid, bx in by_x.items():
        if abs(bx - 60) < 1:
            assert j_by_tid[tid] == pytest.approx(10.0)
        elif abs(bx - 260) < 1:
            assert j_by_tid[tid] == pytest.approx(23.0)
        elif abs(bx - 460) < 1:
            assert np.isnan(j_by_tid[tid])

    preds = state_to_predictions(state, "valid", use_jersey=True)["predictions"]
    player_jerseys = {
        p["attributes"]["jersey"]
        for p in preds if p["attributes"]["role"] == "player"
    }
    assert player_jerseys == {"10", "23", None}
    assert all(
        p["attributes"]["jersey"] is None
        for p in preds if p["attributes"]["role"] != "player"
    )


def test_run_frames_jersey_ocr_off_by_default():
    n = 8
    state, report = run_frames(
        _tagged_frames(n), 25.0, homography_mapper([H] * n), FakeJerseyDetector(),
        MatchMeta(match_id="t", fps=25.0),
        PipelineConfig(verbose=False, montage=False, stitch=False,
                       touchline_people=False, goalkeepers=False),
    )
    assert state.tracks["jersey"].isna().all()
    assert "jersey_reads" not in report


def test_jersey_suite_loads():
    suite = load_suite("jersey")
    assert suite["name"] == "jersey"
    assert len(suite["tasks"]) == 1
    assert suite["tasks"][0]["pipeline"]["jersey_ocr"] is True

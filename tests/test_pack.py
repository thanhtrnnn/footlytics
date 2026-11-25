"""Tests for Phase 2 Opposition Pack exporter (`footlytics pack export`) and `--club` queries."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from footlytics.analytics.pack import export_pack, format_clock
from footlytics.analytics.query import search_sequences, summarize_sequences
from footlytics.analytics.sequences import COLUMNS
from footlytics.cli import app
from footlytics.state.schema import MatchMeta, MatchState, Role, Team, TeamInfo


def _make_match_dir(
    root: Path,
    folder: str,
    *,
    match_id: str,
    home: str,
    away: str,
    seq_rows: list[dict],
    marks_sequences: dict | None = None,
    compute_seconds: float = 3600.0,
    with_tracks_and_video: bool = False,
) -> Path:
    d = root / folder
    d.mkdir(parents=True, exist_ok=True)

    full_rows = []
    for r in seq_rows:
        base = {c: None for c in COLUMNS}
        base.update(r)
        full_rows.append(base)
    pd.DataFrame(full_rows, columns=COLUMNS).to_csv(d / "sequences.csv", index=False)

    meta = MatchMeta(
        match_id=match_id,
        fps=25.0,
        source_type="broadcast",
        home=TeamInfo(home, home[:3].upper()),
        away=TeamInfo(away, away[:3].upper()),
    )
    if with_tracks_and_video:
        t_rows = []
        for f in range(150):  # 6.0 s @ 25 fps
            t_rows.append(
                {
                    "frame_idx": f,
                    "period": 1,
                    "timestamp": f / 25.0,
                    "track_id": 1,
                    "role": Role.PLAYER.value,
                    "team": Team.HOME.value,
                    "x": -32.0 + 0.05 * f,
                    "y": -5.0,
                    "live_wide": True,
                    "shot": 0,
                }
            )
            t_rows.append(
                {
                    "frame_idx": f,
                    "period": 1,
                    "timestamp": f / 25.0,
                    "track_id": 2,
                    "role": Role.PLAYER.value,
                    "team": Team.AWAY.value,
                    "x": -20.0 + 0.05 * f,
                    "y": 6.0,
                    "live_wide": True,
                    "shot": 0,
                }
            )
            t_rows.append(
                {
                    "frame_idx": f,
                    "period": 1,
                    "timestamp": f / 25.0,
                    "track_id": -1,
                    "role": Role.BALL.value,
                    "team": Team.UNKNOWN.value,
                    "x": -24.0 + 0.05 * f,
                    "y": 2.0,
                    "live_wide": True,
                    "shot": 0,
                }
            )
        MatchState(meta, pd.DataFrame(t_rows)).save(d)

        vid_path = d / f"{match_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(str(vid_path), fourcc, 25.0, (640, 360))
        for f in range(150):
            img = np.full((360, 640, 3), (35, 95, 40), dtype=np.uint8)
            cv2.circle(img, (100 + 2 * f, 180), 8, (255, 255, 255), -1)
            w.write(img)
        w.release()
    else:
        (d / "meta.json").write_text(
            json.dumps(
                {
                    "match_id": match_id,
                    "fps": 25.0,
                    "home": {"name": home, "short_name": home[:3].upper()},
                    "away": {"name": away, "short_name": away[:3].upper()},
                }
            )
        )

    (d / "quality.json").write_text(
        json.dumps(
            {
                "calib_success_rate": 0.84,
                "ball_coverage": 0.56,
                "compute_seconds": compute_seconds,
            }
        )
    )
    if marks_sequences is not None:
        (d / "marks.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "reviewer": "qa",
                    "video": f"{match_id}.mp4",
                    "video_offset_s": 0.0,
                    "sequences": marks_sequences,
                    "missed": [],
                }
            )
        )
    return d


@pytest.fixture
def two_fixture_root(tmp_path: Path) -> Path:
    # Fixture 1: PVFCAND is HOME vs HANOI
    _make_match_dir(
        tmp_path,
        "m1_pvf_hanoi",
        match_id="pvf_hanoi",
        home="PVFCAND",
        away="HANOI",
        seq_rows=[
            {
                "label": "low_block", "team": "home", "period": 1,
                "start_s": 1.5, "end_s": 4.5, "start_frame": 38, "end_frame": 112,
                "duration_s": 3.0, "def_line_x": -36.0, "compactness_m": 10.5,
                "n_within_10m": 4, "flank": "", "n_attackers": 0, "n_defenders": 0,
                "in_possession_share": 0.0, "n_pressers": 5, "response": "",
            },
            {
                "label": "flank_overload", "team": "home", "period": 1,
                "start_s": 2.0, "end_s": 5.0, "start_frame": 50, "end_frame": 125,
                "duration_s": 3.0, "def_line_x": 20.0, "compactness_m": 14.0,
                "n_within_10m": 0, "flank": "right", "n_attackers": 5, "n_defenders": 2,
                "in_possession_share": 1.0, "n_pressers": 0, "response": "",
            },
            {
                "label": "build_up", "team": "away", "period": 1,
                "start_s": 1.0, "end_s": 5.0, "start_frame": 25, "end_frame": 125,
                "duration_s": 4.0, "def_line_x": -21.0, "compactness_m": 16.0,
                "n_within_10m": 0, "flank": "", "n_attackers": 0, "n_defenders": 0,
                "in_possession_share": 1.0, "n_pressers": 3, "response": "",
            },
        ],
        marks_sequences={
            "low_block-home-1-38": {"verdict": "right", "note": "Compact 4-4-2 box block."},
            "flank_overload-home-1-50": {"verdict": "wrong", "note": "Injury stoppage."},
            "build_up-away-1-25": {"verdict": "right", "note": "Clean CB progression."},
        },
        compute_seconds=5400.0,
        with_tracks_and_video=True,
    )

    # Fixture 2: HANOI is HOME vs PVFCAND (away!)
    _make_match_dir(
        tmp_path,
        "m2_hanoi_pvf",
        match_id="hanoi_pvf",
        home="HANOI",
        away="PVFCAND",
        seq_rows=[
            {
                "label": "low_block", "team": "away", "period": 2,
                "start_s": 100.0, "end_s": 112.0, "start_frame": 2500, "end_frame": 2800,
                "duration_s": 12.0, "def_line_x": -38.0, "compactness_m": 9.5,
                "n_within_10m": 6, "flank": "", "n_attackers": 0, "n_defenders": 0,
                "in_possession_share": 0.0, "n_pressers": 6, "response": "",
            },
            {
                "label": "midfield_loss", "team": "away", "period": 2,
                "start_s": 200.0, "end_s": 206.0, "start_frame": 5000, "end_frame": 5150,
                "duration_s": 6.0, "def_line_x": -10.0, "compactness_m": 12.0,
                "n_within_10m": 3, "flank": "", "n_attackers": 0, "n_defenders": 0,
                "in_possession_share": 0.0, "n_pressers": 0,
                "ball_dist_m": 4.8, "retreat_mps": -1.5, "response": "counter_press",
            },
        ],
        marks_sequences={
            "low_block-away-2-2500": {"verdict": "right", "note": "Deep away block at Hang Day."},
        },
        compute_seconds=3600.0,
        with_tracks_and_video=False,
    )
    return tmp_path


def test_format_clock():
    assert format_clock(0) == "00:00"
    assert format_clock(175.4) == "02:55"
    assert format_clock(3665.0) == "1:01:05"


def test_query_club_filter_aggregates_home_and_away_fixtures(two_fixture_root: Path):
    df = search_sequences(two_fixture_root, club="pvfcand")
    assert len(df) == 4
    assert set(df["team"]) == {"home", "away"}
    assert (df["club"] == "PVFCAND").all()
    assert set(df["opponent"]) == {"HANOI"}

    agg = summarize_sequences(two_fixture_root, club="PVFCAND", by_match=False)
    lb = agg[agg["label"] == "low_block"].iloc[0]
    assert lb["count"] == 2
    assert lb["total_duration_s"] == pytest.approx(15.0)


def test_export_pack_filters_marks_writes_xml_dossier_and_cuts_radar_clips(
    tmp_path: Path, two_fixture_root: Path
):
    pack_dir = tmp_path / "pvf_pack"
    manifest = export_pack(
        [two_fixture_root / "m1_pvf_hanoi", two_fixture_root / "m2_hanoi_pvf"],
        pack_dir,
        club="PVFCAND",
        verified_only=False,  # excludes 'wrong' (#50), keeps 'right' (#38, #2500) + unmarked (#5000)
        export_clips=True,
        pre_roll_s=1.0,
        post_roll_s=1.0,
        gpu_hourly_usd=1.0,
        operator_minutes=12.0,
    )
    econ = manifest["economics"]
    assert econ["matches_count"] == 2
    assert econ["sequences_exported"] == 3
    assert econ["verified_right_sequences"] == 2
    assert econ["clips_exported"] == 1  # m1 has video + tracks; m2 has no video
    assert econ["total_compute_hours"] == pytest.approx(2.5)
    assert econ["estimated_compute_usd"] == pytest.approx(2.5)

    playlist = pd.read_csv(pack_dir / "playlist.csv")
    assert len(playlist) == 3
    assert "flank_overload-home-1-50" not in set(playlist["sequence_id"])
    mp4_rows = playlist[playlist["clip_path"].notna() & (playlist["clip_path"] != "")]
    assert len(mp4_rows) == 1
    mp4_file = pack_dir / mp4_rows.iloc[0]["clip_path"]
    assert mp4_file.is_file() and mp4_file.stat().st_size > 1000

    # Sportscode XML parses cleanly and has 3 instances
    tree = ET.parse(pack_dir / "sportscode.xml")
    instances = tree.getroot().findall("./ALL_INSTANCES/instance")
    assert len(instances) == 3
    codes = [inst.findtext("code") for inst in instances]
    assert "PVFCAND - low_block" in codes

    md = (pack_dir / "dossier.md").read_text()
    html_doc = (pack_dir / "dossier.html").read_text()
    assert "PVFCAND" in md and "Khối phòng ngự thấp" in md
    assert "<html" in html_doc and "MP4 + Radar" in html_doc


def test_cli_pack_export_verified_only(tmp_path: Path, two_fixture_root: Path):
    runner = CliRunner()
    out_dir = tmp_path / "cli_verified_pack"
    res = runner.invoke(
        app,
        [
            "pack", "export",
            str(two_fixture_root / "m1_pvf_hanoi"),
            str(two_fixture_root / "m2_hanoi_pvf"),
            "--out", str(out_dir),
            "--club", "PVFCAND",
            "--verified-only",
            "--no-clips",
        ],
    )
    assert res.exit_code == 0, res.output
    summary = json.loads(res.stdout)
    assert summary["sequences_exported"] == 2
    assert summary["verified_right_sequences"] == 2
    assert summary["clips_exported"] == 0

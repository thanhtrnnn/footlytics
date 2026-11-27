"""Tests for per-club tactical threshold profiles, live workspace recompute, and static web pack export."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from footlytics.analytics.profiles import (
    BUILTIN_PROFILES,
    TacticalProfile,
    load_profile,
    recompute_match_sequences,
    save_profile,
)
from footlytics.analytics.static_site import build_static_pack_site
from footlytics.state.schema import MatchMeta, MatchState, Role, Team, TeamInfo
from footlytics.workspace_server import make_workspace_server


def _synthetic_match(match_dir: Path, match_id: str = "club_a_vs_b", home: str = "HANOI", away: str = "PVFCAND") -> Path:
    """Create a minimal valid MatchState with a clear deep low-block stretch for AWAY."""
    fps = 10.0
    n_frames = 120  # 12.0s
    rows: list[dict] = []
    # Home attacks +x (GK at -45m), Away attacks -x (GK at +46m, defending compactly at x=+30..+36m).
    home_xs = [-10.0, 5.0, 12.0, 18.0, 22.0, 25.0]
    home_ys = [-12.0, 8.0, -5.0, 10.0, -8.0, 0.0]
    away_xs = [28.0, 29.0, 30.0, 31.0, 32.0, 34.0]
    away_ys = [-10.0, -4.0, 2.0, 8.0, -6.0, 4.0]

    for f in range(n_frames):
        t = f / fps
        # Goalkeepers anchor normalise_direction unambiguously
        rows.append(
            {
                "frame_idx": f, "period": 1, "timestamp": t, "track_id": 100,
                "role": Role.GOALKEEPER.value, "team": Team.HOME.value, "jersey": pd.NA,
                "bbox_x": 20.0, "bbox_y": 100.0, "bbox_w": 20.0, "bbox_h": 50.0,
                "det_conf": 0.9, "x": -46.0, "y": 0.0, "z": 0.0, "speed": 0.5, "accel": 0.0,
                "interpolated": False, "live_wide": True, "shot": 1,
            }
        )
        rows.append(
            {
                "frame_idx": f, "period": 1, "timestamp": t, "track_id": 101,
                "role": Role.GOALKEEPER.value, "team": Team.AWAY.value, "jersey": pd.NA,
                "bbox_x": 420.0, "bbox_y": 100.0, "bbox_w": 20.0, "bbox_h": 50.0,
                "det_conf": 0.9, "x": 46.0, "y": 0.0, "z": 0.0, "speed": 0.5, "accel": 0.0,
                "interpolated": False, "live_wide": True, "shot": 1,
            }
        )
        for k, (x, y) in enumerate(zip(home_xs, home_ys), 1):
            rows.append(
                {
                    "frame_idx": f,
                    "period": 1,
                    "timestamp": t,
                    "track_id": k,
                    "role": Role.PLAYER.value,
                    "team": Team.HOME.value,
                    "jersey": pd.NA,
                    "bbox_x": 100.0,
                    "bbox_y": 100.0,
                    "bbox_w": 20.0,
                    "bbox_h": 50.0,
                    "det_conf": 0.9,
                    "x": x + 0.8 * np.sin(f * 0.18),
                    "y": y + 2.4 * np.cos(f * 0.18),
                    "z": 0.0,
                    "speed": 2.0,
                    "accel": 0.0,
                    "interpolated": False,
                    "live_wide": True,
                    "shot": 1,
                }
            )
        for k, (x, y) in enumerate(zip(away_xs, away_ys), 20):
            rows.append(
                {
                    "frame_idx": f,
                    "period": 1,
                    "timestamp": t,
                    "track_id": k,
                    "role": Role.PLAYER.value,
                    "team": Team.AWAY.value,
                    "jersey": pd.NA,
                    "bbox_x": 200.0,
                    "bbox_y": 100.0,
                    "bbox_w": 20.0,
                    "bbox_h": 50.0,
                    "det_conf": 0.9,
                    "x": x + 0.6 * np.cos(f * 0.18),
                    "y": y + 2.4 * np.sin(f * 0.18),
                    "z": 0.0,
                    "speed": 1.8,
                    "accel": 0.0,
                    "interpolated": False,
                    "live_wide": True,
                    "shot": 1,
                }
            )
        # Active lateral ball circulation by HOME in front of AWAY low block
        rows.append(
            {
                "frame_idx": f,
                "period": 1,
                "timestamp": t,
                "track_id": 999,
                "role": Role.BALL.value,
                "team": Team.UNKNOWN.value,
                "jersey": pd.NA,
                "bbox_x": 150.0,
                "bbox_y": 120.0,
                "bbox_w": 8.0,
                "bbox_h": 8.0,
                "det_conf": 0.9,
                "x": 23.0 + 3.0 * np.sin(f * 0.12),
                "y": -14.0 + 28.0 * ((f % 40) / 40.0),
                "z": 0.0,
                "speed": 6.0,
                "accel": 0.0,
                "interpolated": False,
                "live_wide": True,
                "shot": 1,
            }
        )

    meta = MatchMeta(
        match_id=match_id,
        fps=fps,
        source_type="broadcast",
        home=TeamInfo(home, home[:3]),
        away=TeamInfo(away, away[:3]),
    )
    st = MatchState(meta, pd.DataFrame(rows))
    match_dir.mkdir(parents=True, exist_ok=True)
    st.save(match_dir)
    poss = pd.DataFrame(
        {"frame_idx": range(n_frames), "period": 1, "timestamp": [f / fps for f in range(n_frames)], "ballStatus": "HOME"}
    )
    poss.to_csv(match_dir / "possession.csv", index=False)
    recompute_match_sequences(match_dir, BUILTIN_PROFILES["default"], write=True, state=st)
    return match_dir


def test_profile_yaml_roundtrip_and_validation(tmp_path: Path) -> None:
    prof = TacticalProfile(name="hanoi_fc", low_block_line=-24.0, high_press_n=4, counter_press_m=5.0)
    yp = save_profile(prof, tmp_path / "hanoi_fc.yaml")
    loaded = load_profile(yp)
    assert loaded == prof
    assert loaded.to_detect_kwargs()["high_press_n"] == 4

    jp = save_profile(prof, tmp_path / "hanoi_fc.json")
    assert load_profile(jp) == prof

    for preset_name in BUILTIN_PROFILES:
        assert load_profile(preset_name).name == preset_name

    with pytest.raises(ValueError, match="unknown tactical profile field"):
        TacticalProfile.from_dict({"not_a_real_param": 123})
    with pytest.raises(ValueError, match="low_block_line"):
        TacticalProfile(low_block_line=5.0)
    with pytest.raises(ValueError, match="retreat_mps"):
        TacticalProfile(retreat_mps=1.5)


def test_recompute_match_sequences_and_workspace_api(tmp_path: Path) -> None:
    mdir = _synthetic_match(tmp_path / "m1")
    default_df = pd.read_csv(mdir / "sequences.csv")
    assert "low_block" in set(default_df["label"])

    # Tightening low_block_line to -45.0 should eliminate the -30m low_block
    strict_prof = TacticalProfile(name="ultra_deep", low_block_line=-45.0)
    strict_df = recompute_match_sequences(mdir, strict_prof, write=False)
    assert "low_block" not in set(strict_df["label"])

    # Test live POST /api/recompute & GET /api/export/profile.yaml on workspace server
    srv = make_workspace_server(mdir, port=0)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        req = Request(
            f"http://127.0.0.1:{port}/api/recompute",
            data=json.dumps({"persist": False, "profile": {"name": "ultra_deep", "low_block_line": -45.0}}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            snap = json.loads(resp.read())
        assert snap["profile"]["name"] == "ultra_deep"
        assert snap["profile"]["low_block_line"] == -45.0
        assert all(s["label"] != "low_block" for s in snap["sequences"])

        with urlopen(f"http://127.0.0.1:{port}/api/export/profile.yaml", timeout=5) as resp:
            assert resp.status == 200
            yml_text = resp.read().decode("utf-8")
        assert "ultra_deep" in yml_text
        assert "low_block_line: -45.0" in yml_text
    finally:
        srv.shutdown()
        srv.server_close()


def test_build_static_pack_site_end_to_end(tmp_path: Path) -> None:
    mdir = _synthetic_match(tmp_path / "m1")
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "dossier.html").write_text("<html><body>Dossier</body></html>", encoding="utf-8")
    clips_sub = pack_dir / "clips" / "03_low_block"
    clips_sub.mkdir(parents=True)
    dummy_mp4 = clips_sub / "001_demo.mp4"
    dummy_mp4.write_bytes(b"ftypisom")
    seqs = pd.read_csv(mdir / "sequences.csv")
    assert len(seqs) >= 1
    first = seqs.iloc[0]
    pd.DataFrame(
        [
            {
                "clip_index": 1,
                "sequence_id": f"{first['label']}-{first['team']}-{int(first['period'])}-{int(first['start_frame'])}",
                "match_id": "club_a_vs_b",
                "label": first["label"],
                "period": int(first["period"]),
                "start_s": float(first["start_s"]),
                "clip_path": "clips/03_low_block/001_demo.mp4",
            }
        ]
    ).to_csv(pack_dir / "playlist.csv", index=False)

    out_site = tmp_path / "static_site"
    info = build_static_pack_site(
        mdir,
        out_site,
        pack_dir=pack_dir,
        club="PVFCAND",
        title="PVF-CAND Pilot Pack",
        passcode="vleague2026",
        radar_hz=10.0,
    )
    assert info["sequences_count"] >= 1
    assert info["clips_count"] == 1
    assert info["radar_files_count"] == info["sequences_count"]
    assert info["passcode_gate"] == "client-side, not access control"
    assert "passcode_protected" not in info
    manifest = json.loads((out_site / "site_manifest.json").read_text())
    assert manifest["passcode_gate"] == "client-side, not access control"
    assert manifest["access_control"].startswith("none")

    index_html = (out_site / "index.html").read_text(encoding="utf-8")
    assert '"static_mode":true' in index_html
    assert "không phải kiểm soát truy cập" in index_html
    assert "PVF-CAND Pilot Pack" in index_html
    assert '"dossier_url":"dossier.html"' in index_html
    assert (out_site / "dossier.html").is_file()
    assert (out_site / "clips" / "03_low_block" / "001_demo.mp4").is_file()
    assert (out_site / "radar" / "seq_0001.json").is_file()
    assert "X-Robots-Tag: noindex, nofollow" in (out_site / "_headers").read_text()

    rad = json.loads((out_site / "radar" / "seq_0001.json").read_text())
    assert rad["match_id"] == "club_a_vs_b"
    assert len(rad["frames"]) > 0


def test_passcode_help_says_it_is_not_access_control():
    from typer.testing import CliRunner

    from footlytics.cli import app

    for cmd in (["pack", "build-site"], ["pack", "export"]):
        out = " ".join(CliRunner().invoke(app, [*cmd, "--help"]).output.replace("│", " ").split())
        assert "NOT access control" in out, cmd

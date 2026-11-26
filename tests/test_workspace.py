"""Tests for multi-match Web Tactical Query & 2D Radar Workspace (`footlytics workspace serve`)."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from footlytics.analytics.pack import export_pack
from footlytics.workspace_server import make_workspace_server
from tests.test_pack import _make_match_dir


@pytest.fixture
def workspace_env(tmp_path: Path):
    matches_root = tmp_path / "matches"
    m1 = _make_match_dir(
        matches_root,
        "m1_pvf_hanoi",
        match_id="pvf_hanoi",
        home="PVFCAND",
        away="HANOI",
        seq_rows=[
            {
                "label": "low_block", "team": "home", "period": 1,
                "start_s": 1.0, "end_s": 4.0, "start_frame": 25, "end_frame": 100,
                "duration_s": 3.0, "def_line_x": -36.0, "compactness_m": 10.2,
                "n_within_10m": 4, "flank": "", "n_attackers": 0, "n_defenders": 0,
                "in_possession_share": 0.0, "n_pressers": 5, "response": "",
            },
            {
                "label": "build_up", "team": "away", "period": 1,
                "start_s": 1.5, "end_s": 4.5, "start_frame": 38, "end_frame": 112,
                "duration_s": 3.0, "def_line_x": -20.0, "compactness_m": 15.0,
                "n_within_10m": 1, "flank": "", "n_attackers": 0, "n_defenders": 0,
                "in_possession_share": 1.0, "n_pressers": 4, "response": "",
            },
        ],
        with_tracks_and_video=True,
    )
    m2 = _make_match_dir(
        matches_root,
        "m2_hanoi_pvf",
        match_id="hanoi_pvf",
        home="HANOI",
        away="PVFCAND",
        seq_rows=[
            {
                "label": "midfield_loss", "team": "away", "period": 2,
                "start_s": 3500.0, "end_s": 3506.0, "start_frame": 87500, "end_frame": 87650,
                "duration_s": 6.0, "def_line_x": -12.0, "compactness_m": 11.5,
                "n_within_10m": 3, "flank": "", "n_attackers": 0, "n_defenders": 0,
                "in_possession_share": 0.0, "n_pressers": 0,
                "ball_dist_m": 4.2, "retreat_mps": -1.4, "response": "counter_press",
            },
        ],
        with_tracks_and_video=False,
    )
    meta2 = json.loads((m2 / "meta.json").read_text())
    meta2["provenance"] = {"period_start_s": {"1": 145.0, "2": 3160.0}}
    (m2 / "meta.json").write_text(json.dumps(meta2))

    pack_dir = tmp_path / "packs" / "pvf_pack"
    export_pack([m1, m2], pack_dir, club="PVFCAND", export_clips=True, pre_roll_s=0.5, post_roll_s=0.5)
    return {"matches_root": matches_root, "m1": m1, "m2": m2, "pack_dir": pack_dir}


def test_workspace_server_queries_radar_clip_range_verdict_and_exports(workspace_env):
    srv = make_workspace_server(
        workspace_env["matches_root"],
        pack_dirs=[workspace_env["pack_dir"]],
        reviewer="coach_hanoi",
        port=0,
    )
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        # 1. HTML index renders with embedded snapshot
        html = urllib.request.urlopen(f"{base}/").read().decode("utf-8")
        assert "FOOTLYTICS" in html and "pvf_hanoi" in html and "50:40-50:46" in html

        # 2. JSON state lists both matches and clubs
        snap = json.loads(urllib.request.urlopen(f"{base}/api/state").read().decode("utf-8"))
        assert len(snap["sequences"]) == 3
        assert set(snap["clubs"]) == {"HANOI", "PVFCAND"}
        lb_seq = next(s for s in snap["sequences"] if s["label"] == "low_block")
        assert lb_seq["has_clip"] is True

        # 3. 2D Radar trajectory slice returns frames with players and ball
        radar = json.loads(
            urllib.request.urlopen(
                f"{base}/api/radar?match_id=pvf_hanoi&start_s=1.0&end_s=4.0"
            ).read().decode("utf-8")
        )
        assert radar["home"] == "PVFCAND" and len(radar["frames"]) > 5
        assert radar["frames"][0]["b"] is not None and len(radar["frames"][0]["p"]) == 2

        # 4. Clip Range request returns HTTP 206 Partial Content
        req_rng = urllib.request.Request(
            f"{base}{lb_seq['clip_url']}", headers={"Range": "bytes=0-127"}
        )
        with urllib.request.urlopen(req_rng) as resp:
            assert resp.status == 206
            assert len(resp.read()) == 128

        # 5. Live verdict + note persists atomically to marks.json
        uid = "hanoi_pvf::midfield_loss-away-2-87500"
        post_req = urllib.request.Request(
            f"{base}/api/verdict",
            data=json.dumps({"uid": uid, "verdict": "right", "note": "Good counter-press trigger"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        saved = json.loads(urllib.request.urlopen(post_req).read().decode("utf-8"))
        assert saved["verdict"] == "right"
        marks_doc = json.loads((workspace_env["m2"] / "marks.json").read_text())
        assert marks_doc["sequences"]["midfield_loss-away-2-87500"]["verdict"] == "right"

        # 6. Filtered Sportscode XML & CSV exports honor club + threshold query params
        xml_bytes = urllib.request.urlopen(
            f"{base}/api/export/sportscode.xml?club=PVFCAND&max_def_line=-30"
        ).read()
        insts = ET.fromstring(xml_bytes).findall("./ALL_INSTANCES/instance")
        assert len(insts) == 1
        assert insts[0].findtext("code") == "PVFCAND - low_block"

        csv_txt = urllib.request.urlopen(
            f"{base}/api/export/playlist.csv?club=HANOI&min_pressers=3"
        ).read().decode("utf-8")
        assert "build_up-away-1-38" in csv_txt
    finally:
        srv.shutdown()
        srv.server_close()


def test_workspace_server_enforces_token_auth_and_sets_session_cookie(workspace_env):
    srv = make_workspace_server(
        workspace_env["matches_root"],
        port=0,
        token="club-pilot-secret",
    )
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{base}/api/state")
        assert exc.value.code == 401

        # Query token grants access and sets HttpOnly cookie
        with urllib.request.urlopen(f"{base}/?token=club-pilot-secret") as resp:
            assert resp.status == 200
            cookie_hdr = resp.headers.get("Set-Cookie", "")
            assert "footlytics_ws=" in cookie_hdr and "HttpOnly" in cookie_hdr
            cookie_pair = cookie_hdr.split(";", 1)[0]

        # Subsequent API call succeeds using the session cookie alone
        req_cookie = urllib.request.Request(f"{base}/api/state", headers={"Cookie": cookie_pair})
        with urllib.request.urlopen(req_cookie) as resp2:
            assert resp2.status == 200
    finally:
        srv.shutdown()
        srv.server_close()

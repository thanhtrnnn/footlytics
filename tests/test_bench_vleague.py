"""V.League mini-GT: frame sampling, splits and the manifest."""
import json

import cv2
import numpy as np

from footlytics.bench import vleague


def test_match_id_from_the_three_real_file_names():
    f = vleague.match_id_from_name
    assert f("CÔNG AN TP HỒ CHÍ MINH - HÀ NỘI - LPBANK V.LEAGUE 1 - 2026-27.mp4") == \
        "cong-an-tp-ho-chi-minh-ha-noi"
    assert f("SHB ĐÀ NẴNG - HẢI PHÒNG - LPBANK V.LEAGUE 1 - 2025-26.mp4") == \
        "shb-da-nang-hai-phong"
    assert f("NINH BÌNH - PVF-CAND - LPBANK V.LEAGUE 1 - 2025-26.mp4") == "ninh-binh-pvf-cand"


def test_choose_frames_is_deterministic_and_spread():
    n, fps = 60_000, 50
    step = 30 * fps
    cands = list(range(vleague.WINDOW, n - vleague.WINDOW, step))
    assert len(cands) == 40 and cands[1] == 1512
    reject = {c for i, c in enumerate(cands) if i % 3 == 2}

    def is_pitch(i):
        return i not in reject

    got = vleague.choose_frames(n, fps, is_pitch)
    assert len(got) == 15 and got == sorted(got) and len(set(got)) == 15
    assert all(vleague.WINDOW <= i < n - vleague.WINDOW for i in got)
    assert not reject & set(got)
    assert got[0] == 12 and got[-1] == 58512
    assert vleague.choose_frames(n, fps, is_pitch) == got


def test_choose_frames_returns_fewer_when_few_candidates_pass():
    got = vleague.choose_frames(10_000, 50, lambda i: i < 3000)
    assert got == [12, 1512]


def test_assign_splits_one_match():
    keys = [f"m/{i:07d}" for i in range(15)]
    out = vleague.assign_splits(keys)
    assert sum(v["split"] == "dev" for v in out.values()) == 8
    assert sum(v["split"] == "frozen" for v in out.values()) == 7
    assert sum(v["relabel"] for v in out.values()) == 4
    assert out["m/0000000"] == {"split": "dev", "relabel": False}
    assert out["m/0000001"] == {"split": "frozen", "relabel": True}


def test_assign_splits_ranks_within_each_match_and_sorts_by_frame():
    keys = [f"b/{i:07d}" for i in (900, 100, 500)] + [f"a/{i:07d}" for i in (7, 3)]
    out = vleague.assign_splits(keys)
    assert out["a/0000003"]["split"] == "dev"
    assert out["a/0000007"] == {"split": "frozen", "relabel": True}
    assert [out[f"b/{i:07d}"]["split"] for i in (100, 500, 900)] == ["dev", "frozen", "dev"]


def test_pitch_share_green_vs_grey():
    green = np.zeros((200, 300, 3), np.uint8)
    green[:] = (40, 150, 40)
    grey = np.full((200, 300, 3), 128, np.uint8)
    assert vleague.pitch_share(green) > 0.8
    assert vleague.pitch_share(grey) == 0.0


def _write_video(path, n=40, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    for i in range(n):
        frame = np.full((size[1], size[0], 3), 30 + i, np.uint8)
        w.write(frame)
    w.release()


def test_extract_windows_writes_25_files_and_never_rewrites(tmp_path):
    video = tmp_path / "t.avi"
    _write_video(video)
    out = tmp_path / "out"
    vleague.extract_windows(video, [12, 20], out)
    for idx in (12, 20):
        files = {p.name for p in (out / f"{idx:07d}").iterdir()}
        assert len(files) == 25
        assert {"-12.jpg", "-01.jpg", "+00.jpg", "+01.jpg", "+12.jpg"} <= files
    img = cv2.imread(str(out / "0000012" / "+00.jpg"))
    assert img.shape == (1080, 1920, 3)
    before = {p: p.stat().st_mtime_ns for p in out.rglob("*.jpg")}
    assert len(before) == 50
    vleague.extract_windows(video, [12, 20], out)
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*.jpg")} == before


def test_sample_match_and_manifest(tmp_path, monkeypatch):
    video = tmp_path / "SHB ĐÀ NẴNG - HẢI PHÒNG - LPBANK V.LEAGUE 1 - 2025-26.avi"
    _write_video(video, n=200)
    monkeypatch.setattr(vleague, "make_is_pitch", lambda cap: (lambda i: True))
    entry = vleague.sample_match(video, "shb-da-nang-hai-phong", tmp_path / "data",
                                 k=3, step_s=5.0)
    assert entry["video"] == video.name and entry["fps"] == 10.0
    assert len(entry["sha256"]) == 64
    assert entry["frames"] == {"0000012": {"split": "dev", "relabel": False},
                               "0000112": {"split": "frozen", "relabel": True},
                               "0000162": {"split": "dev", "relabel": False}}
    assert (tmp_path / "data" / "shb-da-nang-hai-phong" / "0000162" / "+12.jpg").exists()
    mpath = tmp_path / "m" / "manifest.json"
    vleague.write_manifest({"shb-da-nang-hai-phong": entry}, mpath)
    m = json.loads(mpath.read_text())
    assert m["version"] == 1 and m["size"] == [1920, 1080] and m["window"] == 12
    assert m["matches"]["shb-da-nang-hai-phong"] == entry


def test_vleague_sample_cli(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from footlytics.bench import cli

    video = tmp_path / "NINH BÌNH - PVF-CAND - LPBANK V.LEAGUE 1 - 2025-26.avi"
    _write_video(video, n=60)
    monkeypatch.setattr(vleague, "make_is_pitch", lambda cap: (lambda i: True))
    mpath = tmp_path / "m.json"
    r = CliRunner().invoke(cli.bench_app, ["vleague-sample", str(video), "--data",
                                           str(tmp_path / "d"), "--manifest", str(mpath)])
    assert r.exit_code == 0, r.output
    m = json.loads(mpath.read_text())
    assert list(m["matches"]) == ["ninh-binh-pvf-cand"]
    assert (tmp_path / "d" / "ninh-binh-pvf-cand" / "0000012" / "-12.jpg").exists()
    r = CliRunner().invoke(cli.bench_app, ["vleague-sample", str(tmp_path / "none.mp4")])
    assert r.exit_code != 0

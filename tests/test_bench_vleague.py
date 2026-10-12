"""V.League mini-GT: frame sampling, splits and the manifest."""
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

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
    monkeypatch.setattr(vleague, "PER_MATCH", 1)
    mpath = tmp_path / "m.json"
    r = CliRunner().invoke(cli.bench_app, ["vleague-sample", str(video), "--data",
                                           str(tmp_path / "d"), "--manifest", str(mpath)])
    assert r.exit_code == 0, r.output
    m = json.loads(mpath.read_text())
    assert list(m["matches"]) == ["ninh-binh-pvf-cand"]
    assert (tmp_path / "d" / "ninh-binh-pvf-cand" / "0000012" / "-12.jpg").exists()
    r = CliRunner().invoke(cli.bench_app, ["vleague-sample", str(tmp_path / "none.mp4")])
    assert r.exit_code != 0


# ---- review round 1 -------------------------------------------------------------------

import pytest


def _no_hash(path):
    raise AssertionError("hashed a video that failed validation")


def test_sample_match_rejects_an_unreadable_video_before_hashing(tmp_path, monkeypatch):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video at all")
    monkeypatch.setattr(vleague, "_sha256", _no_hash)
    with pytest.raises(ValueError, match="broken.mp4"):
        vleague.sample_match(bad, "x", tmp_path / "d")


def test_sample_match_rejects_a_video_shorter_than_two_windows(tmp_path, monkeypatch):
    short = tmp_path / "short.avi"
    _write_video(short, n=20)
    monkeypatch.setattr(vleague, "_sha256", _no_hash)
    with pytest.raises(ValueError, match="short.avi"):
        vleague.sample_match(short, "x", tmp_path / "d")


def test_sample_match_raises_when_too_few_frames_pass(tmp_path, monkeypatch):
    video = tmp_path / "v.avi"
    _write_video(video, n=200)
    monkeypatch.setattr(vleague, "make_is_pitch", lambda cap: (lambda i: i < 100))
    monkeypatch.setattr(vleague, "_sha256", _no_hash)
    with pytest.raises(ValueError, match=r"v.avi.*2 of 3"):
        vleague.sample_match(video, "x", tmp_path / "d", k=3, step_s=5.0)


def test_extract_windows_raises_when_imwrite_fails_and_leaves_no_file(tmp_path, monkeypatch):
    video = tmp_path / "t.avi"
    _write_video(video)
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="-12.jpg"):
        vleague.extract_windows(video, [12], tmp_path / "out")
    assert list((tmp_path / "out").rglob("*")) == [tmp_path / "out" / "0000012"]


def test_extract_windows_leaves_no_temp_files(tmp_path):
    video = tmp_path / "t.avi"
    _write_video(video)
    vleague.extract_windows(video, [12], tmp_path / "out")
    names = [p.name for p in (tmp_path / "out" / "0000012").iterdir()]
    assert len(names) == 25 and not [n for n in names if n.startswith(".tmp")]


def _cli(args):
    from typer.testing import CliRunner

    from footlytics.bench import cli

    return CliRunner().invoke(cli.bench_app, ["vleague-sample", *map(str, args)])


def test_cli_checks_every_path_before_any_work(tmp_path, monkeypatch):
    good = tmp_path / "A - B - LPBANK V.LEAGUE 1 - 2025-26.avi"
    _write_video(good, n=60)

    def boom(*a, **k):
        raise AssertionError("did work before validating paths")
    monkeypatch.setattr(vleague, "sample_match", boom)
    r = _cli([good, tmp_path / "missing.mp4", "--data", tmp_path / "d",
              "--manifest", tmp_path / "m.json"])
    assert r.exit_code != 0 and "missing.mp4" in r.output
    assert not (tmp_path / "m.json").exists()


def test_cli_unreadable_video_exits_nonzero_and_writes_no_manifest(tmp_path):
    bad = tmp_path / "A - B - LPBANK V.LEAGUE 1 - 2025-26.mp4"
    bad.write_bytes(b"junk")
    r = _cli([bad, "--data", tmp_path / "d", "--manifest", tmp_path / "m.json"])
    assert r.exit_code != 0 and "cannot open" in r.output
    assert not (tmp_path / "m.json").exists()


def test_cli_duplicate_match_ids_are_refused(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    name = "A - B - LPBANK V.LEAGUE 1 - 2025-26.avi"
    _write_video(tmp_path / "x" / name, n=60)
    _write_video(tmp_path / "y" / name, n=60)
    r = _cli([tmp_path / "x" / name, tmp_path / "y" / name, "--data", tmp_path / "d",
              "--manifest", tmp_path / "m.json"])
    assert r.exit_code != 0 and "a-b" in r.output
    assert not (tmp_path / "m.json").exists()


def test_cli_merges_into_an_existing_manifest(tmp_path, monkeypatch):
    mpath = tmp_path / "m.json"
    other = {"video": "o.mp4", "sha256": "0" * 64, "fps": 25.0,
             "frames": {"0000012": {"split": "dev", "relabel": False}}}
    vleague.write_manifest({"other-match": other}, mpath)
    video = tmp_path / "A - B - LPBANK V.LEAGUE 1 - 2025-26.avi"
    _write_video(video, n=60)
    monkeypatch.setattr(vleague, "make_is_pitch", lambda cap: (lambda i: True))
    monkeypatch.setattr(vleague, "PER_MATCH", 1)
    r = _cli([video, "--data", tmp_path / "d", "--manifest", mpath])
    assert r.exit_code == 0, r.output
    m = json.loads(mpath.read_text())
    assert sorted(m["matches"]) == ["a-b", "other-match"]
    assert m["matches"]["other-match"] == other
    assert m["matches"]["a-b"]["frames"] == {"0000012": {"split": "dev", "relabel": False}}


def test_rerun_reuses_the_manifest_frames_even_if_the_pitch_test_changes(tmp_path, monkeypatch):
    video = tmp_path / "v.avi"
    _write_video(video, n=200)
    monkeypatch.setattr(vleague, "make_is_pitch", lambda cap: (lambda i: True))
    first = vleague.sample_match(video, "m", tmp_path / "d", k=3, step_s=5.0)
    assert list(first["frames"]) == ["0000012", "0000112", "0000162"]

    def boom(cap):
        raise AssertionError("re-ran frame choice")
    monkeypatch.setattr(vleague, "make_is_pitch", boom)
    again = vleague.sample_match(video, "m", tmp_path / "d", k=3, step_s=5.0, existing=first)
    assert again == first


def test_rerun_with_a_changed_video_raises(tmp_path, monkeypatch):
    video = tmp_path / "v.avi"
    _write_video(video, n=200)
    monkeypatch.setattr(vleague, "make_is_pitch", lambda cap: (lambda i: True))
    first = vleague.sample_match(video, "m", tmp_path / "d", k=3, step_s=5.0)
    _write_video(video, n=201)
    with pytest.raises(ValueError, match="video changed"):
        vleague.sample_match(video, "m", tmp_path / "d", k=3, step_s=5.0, existing=first)


# --- labelling server and label I/O ---------------------------------------------------

import threading
import urllib.error
import urllib.request

FAKE_JPEG = b"\xff\xd8\xff\xe0fake-centre-frame\xff\xd9"
KEY_A, KEY_B = "a/0000012", "a/0000112"


def _fake_manifest(tmp_path):
    data = tmp_path / "data"
    for idx in ("0000012", "0000112"):
        d = data / "a" / idx
        d.mkdir(parents=True)
        (d / "+00.jpg").write_bytes(FAKE_JPEG + idx.encode())
    (tmp_path / "secret.jpg").write_bytes(b"do not serve")
    entry = {"video": "a.mp4", "sha256": "0" * 64, "fps": 25.0,
             "frames": {"0000012": {"split": "dev", "relabel": False},
                        "0000112": {"split": "frozen", "relabel": True}}}
    mpath = tmp_path / "manifest.json"
    vleague.write_manifest({"a": entry}, mpath)
    return mpath, data


class Served:
    def __init__(self, tmp_path, relabel_only=False, labels="labels.json"):
        from footlytics.bench import vleague_label

        self.manifest, self.data = _fake_manifest(tmp_path) if not (
            tmp_path / "manifest.json").exists() else (tmp_path / "manifest.json", tmp_path / "data")
        self.labels = tmp_path / labels
        self.server = vleague_label.make_server(self.manifest, self.data, self.labels, "A", 1,
                                                relabel_only)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def post(self, body):
        req = urllib.request.Request(self.base + "/label", data=json.dumps(body).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


@pytest.fixture
def served(tmp_path):
    s = Served(tmp_path)
    yield s
    s.close()


def test_server_binds_to_loopback_only(served):
    assert served.server.server_address[0] == "127.0.0.1"


def test_post_a_point_is_on_disk_and_null_deletes_it(served):
    status, body = served.post({"frame": KEY_A, "landmark": "pen_area_LT_front",
                                "uv": [812.5, 403.0]})
    assert status == 200 and body["points"] == {"pen_area_LT_front": [812.5, 403.0]}
    on_disk = json.loads(served.labels.read_text())
    assert on_disk == {"version": 1, "annotator": "A", "session": 1, "image_size": [1920, 1080],
                       "frames": {KEY_A: {"pen_area_LT_front": [812.5, 403.0]}}}
    status, body = served.post({"frame": KEY_A, "landmark": "pen_area_LT_front", "uv": None})
    assert status == 200 and body["points"] == {}
    assert json.loads(served.labels.read_text())["frames"] == {}


def test_post_moves_a_point_and_keeps_the_others(served):
    served.post({"frame": KEY_A, "landmark": "corner_LT", "uv": [10, 20]})
    served.post({"frame": KEY_A, "landmark": "centre_spot", "uv": [960, 540]})
    served.post({"frame": KEY_B, "landmark": "corner_LT", "uv": [1, 2]})
    served.post({"frame": KEY_A, "landmark": "corner_LT", "uv": [30.5, 40.25]})
    frames = json.loads(served.labels.read_text())["frames"]
    assert frames == {KEY_A: {"corner_LT": [30.5, 40.25], "centre_spot": [960, 540]},
                      KEY_B: {"corner_LT": [1, 2]}}


def test_post_unknown_landmark_is_400_naming_it_and_file_unchanged(served):
    served.post({"frame": KEY_A, "landmark": "corner_LT", "uv": [10, 20]})
    before = served.labels.read_bytes()
    status, body = served.post({"frame": KEY_A, "landmark": "mid_field_banana", "uv": [5, 5]})
    assert status == 400 and "mid_field_banana" in body
    assert served.labels.read_bytes() == before


@pytest.mark.parametrize("uv", [[1920.5, 10], [10, 1081], [-0.5, 10], [10, -1], [1, 2, 3],
                                ["1", 2], [float("nan"), 3], [True, 3], "12", {"u": 1}])
def test_post_a_point_outside_the_image_or_malformed_is_400(served, uv):
    before = served.labels.read_bytes()
    status, _ = served.post({"frame": KEY_A, "landmark": "corner_LT", "uv": uv})
    assert status == 400
    assert served.labels.read_bytes() == before


def test_post_frame_not_in_the_manifest_or_bad_json_is_400(served):
    before = served.labels.read_bytes()
    assert served.post({"frame": "a/0000013", "landmark": "corner_LT", "uv": [1, 1]})[0] == 400
    assert served.post({"landmark": "corner_LT", "uv": [1, 1]})[0] == 400
    req = urllib.request.Request(served.base + "/label", data=b"{not json", method="POST",
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400
    assert served.labels.read_bytes() == before


def test_frames_are_served_only_for_manifest_keys(served):
    assert served.get("/frames/a/0000012.jpg") == (200, FAKE_JPEG + b"0000012")
    assert served.get("/frames/a/0000112.jpg") == (200, FAKE_JPEG + b"0000112")
    assert served.get("/frames/a/0000013.jpg")[0] == 404
    assert served.get("/frames/b/0000012.jpg")[0] == 404
    assert served.get("/frames/../../etc/passwd")[0] == 404
    assert served.get("/frames/a/../../secret.jpg")[0] == 404
    assert served.get("/frames/%2e%2e/secret.jpg")[0] == 404
    assert served.get("/frames/a/0000012/+01.jpg")[0] == 404
    assert served.get("/nothing")[0] == 404


def test_frames_list_has_split_relabel_and_point_counts(served):
    served.post({"frame": KEY_B, "landmark": "corner_LT", "uv": [1, 2]})
    served.post({"frame": KEY_B, "landmark": "corner_RT", "uv": [3, 4]})
    status, body = served.get("/frames-list")
    assert status == 200
    assert json.loads(body) == [
        {"key": KEY_A, "split": "dev", "relabel": False, "points": 0},
        {"key": KEY_B, "split": "frozen", "relabel": True, "points": 2}]


def test_relabel_only_lists_only_flagged_frames(tmp_path):
    s = Served(tmp_path, relabel_only=True)
    try:
        assert [f["key"] for f in json.loads(s.get("/frames-list")[1])] == [KEY_B]
    finally:
        s.close()


def test_get_labels_returns_the_file(served):
    served.post({"frame": KEY_A, "landmark": "centre_spot", "uv": [960, 540]})
    status, body = served.get("/labels")
    assert status == 200 and json.loads(body) == json.loads(served.labels.read_text())


def test_restart_on_the_same_label_file_keeps_every_point(tmp_path):
    s = Served(tmp_path)
    s.post({"frame": KEY_A, "landmark": "centre_spot", "uv": [960, 540]})
    s.post({"frame": KEY_B, "landmark": "goalpost_RB", "uv": [100.5, 900]})
    s.close()
    s2 = Served(tmp_path)
    try:
        assert json.loads(s2.get("/labels")[1])["frames"] == {
            KEY_A: {"centre_spot": [960, 540]}, KEY_B: {"goalpost_RB": [100.5, 900]}}
        assert json.loads(s2.get("/frames-list")[1])[1]["points"] == 1
    finally:
        s2.close()


def test_page_has_the_stable_ids_guide_line_and_injected_data(served):
    from footlytics.geometry.annotate import HELP, SUGGESTED
    from footlytics.geometry.pitch import DEFAULT_PITCH

    status, body = served.get("/")
    page = body.decode()
    assert status == 200
    for ident in ("canvas", "frames", "landmarks", "pitch", "status", "check"):
        assert f'id="{ident}"' in page
    assert "Click the centre of the line intersection. Goal posts: the base of the post on " \
           "the goal line. Label every clearly visible landmark, at least 3." in page
    assert "/*DATA*/null" not in page
    start = page.index("const DATA = ") + len("const DATA = ")
    data = json.JSONDecoder().raw_decode(page[start:])[0]
    lm = DEFAULT_PITCH.landmarks()
    assert data["names"][:len(SUGGESTED)] == SUGGESTED
    assert sorted(data["names"]) == sorted(lm) and len(data["names"]) == 35
    assert data["help"] == HELP and data["pitch"]["corner_LT"] == [-52.5, -34.0]
    assert data["size"] == [1920, 1080] and data["annotator"] == "A" and data["session"] == 1


def test_template_ships_next_to_the_server_module():
    from footlytics.bench import vleague_label

    assert (Path(vleague_label.__file__).with_name("vleague_label.html")).is_file()


def test_load_labels_accepts_what_the_server_wrote(served):
    served.post({"frame": KEY_A, "landmark": "pen_area_LT_front", "uv": [812.5, 403.0]})
    manifest = json.loads(served.manifest.read_text())
    got = vleague.load_labels(served.labels, manifest)
    assert got["frames"] == {KEY_A: {"pen_area_LT_front": [812.5, 403.0]}}
    assert got["annotator"] == "A" and got["session"] == 1


def _labels_doc(frames, **over):
    doc = {"version": 1, "annotator": "A", "session": 1, "image_size": [1920, 1080],
           "frames": frames}
    return {**doc, **over}


def _load(tmp_path, doc):
    mpath, _ = _fake_manifest(tmp_path) if not (tmp_path / "manifest.json").exists() else (
        tmp_path / "manifest.json", None)
    p = tmp_path / "l.json"
    p.write_text(json.dumps(doc))
    return vleague.load_labels(p, json.loads(mpath.read_text()))


def test_load_labels_rejections(tmp_path):
    with pytest.raises(ValueError, match="no_such_point"):
        _load(tmp_path, _labels_doc({KEY_A: {"no_such_point": [1, 1]}}))
    with pytest.raises(ValueError, match="1921"):
        _load(tmp_path, _labels_doc({KEY_A: {"corner_LT": [1921, 1]}}))
    with pytest.raises(ValueError, match="a/0000099"):
        _load(tmp_path, _labels_doc({"a/0000099": {"corner_LT": [1, 1]}}))
    with pytest.raises(ValueError, match="version"):
        _load(tmp_path, _labels_doc({}, version=2))
    with pytest.raises(ValueError, match="image_size"):
        _load(tmp_path, _labels_doc({}, image_size=[1280, 720]))
    assert _load(tmp_path, _labels_doc({}))["frames"] == {}


def test_save_label_validates_and_leaves_no_temp_files(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(json.dumps(_labels_doc({})))
    out = vleague.save_label(p, KEY_A, "corner_LT", [5, 6])
    assert out["frames"] == {KEY_A: {"corner_LT": [5, 6]}}
    with pytest.raises(ValueError, match="bogus"):
        vleague.save_label(p, KEY_A, "bogus", [5, 6])
    with pytest.raises(ValueError, match="outside"):
        vleague.save_label(p, KEY_A, "corner_LT", [5, 2000])
    assert json.loads(p.read_text())["frames"] == {KEY_A: {"corner_LT": [5, 6]}}
    assert sorted(x.name for x in tmp_path.iterdir()) == ["l.json"]


def test_make_server_refuses_a_label_file_of_another_annotator(tmp_path):
    from footlytics.bench import vleague_label

    mpath, data = _fake_manifest(tmp_path)
    (tmp_path / "l.json").write_text(json.dumps(_labels_doc({}, annotator="B")))
    with pytest.raises(ValueError, match="annotator"):
        vleague_label.make_server(mpath, data, tmp_path / "l.json", "A", 1, False)


def test_cli_vleague_label_creates_the_file_and_prints_the_url(tmp_path, monkeypatch):
    from http.server import ThreadingHTTPServer
    from typer.testing import CliRunner

    from footlytics.bench import cli

    mpath, data = _fake_manifest(tmp_path)

    def interrupt(self, *a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(ThreadingHTTPServer, "serve_forever", interrupt)
    labels = tmp_path / "sub" / "labels-A-s1.json"
    r = CliRunner().invoke(cli.bench_app, [
        "vleague-label", "--manifest", str(mpath), "--data", str(data), "--labels", str(labels),
        "--annotator", "A", "--session", "1", "--relabel-only", "--port", "0"])
    assert r.exit_code == 0, r.output
    assert "http://127.0.0.1:" in r.output
    assert json.loads(labels.read_text()) == _labels_doc({})


# --- fix round 1: pruning, huge numbers, failing writes, concurrency, request hardening -----

def test_deleting_the_last_point_removes_the_frame_entry_and_never_creates_one(served):
    served.post({"frame": KEY_A, "landmark": "corner_LT", "uv": [10, 20]})
    served.post({"frame": KEY_A, "landmark": "centre_spot", "uv": [30, 40]})
    served.post({"frame": KEY_A, "landmark": "corner_LT", "uv": None})
    assert json.loads(served.labels.read_text())["frames"] == {KEY_A: {"centre_spot": [30, 40]}}
    served.post({"frame": KEY_A, "landmark": "centre_spot", "uv": None})
    assert json.loads(served.labels.read_text())["frames"] == {}
    status, body = served.post({"frame": KEY_B, "landmark": "corner_LT", "uv": None})
    assert status == 200 and body["points"] == {}
    assert json.loads(served.labels.read_text())["frames"] == {}


def _raw_post(served, raw: bytes, ctype="application/json", host=None):
    headers = {"Content-Type": ctype} if ctype else {}
    if host:
        headers["Host"] = host
    req = urllib.request.Request(served.base + "/label", data=raw, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_huge_or_non_finite_numbers_are_400_not_a_dropped_connection(served):
    before = served.labels.read_bytes()
    body = ('{"frame": "%s", "landmark": "corner_LT", "uv": [1%s, 1]}' % (KEY_A, "0" * 3000))
    assert _raw_post(served, body.encode())[0] == 400
    for uv in ("[1e400, 1]", "[-1e400, 1]", "[NaN, 1]", "[Infinity, 1]", "[20000, 1]", "[1, -10001]"):
        raw = ('{"frame": "%s", "landmark": "corner_LT", "uv": %s}' % (KEY_A, uv)).encode()
        assert _raw_post(served, raw)[0] == 400, uv
    assert served.labels.read_bytes() == before


def test_load_labels_and_save_label_reject_a_huge_int_with_value_error(tmp_path):
    with pytest.raises(ValueError, match="corner_LT"):
        _load(tmp_path, _labels_doc({KEY_A: {"corner_LT": [10 ** 3000, 1]}}))
    p = tmp_path / "l.json"
    p.write_text(json.dumps(_labels_doc({})))
    with pytest.raises(ValueError, match="corner_LT"):
        vleague.save_label(p, KEY_A, "corner_LT", [10 ** 3000, 1])


def test_a_failing_write_keeps_the_old_file_intact_and_is_a_5xx(served, monkeypatch):
    served.post({"frame": KEY_A, "landmark": "corner_LT", "uv": [10, 20]})
    before = served.labels.read_bytes()

    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(vleague.os, "replace", boom)
    status, _ = served.post({"frame": KEY_A, "landmark": "centre_spot", "uv": [30, 40]})
    assert 500 <= status < 600
    monkeypatch.undo()
    assert served.labels.read_bytes() == before
    assert json.loads(served.labels.read_text())["frames"] == {KEY_A: {"corner_LT": [10, 20]}}
    assert sorted(x.name for x in served.labels.parent.iterdir() if x.name.startswith(".tmp")) == []
    assert served.post({"frame": KEY_A, "landmark": "centre_spot", "uv": [30, 40]})[0] == 200


def test_concurrent_posts_of_different_landmarks_all_land(served):
    from footlytics.geometry.pitch import DEFAULT_PITCH

    names = list(DEFAULT_PITCH.landmarks())[:24]
    results = {}

    def worker(i, name):
        results[name] = served.post({"frame": KEY_A if i % 2 else KEY_B, "landmark": name,
                                     "uv": [float(i), float(100 + i)]})[0]
    threads = [threading.Thread(target=worker, args=(i, n)) for i, n in enumerate(names)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(results.values()) == {200}
    frames = json.loads(served.labels.read_text())["frames"]
    expect = {KEY_A: {}, KEY_B: {}}
    for i, n in enumerate(names):
        expect[KEY_A if i % 2 else KEY_B][n] = [float(i), float(100 + i)]
    assert frames == expect


def test_post_label_requires_a_json_content_type(served):
    before = served.labels.read_bytes()
    raw = json.dumps({"frame": KEY_A, "landmark": "corner_LT", "uv": [1, 1]}).encode()
    assert _raw_post(served, raw, ctype="text/plain")[0] == 415
    assert _raw_post(served, raw, ctype=None)[0] == 415
    assert _raw_post(served, raw, ctype="application/x-www-form-urlencoded")[0] == 415
    assert served.labels.read_bytes() == before
    assert _raw_post(served, raw, ctype="application/json; charset=utf-8")[0] == 200


def test_requests_with_a_foreign_host_header_are_403(served):
    port = served.server.server_address[1]
    raw = json.dumps({"frame": KEY_A, "landmark": "corner_LT", "uv": [1, 1]}).encode()
    for host in ("evil.example", f"evil.example:{port}", f"127.0.0.1:{port + 1}", "127.0.0.1"):
        assert _raw_post(served, raw, host=host)[0] == 403, host
        req = urllib.request.Request(served.base + "/labels", headers={"Host": host})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 403, host
    assert json.loads(served.labels.read_text())["frames"] == {}
    for host in (f"127.0.0.1:{port}", f"localhost:{port}"):
        assert _raw_post(served, raw, host=host)[0] == 200, host
        req = urllib.request.Request(served.base + "/labels", headers={"Host": host})
        assert urllib.request.urlopen(req).status == 200


# ------------------------------------------------------------------ metrics

from footlytics.geometry.homography import apply_homography  # noqa: E402
from footlytics.geometry.pitch import DEFAULT_PITCH  # noqa: E402
from footlytics.geometry.sn_camera import ground_homography, make_camera  # noqa: E402

LANDMARKS = DEFAULT_PITCH.landmarks()
BROADCAST = make_camera(0, 80, 0, 1500, (0, 55, -15), 1920, 1080)
H_TRUE = np.linalg.inv(ground_homography(BROADCAST))
H_TRUE = H_TRUE / H_TRUE[2, 2]


def visible_points(cam=BROADCAST) -> dict[str, tuple[float, float]]:
    names = sorted(LANDMARKS)
    uv = apply_homography(ground_homography(cam), np.array([LANDMARKS[n] for n in names]))
    return {n: (float(u), float(v)) for n, (u, v) in zip(names, uv)
            if 0 <= u <= 1920 and 0 <= v <= 1080}


def test_landmark_errors_zero_with_the_true_homography():
    pts = visible_points()
    assert len(pts) >= 8
    errs = vleague.landmark_errors(H_TRUE, pts, LANDMARKS)
    assert set(errs) == set(pts)
    assert max(errs.values()) < 1e-6
    assert vleague.frame_error(H_TRUE, pts, LANDMARKS) < 1e-6


def test_landmark_errors_equal_the_shift_when_h_is_offset_by_one_metre():
    pts = visible_points()
    shift = np.array([[1, 0, 1.0], [0, 1, 0], [0, 0, 1]])
    errs = vleague.landmark_errors(shift @ H_TRUE, pts, LANDMARKS)
    assert all(abs(e - 1.0) < 1e-6 for e in errs.values())
    assert abs(vleague.frame_error(shift @ H_TRUE, pts, LANDMARKS) - 1.0) < 1e-6


def test_frame_error_is_the_median_not_the_mean():
    pts = visible_points()
    names = sorted(pts)
    bad = dict(pts)
    bad[names[0]] = (pts[names[0]][0] + 400, pts[names[0]][1])     # one wild click
    assert vleague.frame_error(H_TRUE, bad, LANDMARKS) < 1e-6
    assert vleague.landmark_errors(H_TRUE, bad, LANDMARKS)[names[0]] > 1.0


def test_frame_error_is_none_when_uncalibrated_or_unlabelled():
    assert vleague.frame_error(None, visible_points(), LANDMARKS) is None
    assert vleague.frame_error(H_TRUE, {}, LANDMARKS) is None


def noise_texture(seed=0, size=(1080, 1920)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size, dtype=np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 2.0)


def shift_h(dx: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, 0], [0, 0, 1.0]])


def pan_set(n=6, px=3):
    """Frame t is the texture moved right by px*t, so H_t = H_0 o shift(-px*t)."""
    base = noise_texture()
    grays = [np.roll(base, px * t, axis=1) for t in range(n)]
    Hs = [H_TRUE @ shift_h(-px * t) for t in range(n)]
    grid = np.array([[u, v] for u in range(400, 1500, 150) for v in range(300, 900, 150)], float)
    return Hs, grays, apply_homography(H_TRUE, grid), ["bridged"] * n


def test_image_motion_recovers_a_pure_pan():
    _, grays, _, _ = pan_set()
    M = vleague.image_motion(grays[0], grays[1])
    assert M is not None
    assert np.allclose(apply_homography(M, np.array([[960.0, 540.0]])), [[963.0, 540.0]], atol=0.2)


def test_image_motion_is_none_across_a_cut_or_without_texture():
    base = noise_texture(0)
    assert vleague.image_motion(base, noise_texture(1)) is None
    flat = np.full_like(base, 90)
    assert vleague.image_motion(flat, flat) is None


def test_jitter_of_a_pure_pan_is_below_five_centimetres():
    Hs, grays, P, sources = pan_set()
    out = vleague.jitter_m(Hs, grays, P, sources)
    assert out["pairs_switch"] == 0 and out["pairs_between"] == 5 and out["pairs_skipped"] == 0
    assert out["switch_m"] is None
    assert out["between_m"] < 0.05


def test_jitter_lands_a_perturbed_h_in_the_switch_bucket_only():
    Hs, grays, P, sources = pan_set()
    Hs[3] = shift_h(0.5) @ Hs[3]                       # 0.5 m in x, in pitch metres
    sources[3] = "keyframe"
    out = vleague.jitter_m(Hs, grays, P, sources)
    assert out["pairs_switch"] == 2 and out["pairs_between"] == 3 and out["pairs_skipped"] == 0
    assert out["switch_m"] > 0.1
    assert out["between_m"] < 0.05


def test_jitter_counts_a_cut_in_pairs_skipped_not_in_the_mean():
    Hs, grays, P, sources = pan_set()
    grays[3] = noise_texture(7)                        # the picture cuts to something else
    out = vleague.jitter_m(Hs, grays, P, sources)
    assert out["pairs_skipped"] == 2                   # (2,3) and (3,4)
    assert out["pairs_between"] == 3
    assert out["between_m"] < 0.05


def test_jitter_skips_pairs_with_an_uncalibrated_frame_and_gives_none_for_empty_buckets():
    Hs, grays, P, sources = pan_set(n=3)
    Hs[1] = None
    out = vleague.jitter_m(Hs, grays, P, sources)
    assert out == {"switch_m": None, "between_m": None, "pairs_switch": 0, "pairs_between": 0,
                   "pairs_skipped": 2}


def labels_from(points, dx=0.0):
    return {n: (u + dx, v) for n, (u, v) in points.items()}


def test_labelling_error_is_zero_for_identical_labels():
    pts = visible_points()
    assert vleague.labelling_error_m(pts, dict(pts), LANDMARKS, (1920, 1080)) < 1e-6


def test_labelling_error_of_a_two_pixel_shift_is_near_the_measured_value():
    pts = visible_points()
    err = vleague.labelling_error_m(pts, labels_from(pts, 2.0), LANDMARKS, (1920, 1080))
    assert 0.05 < err < 0.2                            # measured 0.095 m


def test_labelling_error_needs_four_shared_landmarks():
    pts = visible_points()
    spread = ["corner_LT", "corner_RT", "centre_spot", "pen_arc_LT"]    # in general position
    a = {n: pts[n] for n in [*spread, "halfway_T", "centre_circle_T"]}
    three = {n: pts[n] for n in [*spread[:3], "pen_area_LT_front", "pen_area_RT_front"]}
    assert len(set(a) & set(three)) == 3
    assert vleague.labelling_error_m(a, three, LANDMARKS, (1920, 1080)) is None
    four = {n: pts[n] for n in spread}                      # shares exactly the 4 spread points
    assert len(set(a) & set(four)) == 4
    assert vleague.labelling_error_m(a, four, LANDMARKS, (1920, 1080)) == pytest.approx(0, abs=1e-6)
    shifted = {n: (u + 2.0, v) for n, (u, v) in four.items()}
    err = vleague.labelling_error_m(a, shifted, LANDMARKS, (1920, 1080))
    assert err is not None and 0 < err < 1.0


WIDE = make_camera(0, 70, 0, 900, (0, 70, -25), 1920, 1080)
HALFWAY = ["halfway_T", "halfway_B", "centre_spot", "centre_circle_T", "centre_circle_B"]
PEN_BOX = ["pen_area_LT_goalline", "pen_area_LB_goalline", "pen_area_LT_front", "pen_area_LB_front"]


def test_labelling_error_is_none_for_collinear_halfway_line_landmarks():
    pts = visible_points(WIDE)
    a = {n: pts[n] for n in HALFWAY}
    assert len(a) == 5                                 # all five exactly projected, all in frame
    assert vleague.labelling_error_m(a, labels_from(a, 2.0), LANDMARKS, (1920, 1080)) is None
    four = {n: a[n] for n in HALFWAY[:4]}
    assert vleague.labelling_error_m(four, dict(four), LANDMARKS, (1920, 1080)) is None


def test_labelling_error_accepts_the_four_corners_of_a_penalty_box():
    pts = visible_points(WIDE)
    a = {n: pts[n] for n in PEN_BOX}
    assert len(a) == 4
    err = vleague.labelling_error_m(a, labels_from(a, 2.0), LANDMARKS, (1920, 1080))
    assert err is not None and 0 < err < 1.0
    assert vleague.labelling_error_m(a, dict(a), LANDMARKS, (1920, 1080)) == pytest.approx(0, abs=1e-6)


def test_labelling_error_is_none_for_duplicated_points():
    a = {n: (500.0, 400.0) for n in PEN_BOX}
    assert vleague.labelling_error_m(a, dict(a), LANDMARKS, (1920, 1080)) is None


def test_labelling_error_drops_a_nan_pixel_and_is_none_when_too_few_remain():
    pts = visible_points()
    bad = dict(pts)
    bad[sorted(pts)[0]] = (float("nan"), 100.0)
    ten = vleague.labelling_error_m(pts, bad, LANDMARKS, (1920, 1080))
    assert ten == pytest.approx(0, abs=1e-6)           # the NaN point is dropped, rest agree
    four = {n: pts[n] for n in sorted(pts)[:4]}
    four_bad = dict(four)
    four_bad[sorted(four)[0]] = (float("nan"), float("nan"))
    assert vleague.labelling_error_m(four, four_bad, LANDMARKS, (1920, 1080)) is None


def test_landmark_errors_never_return_non_finite_values():
    pts = visible_points()
    name = sorted(pts)[0]
    bad = dict(pts)
    bad[name] = (float("nan"), 5.0)
    errs = vleague.landmark_errors(H_TRUE, bad, LANDMARKS)
    assert name not in errs and len(errs) == len(pts) - 1
    assert all(np.isfinite(e) for e in errs.values())
    assert vleague.frame_error(H_TRUE, {name: (float("nan"), 5.0)}, LANDMARKS) is None
    horizon = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.0]])
    assert vleague.landmark_errors(horizon, pts, LANDMARKS) == {}
    assert vleague.frame_error(horizon, pts, LANDMARKS) is None


def test_jitter_skips_a_singular_h_without_raising():
    Hs, grays, P, sources = pan_set(n=4)
    Hs[2] = np.array([[1.0, 2, 3], [2, 4, 6], [1, 2, 3]])      # rank 1
    out = vleague.jitter_m(Hs, grays, P, sources)
    assert out["pairs_skipped"] == 2 and out["pairs_between"] == 1
    Hs[2] = np.zeros((3, 3))
    assert vleague.jitter_m(Hs, grays, P, sources)["pairs_skipped"] == 2
    Hs[2] = np.full((3, 3), np.nan)
    assert vleague.jitter_m(Hs, grays, P, sources)["pairs_skipped"] == 2


def test_jitter_skips_pitch_points_behind_the_camera_or_on_the_horizon():
    Hs, grays, P, sources = pan_set(n=2)
    assert vleague.jitter_m(Hs, grays, P, sources)["pairs_between"] == 1
    flipped = [-h for h in Hs]                           # same mapping, opposite projective sign
    assert vleague.jitter_m(flipped, grays, P, sources)["pairs_between"] == 1
    behind = np.vstack([P, [[0.0, 100.0]]])              # behind the camera at y = 55
    out = vleague.jitter_m(Hs, grays, behind, sources)
    assert out["pairs_skipped"] == 1 and out["between_m"] is None
    r = np.linalg.inv(Hs[0])[2]                          # pitch points with zero image depth
    on_plane = np.vstack([P, [[0.0, -r[2] / r[1]]]])
    out = vleague.jitter_m(Hs, grays, on_plane, sources)
    assert out["pairs_skipped"] == 1 and out["between_m"] is None


def test_jitter_does_not_cap_a_finite_but_badly_solved_keyframe():
    Hs, grays, P, sources = pan_set()
    Hs[3] = shift_h(5.0) @ Hs[3]                         # a keyframe 5 m off
    sources[3] = "keyframe"
    out = vleague.jitter_m(Hs, grays, P, sources)
    assert out["pairs_switch"] == 2 and out["pairs_skipped"] == 0
    assert out["switch_m"] > 1.0 and out["switch_m"] == pytest.approx(5.0, abs=0.1)


def test_labelling_error_is_none_for_three_or_four_halfway_points_plus_one_off_line():
    pts = visible_points(WIDE)
    for line in (HALFWAY[:3], HALFWAY[:4], HALFWAY):
        for off in ("corner_LT", "pen_area_LT_front"):
            a = {n: pts[n] for n in [*line, off]}
            assert len(a) == len(line) + 1
            assert vleague.labelling_error_m(a, labels_from(a, 2.0), LANDMARKS, (1920, 1080)) is None


def test_labelling_error_is_finite_once_the_same_set_gains_spread_points():
    pts = visible_points(WIDE)
    names = [*HALFWAY[:4], "corner_LT", "corner_RT", "pen_area_LT_front"]
    a = {n: pts[n] for n in names}
    err = vleague.labelling_error_m(a, labels_from(a, 2.0), LANDMARKS, (1920, 1080))
    assert err is not None and np.isfinite(err) and 0 < err < 0.5


def test_labelling_error_of_a_penalty_box_is_still_about_its_measured_value():
    pts = visible_points(WIDE)
    a = {n: pts[n] for n in PEN_BOX}
    err = vleague.labelling_error_m(a, labels_from(a, 2.0), LANDMARKS, (1920, 1080))
    assert err == pytest.approx(0.165, abs=0.02)


def test_labelling_error_is_none_for_nearly_collinear_clicks_on_a_line():
    pts = visible_points(WIDE)
    names = [*HALFWAY[:3], "corner_LT"]
    a = {n: (u + (3.0 if i == 1 else 0.0), v) for i, (n, (u, v)) in
         enumerate((n, pts[n]) for n in names)}            # 3 px click noise across the line
    assert vleague.labelling_error_m(a, dict(a), LANDMARKS, (1920, 1080)) is None


def test_labelling_error_is_none_when_pixels_are_collinear_though_the_pitch_points_are_not():
    a = {n: (400.0 + 300 * i, 500.0 + 0.5 * i) for i, n in enumerate(PEN_BOX)}   # 1.5 px sag
    assert vleague.labelling_error_m(a, dict(a), LANDMARKS, (1920, 1080)) is None


# ------------------------------------------------------------------ the vleague task kind

from footlytics.bench import runner  # noqa: E402
from footlytics.bench.paths import BenchPaths  # noqa: E402
from footlytics.geometry.calib_nn import CameraFit  # noqa: E402


class _VCal:
    """Every keyframe solves to the true broadcast camera (or to None)."""
    none = False

    def fit(self, frames, which=None, **kw):
        if self.none:
            return {i: None for i in (which or range(len(frames)))}
        return {i: CameraFit(BROADCAST, 1.0, H_TRUE) for i in (which or range(len(frames)))}


class _IdTracker:
    def __init__(self, frame, mask=None):
        pass

    def update(self, frame, mask=None):
        return np.eye(3)


def _vl_manifest(frames: dict) -> dict:
    return {"version": 1, "size": [1920, 1080], "window": 12,
            "matches": {"m1": {"video": "m1.mp4", "sha256": "0" * 64, "fps": 50.0,
                               "frames": frames}}}


def _vl_window(root: Path, match: str, idx: int) -> None:
    d = root / "data" / "vleague" / match / f"{idx:07d}"
    d.mkdir(parents=True)
    rng = np.random.default_rng(idx)
    for off in range(-12, 13):
        cv2.imwrite(str(d / f"{off:+03d}.jpg"), rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))


def _vl_labels(frames: dict, shift: float = 0.0) -> dict:
    return {"version": 1, "annotator": "A", "session": 1, "image_size": [1920, 1080],
            "frames": {k: {n: [u + shift, v] for n, (u, v) in visible_points().items()}
                       for k in frames}}


@pytest.fixture
def vl(tmp_path, monkeypatch):
    frames = {"0000100": {"split": "dev", "relabel": False},
              "0000200": {"split": "dev", "relabel": False},
              "0000300": {"split": "frozen", "relabel": True}}
    manifest = _vl_manifest(frames)
    repo = tmp_path / "repo"
    (repo / "bench" / "vleague" / "labels").mkdir(parents=True)
    (repo / "bench" / "vleague" / "manifest.json").write_text(json.dumps(manifest))
    for idx in (100, 200, 300):
        _vl_window(tmp_path, "m1", idx)
    cal = _VCal()
    monkeypatch.setattr(vleague, "REPO", repo)
    monkeypatch.setattr(runner, "_calibrator", lambda p: cal)
    monkeypatch.setattr(runner, "require_vram", lambda mib: None)
    monkeypatch.setattr(runner, "TRACKER", _IdTracker)
    monkeypatch.setattr(runner, "contention", lambda: {"load1": 0.0})
    monkeypatch.setattr(runner, "require_disk", lambda *a, **k: None)

    class Env:
        pass
    e = Env()
    e.repo, e.cal, e.manifest, e.paths = repo, cal, manifest, BenchPaths(tmp_path)
    e.write = lambda name, doc: (repo / "bench" / "vleague" / "labels" / name).write_text(
        json.dumps(doc))
    e.write("a.json", {**_vl_labels({"m1/0000100": 1, "m1/0000200": 1, "m1/0000300": 1})})

    def run(**task):
        t = {"id": "v", "kind": "vleague", "split": "dev",
             "labels": ["bench/vleague/labels/a.json"], "every": 10, "gate_px": 25, **task}
        row = runner.run_suite("t", "r", e.paths, "sha", "h",
                               suite_spec={"name": "t", "tasks": [t]})
        return row["tasks"]["v"]
    e.run = run
    return e


def test_vleague_perfect_camera_scores_zero_error_and_all_under_one_metre(vl):
    t = vl.run()
    assert t["status"] == "complete", t
    a = t["aggregate"]
    assert a["frames"] == 2 and a["uncalibrated"] == 0
    assert a["median_error_m"] == pytest.approx(0, abs=1e-6)
    assert a["share_under_1m"] == 1.0
    f = t["per_frame"]["m1/0000100"]
    assert f["n_landmarks"] == len(visible_points()) and f["source"] in ("keyframe", "bridged")
    assert f["error_m"] == pytest.approx(0, abs=1e-6) and "jitter" in f
    assert "labelling_error_m" not in a and a["jitter_switch_m"] is None


def test_vleague_scoreboard_headline_is_the_share_under_one_metre(vl):
    from footlytics.bench.results import headline
    assert headline(vl.run()) == ("share_under_1m", 1.0, None)


def test_vleague_uncalibrated_frames_count_against_the_share(vl):
    vl.cal.none = True
    a = vl.run()["aggregate"]
    assert a["frames"] == 2 and a["uncalibrated"] == 2
    assert a["share_under_1m"] == 0.0 and a["median_error_m"] is None
    f = vl.run()["per_frame"]["m1/0000100"]
    assert f["error_m"] is None and f["source"] == "none"


def test_vleague_one_uncalibrated_frame_halves_the_share(vl, monkeypatch):
    calls = []

    class Half(_VCal):
        def fit(self, frames, which=None, **kw):
            calls.append(1)
            if len(calls) == 2:
                return {i: None for i in which}
            return super().fit(frames, which, **kw)
    monkeypatch.setattr(runner, "_calibrator", lambda p: Half())
    a = vl.run()["aggregate"]
    assert a["frames"] == 2 and a["uncalibrated"] == 1 and a["share_under_1m"] == 0.5
    assert a["median_error_m"] == pytest.approx(0, abs=1e-6)


def test_vleague_split_dev_scores_only_dev_and_frozen_only_frozen(vl):
    assert sorted(vl.run()["per_frame"]) == ["m1/0000100", "m1/0000200"]
    vl.digest_ok = None
    d = vleague.frozen_digest(vleague.load_labels(vl.repo / "bench/vleague/labels/a.json",
                                                  vl.manifest), vl.manifest)
    t = vl.run(split="frozen", frozen_sha256={"bench/vleague/labels/a.json": d})
    assert t["status"] == "complete", t
    assert sorted(t["per_frame"]) == ["m1/0000300"]


def test_frozen_digest_ignores_dev_frames_but_not_frozen_ones(vl):
    path = "bench/vleague/labels/a.json"
    doc = json.loads((vl.repo / path).read_text())
    d = vleague.frozen_digest(doc, vl.manifest)
    assert len(d) == 64
    doc["frames"]["m1/0000100"]["halfway_T"] = [1.0, 2.0]
    vl.write("a.json", doc)
    assert vl.run(split="frozen", frozen_sha256={path: d})["status"] == "complete"
    doc["frames"]["m1/0000300"]["halfway_T"] = [1.0, 2.0]
    vl.write("a.json", doc)
    t = vl.run(split="frozen", frozen_sha256={path: d})
    assert t["status"] == "failed" and "frozen labels changed" in t["error"]


def test_frozen_pending_fails_with_a_not_frozen_message(vl):
    t = vl.run(split="frozen", frozen_sha256={"bench/vleague/labels/a.json": "PENDING"})
    assert t["status"] == "failed" and "not frozen yet" in t["error"]


def test_missing_window_jpeg_fails_naming_the_frame(vl):
    (vl.paths.data / "vleague" / "m1" / "0000200" / "+05.jpg").unlink()
    t = vl.run()
    assert t["status"] == "failed" and "m1/0000200" in t["error"] and "+05.jpg" in t["error"]


def test_vleague_labelling_error_is_a_median_with_counts(vl):
    vl.write("b.json", _vl_labels({"m1/0000300": 1}, shift=2.0))
    t = vl.run(relabels=["bench/vleague/labels/b.json"])
    a = t["aggregate"]
    assert a["labelling_frames"] + a["labelling_none"] == 1
    assert a["labelling_frames"] == 1 and a["labelling_error_m"] > 0


def test_pair_weighted_mean():
    f = vleague._pair_weighted
    assert f([(1.0, 1), (4.0, 2)]) == pytest.approx(3.0)
    assert f([(None, 0), (None, 0)]) is None


# ------------------------------------------------------------------ shipping data

def test_ship_data_replaces_atomically_and_rejects_unsafe_names():
    from footlytics.bench import remote as rm
    from tests.test_bench_remote import FakeSsh, ROOT
    ssh = FakeSsh(ROOT)
    d = rm.Remote("h", runner=ssh).ship_data("vleague", b"tar")
    assert d == "/home/iec/footlytics-bench/data/vleague"
    cmd = next(c for c in ssh.calls if "tar -x" in c)
    assert "data/vleague.tmp" in cmd and "mv " in cmd
    with pytest.raises(ValueError):
        rm.Remote("h", runner=FakeSsh(ROOT)).ship_data("../x", b"")


def test_cli_ship_data_tars_only_the_windows(tmp_path, monkeypatch):
    import io
    import tarfile

    from typer.testing import CliRunner

    from footlytics.bench import cli
    d = tmp_path / "data" / "vleague" / "m1" / "0000100"
    d.mkdir(parents=True)
    (d / "+00.jpg").write_bytes(b"j")
    (tmp_path / "data" / "vleague" / "index.html").write_text("x")
    (tmp_path / "data" / "vleague" / "m1" / "notes.txt").write_text("x")
    got = []

    class R:
        def __init__(self, host):
            pass

        def ship_data(self, name, archive):
            got.append((name, sorted(tarfile.open(fileobj=io.BytesIO(archive)).getnames())))
            return "/x"
    monkeypatch.setattr(cli, "REPO", tmp_path)
    monkeypatch.setattr(cli, "_remote", R)
    r = CliRunner().invoke(cli.bench_app, ["ship-data", "vleague", "--remote", "h"])
    assert r.exit_code == 0, r.output
    assert got == [("vleague", ["m1/0000100/+00.jpg"])]


# ------------------------------------------------------------------ fix round 1

def test_a_failed_frame_stays_in_the_denominator(vl, monkeypatch):
    calls = []

    class Boom(_VCal):
        def fit(self, frames, which=None, **kw):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("CUDA out of memory")
            return super().fit(frames, which, **kw)
    monkeypatch.setattr(runner, "_calibrator", lambda p: Boom())
    t = vl.run()
    a = t["aggregate"]
    assert t["status"] == "partial" and list(t["failed"]) == ["m1/0000200"]
    assert a["frames"] == 2 and a["failed"] == 1 and a["share_under_1m"] == 0.5
    assert a["uncalibrated"] == 0


def test_the_fit_gets_the_tuned_gsr_calibration_settings(vl, monkeypatch):
    seen = []

    class Rec(_VCal):
        def fit(self, frames, which=None, **kw):
            seen.append(kw)
            return super().fit(frames, which, **kw)
    monkeypatch.setattr(runner, "_calibrator", lambda p: Rec())
    vl.run()
    assert seen[0] == {"kp_threshold": runner.DEFAULT_KP_THRESHOLD,
                       "line_threshold": runner.DEFAULT_LINE_THRESHOLD,
                       "voting": False, "mode": "full", "ransac": 10.0, "refine_lines": True}
    seen.clear()
    vl.run(calibrator={"kp_threshold": 0.3, "mode": "fast", "ransac": 4, "refine_lines": False})
    assert seen[0]["kp_threshold"] == 0.3 and seen[0]["mode"] == "fast"
    assert seen[0]["ransac"] == 4.0 and seen[0]["refine_lines"] is False


def _shift(i: float) -> np.ndarray:
    H = H_TRUE.copy()
    H[0, 2] += i          # a distinct H per window index: x shifted by i metres
    return H


def test_error_is_scored_at_list_index_window_and_last_errors_are_counted(vl, monkeypatch):
    class PerIndex(_VCal):
        def fit(self, frames, which=None, **kw):
            self.last_errors = {0: "ZeroDivisionError: x"}
            return {i: CameraFit(BROADCAST, 1.0, _shift(0.0 if i == vleague.WINDOW else 5.0))
                    for i in which}

    class Walk:                      # every frame is "bridged" from a keyframe by an identity
        def __init__(self, frame, mask=None):
            pass

        def update(self, frame, mask=None):
            return np.eye(3)
    monkeypatch.setattr(runner, "_calibrator", lambda p: PerIndex())
    monkeypatch.setattr(runner, "TRACKER", Walk)
    t = vl.run(every=1)             # every window index is a keyframe with its own H
    f = t["per_frame"]["m1/0000100"]
    assert f["error_m"] == pytest.approx(0, abs=1e-6) and f["source"] == "keyframe"
    assert t["aggregate"]["solver_errors"] == 2          # one fit call per frame, one error each
    vl.cal.none = False

    class Off(PerIndex):
        def fit(self, frames, which=None, **kw):
            return {i: CameraFit(BROADCAST, 1.0, _shift(0.0 if i == 0 else 5.0)) for i in which}
    monkeypatch.setattr(runner, "_calibrator", lambda p: Off())
    assert vl.run(every=1)["per_frame"]["m1/0000100"]["error_m"] > 1.0     # index 0 is wrong, so offset 0 is not read from it


def test_jitter_is_measured_on_a_textured_window(vl, monkeypatch):
    d = vl.paths.data / "vleague" / "m1" / "0000100"
    tex = noise_texture()
    small = cv2.resize(tex, (1920, 1080))
    for off in range(-12, 13):
        cv2.imwrite(str(d / f"{off:+03d}.jpg"), small)
    t = vl.run(every=1)
    j = t["per_frame"]["m1/0000100"]["jitter"]
    assert j["pairs_switch"] == 24 and j["pairs_between"] == 0 and j["pairs_skipped"] == 0
    assert j["switch_m"] == pytest.approx(0, abs=0.05)
    assert t["aggregate"]["jitter_switch_m"] is not None


def test_frozen_digest_ignores_int_vs_float_spelling(vl):
    doc = json.loads((vl.repo / "bench/vleague/labels/a.json").read_text())
    d = vleague.frozen_digest(doc, vl.manifest)
    for pts in doc["frames"].values():
        for n, (u, v) in pts.items():
            pts[n] = [float(u), float(v)]
    assert vleague.frozen_digest(doc, vl.manifest) == d
    doc["frames"]["m1/0000300"]["corner_LT"] = [960, 100]
    d1 = vleague.frozen_digest(doc, vl.manifest)
    doc["frames"]["m1/0000300"]["corner_LT"] = [960.0, 100.0]
    assert vleague.frozen_digest(doc, vl.manifest) == d1


def test_ship_data_skips_temp_files(tmp_path, monkeypatch):
    import io
    import tarfile

    from typer.testing import CliRunner

    from footlytics.bench import cli
    d = tmp_path / "data" / "vleague" / "m1" / "0000100"
    d.mkdir(parents=True)
    (d / "+00.jpg").write_bytes(b"j")
    (d / ".tmp-+01.jpg").write_bytes(b"half")
    got = []

    class R:
        def __init__(self, host):
            pass

        def ship_data(self, name, archive):
            got.append(sorted(tarfile.open(fileobj=io.BytesIO(archive)).getnames()))
            return "/x"
    monkeypatch.setattr(cli, "REPO", tmp_path)
    monkeypatch.setattr(cli, "_remote", R)
    assert CliRunner().invoke(cli.bench_app, ["ship-data", "vleague", "--remote", "h"]).exit_code == 0
    assert got == [["m1/0000100/+00.jpg"]]


# ------------------------------------------------------------------ fix round 2

@pytest.mark.parametrize("calibrator", [{}, {"kp_threshold": 0.3, "mode": "fast", "ransac": 4,
                                              "refine_lines": False}])
def test_vleague_passes_the_same_fit_settings_as_gsr_calibration(vl, monkeypatch, calibrator):
    from tests.test_bench_runner import write_gsr_clip

    seen = {"gsr": [], "vleague": []}

    class Rec(_VCal):
        def __init__(self, which):
            self.which = which

        def fit(self, frames, which=None, **kw):
            seen[self.which].append(kw)
            return super().fit(frames, which, **kw)
    write_gsr_clip(vl.paths)
    monkeypatch.setattr(runner, "_calibrator", lambda p: Rec("gsr"))

    class _Eval:
        score = jac = precision = recall = 0.0
        completeness = 0.0
    monkeypatch.setattr(runner, "run_sncalib", lambda *a, **k: _Eval())
    row = runner.run_suite("t", "g", vl.paths, "sha", "h", suite_spec={"name": "t", "tasks": [
        {"id": "gc", "kind": "gsr_calibration", "split": "valid", "clips": ["SNGS-990"],
         "every": 1, "calibrator": calibrator}]})
    assert row["tasks"]["gc"]["status"] in ("complete", "partial"), row["tasks"]["gc"]
    monkeypatch.setattr(runner, "_calibrator", lambda p: Rec("vleague"))
    vl.run(calibrator=calibrator)
    assert seen["gsr"] and seen["vleague"]
    assert seen["vleague"][0] == seen["gsr"][0]


# ---- human rejects (close-ups the green-share test lets through) -----------------------

def test_choose_frames_replaces_a_rejected_pick_with_the_next_free_candidate():
    n, fps = 60_000, 50
    cands = list(range(vleague.WINDOW, n - vleague.WINDOW, 30 * fps))
    base = vleague.choose_frames(n, fps, lambda i: True)
    bad = base[3]
    nxt = next(c for c in cands if c > bad and c not in base)
    got = vleague.choose_frames(n, fps, lambda i: True, rejected={bad})
    assert got == sorted([p for p in base if p != bad] + [nxt])
    got2 = vleague.choose_frames(n, fps, lambda i: True, rejected={bad, nxt})
    nxt2 = next(c for c in cands if c > nxt and c not in base)
    assert got2 == sorted([p for p in base if p != bad] + [nxt2])


def test_choose_frames_falls_back_to_an_earlier_candidate_at_the_end():
    # candidates 12, 62, 112, 162; picks 12, 112, 162; 162 rejected and nothing after it
    got = vleague.choose_frames(200, 10, lambda i: True, k=3, step_s=5.0, rejected={162})
    assert got == [12, 62, 112]


def test_sample_match_with_rejects_replaces_frames_records_them_and_drops_their_windows(
        tmp_path, monkeypatch):
    video = tmp_path / "v.avi"
    _write_video(video, n=200)
    monkeypatch.setattr(vleague, "make_is_pitch", lambda cap: (lambda i: True))
    data = tmp_path / "data"
    first = vleague.sample_match(video, "m", data, k=3, step_s=5.0)
    assert sorted(first["frames"]) == ["0000012", "0000112", "0000162"]
    again = vleague.sample_match(video, "m", data, k=3, step_s=5.0, existing=first,
                                 reject=[112])
    assert again["frames"] == {"0000012": {"split": "dev", "relabel": False},
                               "0000062": {"split": "frozen", "relabel": True},
                               "0000162": {"split": "dev", "relabel": False}}
    assert again["rejected"] == [112] and again["sha256"] == first["sha256"]
    assert (data / "m" / "0000062" / "+00.jpg").exists()
    assert not (data / "m" / "0000112").exists()


def test_sample_match_reuses_an_entry_whose_rejects_are_already_recorded(tmp_path, monkeypatch):
    video = tmp_path / "v.avi"
    _write_video(video, n=200)
    monkeypatch.setattr(vleague, "make_is_pitch", lambda cap: (lambda i: True))
    first = vleague.sample_match(video, "m", tmp_path / "d", k=3, step_s=5.0, reject=[112])
    monkeypatch.setattr(vleague, "make_is_pitch", lambda cap: _no_hash)
    assert vleague.sample_match(video, "m", tmp_path / "d", k=3, step_s=5.0, existing=first,
                                reject=[112]) == first
    assert vleague.sample_match(video, "m", tmp_path / "d", k=3, step_s=5.0,
                                existing=first) == first


def test_cli_reject_goes_to_its_match_and_unknown_matches_are_refused(tmp_path, monkeypatch):
    video = tmp_path / "NINH BÌNH - PVF-CAND - LPBANK V.LEAGUE 1 - 2025-26.avi"
    _write_video(video, n=60)
    seen = {}

    def fake(v, mid, data, existing=None, reject=()):
        seen[mid] = sorted(reject)
        return {"video": v.name, "sha256": "0" * 64, "fps": 10.0, "frames": {}}
    monkeypatch.setattr(vleague, "sample_match", fake)
    r = _cli([video, "--data", tmp_path / "d", "--manifest", tmp_path / "m.json",
              "--reject", "ninh-binh-pvf-cand/0000012", "--reject", "ninh-binh-pvf-cand/0000040"])
    assert r.exit_code == 0, r.output
    assert seen == {"ninh-binh-pvf-cand": [12, 40]}
    r = _cli([video, "--manifest", tmp_path / "m2.json", "--reject", "other-match/0000012"])
    assert r.exit_code != 0 and "other-match" in r.output


# ---- orientation_suspects: left/right and far/near swaps ---------------------------------

def test_orientation_suspects_empty_for_a_true_camera():
    pts = visible_points()
    assert len(pts) >= 8
    assert vleague.orientation_suspects(pts, LANDMARKS) == []


def test_orientation_suspects_names_a_swapped_lr_pair():
    pts = visible_points()
    a, b = "pen_area_LT_front", "pen_area_RT_front"
    assert a in pts and b in pts
    pts[a], pts[b] = pts[b], pts[a]
    assert vleague.orientation_suspects(pts, LANDMARKS) == sorted([a, b])


def test_orientation_suspects_single_mislabel_is_the_culprit():
    pts = visible_points()
    pts["pen_area_RT_front"] = pts["corner_LT"]   # clicked on the far-left corner instead
    pts = {n: p for n, p in pts.items() if n != "corner_LT"}
    assert vleague.orientation_suspects(pts, LANDMARKS) == ["pen_area_RT_front"]


def test_orientation_suspects_needs_three_points_and_flags_a_flipped_triangle():
    pts = visible_points()
    names = ["pen_area_LT_front", "pen_area_RT_front"]
    two = {names[0]: pts[names[1]], names[1]: pts[names[0]]}
    assert vleague.orientation_suspects(two, LANDMARKS) == []
    trio = next(t for t in itertools.combinations(sorted(pts), 3)
                if vleague._bad_triples(list(t), {n: pts[n] for n in t},
                                        {n: (-LANDMARKS[n][0], LANDMARKS[n][1]) for n in t}))
    three = {n: pts[n] for n in trio}
    assert vleague.orientation_suspects(three, LANDMARKS) == []
    a, b, c = trio
    swapped = {a: pts[b], b: pts[a], c: pts[c]}
    assert vleague.orientation_suspects(swapped, LANDMARKS) == sorted(trio)
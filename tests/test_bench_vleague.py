"""V.League mini-GT: frame sampling, splits and the manifest."""
import json
from pathlib import Path

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
    for ident in ("canvas", "frames", "landmarks", "pitch", "status"):
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


def test_labelling_error_of_a_two_pixel_shift_is_small_but_positive():
    pts = visible_points()
    err = vleague.labelling_error_m(pts, labels_from(pts, 2.0), LANDMARKS, (1920, 1080))
    assert 0 < err < 0.5


def test_labelling_error_needs_four_shared_landmarks():
    pts = visible_points()
    names = sorted(pts)
    a = {n: pts[n] for n in names[:6]}
    b = {n: pts[n] for n in names[3:9]}                # shares only 3
    assert len(set(a) & set(b)) == 3
    assert vleague.labelling_error_m(a, b, LANDMARKS, (1920, 1080)) is None
    b4 = {n: pts[n] for n in names[2:8]}                # shares 4
    assert vleague.labelling_error_m(a, b4, LANDMARKS, (1920, 1080)) is not None

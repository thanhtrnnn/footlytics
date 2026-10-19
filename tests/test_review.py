"""Sequence review: scoring, stable ids, and the local server as a browser would hit it."""
import http.client
import json
import threading

import pandas as pd
import pytest
from typer.testing import CliRunner

from footlytics.analytics import review
from footlytics.analytics.sequences import COLUMNS
from footlytics.review_server import make_server


def _seqs() -> pd.DataFrame:
    rows = [
        ("low_block", "HOME", 1, 100.0, 130.0, 2500, 3250, 30.0, -28.0, 11.0, 0),
        ("low_block", "AWAY", 1, 200.0, 215.0, 5000, 5375, 15.0, -26.5, 12.5, 1),
        ("low_block", "HOME", 2, 3000.0, 3012.0, 75000, 75300, 12.0, -30.0, 10.0, 0),
        ("high_press", "AWAY", 1, 400.0, 410.0, 10000, 10250, 10.0, 12.0, 9.0, 4),
    ]
    return pd.DataFrame(rows, columns=COLUMNS[:11]).reindex(columns=COLUMNS)


@pytest.fixture
def files(tmp_path):
    csv = tmp_path / "seq.csv"
    _seqs().to_csv(csv, index=False)
    video = tmp_path / "match.mp4"
    video.write_bytes(bytes(range(256)) * 4)
    return csv, video, tmp_path / "marks.json"


@pytest.fixture
def live(files):
    csv, video, marks = files
    server = make_server(video, csv, marks, "ana", 5.0, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], marks, video
    server.shutdown()
    server.server_close()


def _req(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    hdrs = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body)
        hdrs["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=hdrs)
    r = conn.getresponse()
    out = (r.status, dict(r.getheaders()), r.read())
    conn.close()
    return out


def test_ids_are_stable_across_reloads(files):
    csv, _, _ = files
    first = review.sequence_ids(pd.read_csv(csv))
    again = review.sequence_ids(pd.read_csv(csv).sample(frac=1, random_state=3).sort_index())
    assert first == again
    assert first == ["low_block-HOME-1-2500", "low_block-AWAY-1-5000",
                     "low_block-HOME-2-75000", "high_press-AWAY-1-10000"]


def test_score_marks_exact_numbers():
    ids = review.sequence_ids(_seqs())
    marks = review.new_marks("ana", "match.mp4")
    marks["sequences"] = {ids[0]: {"verdict": "right", "note": ""},
                          ids[1]: {"verdict": "right", "note": ""},
                          ids[3]: {"verdict": "wrong", "note": "ball not in third"}}
    marks["missed"] = [{"label": "high_press", "team": "HOME", "start_s": 500.0, "end_s": 508.0, "note": ""}]
    s = review.score_marks(marks, _seqs())
    assert s["low_block"] == {"n": 3, "marked": 2, "right": 2, "wrong": 0, "unsure": 0,
                              "precision": 1.0, "missed": 0}
    assert s["high_press"] == {"n": 1, "marked": 1, "right": 0, "wrong": 1, "unsure": 0,
                               "precision": 0.0, "missed": 1}


def test_precision_is_none_without_judgements():
    marks = review.new_marks("ana", "match.mp4")
    marks["sequences"] = {review.sequence_ids(_seqs())[0]: {"verdict": "unsure", "note": ""}}
    s = review.score_marks(marks, _seqs())["low_block"]
    assert (s["unsure"], s["precision"]) == (1, None)


def test_verdict_post_is_saved_to_the_file(live):
    port, marks, _ = live
    status, _, body = _req(port, "POST", "/verdict",
                           {"id": "low_block-HOME-1-2500", "verdict": "right", "note": "clear"})
    assert status == 200
    saved = review.load_marks(marks)
    assert saved["sequences"] == {"low_block-HOME-1-2500": {"verdict": "right", "note": "clear"}}
    assert saved["video_offset_s"] == 5.0 and json.loads(body) == saved
    assert _req(port, "POST", "/note", {"id": "low_block-HOME-1-2500", "note": "very clear"})[0] == 200
    assert review.load_marks(marks)["sequences"]["low_block-HOME-1-2500"]["note"] == "very clear"


@pytest.mark.parametrize("path,body", [
    ("/verdict", {"id": "low_block-HOME-1-9999", "verdict": "right", "note": ""}),
    ("/verdict", {"id": "low_block-HOME-1-2500", "verdict": "maybe", "note": ""}),
    ("/verdict", {"id": "low_block-HOME-1-2500", "verdict": "right", "note": 5}),
    ("/note", {"id": "low_block-HOME-1-2500", "note": "no verdict yet"}),
    ("/missed", {"label": "low_block", "team": "HOME", "start_s": 9, "end_s": 3}),
    ("/missed", {"label": "low_block", "team": "NOPE", "start_s": 1, "end_s": 3}),
    ("/missed/delete", {"index": 0}),
])
def test_bad_posts_are_400_and_leave_the_file_unchanged(live, path, body):
    port, marks, _ = live
    before = marks.read_bytes()
    assert _req(port, "POST", path, body)[0] == 400
    assert marks.read_bytes() == before


def test_missed_sequences_can_be_added_and_removed(live):
    port, marks, _ = live
    entry = {"label": "high_press", "team": "HOME", "start_s": 500.0, "end_s": 508.5, "note": "n"}
    assert _req(port, "POST", "/missed", entry)[0] == 200
    assert review.load_marks(marks)["missed"] == [entry]
    assert _req(port, "POST", "/missed/delete", {"index": 0})[0] == 200
    assert review.load_marks(marks)["missed"] == []


def test_range_request_returns_exactly_the_bytes(live):
    port, _, video = live
    status, headers, body = _req(port, "GET", "/video", headers={"Range": "bytes=10-19"})
    assert status == 206
    assert body == video.read_bytes()[10:20] == bytes(range(10, 20))
    assert headers["Content-Range"] == "bytes 10-19/1024"
    assert headers["Accept-Ranges"] == "bytes" and headers["Content-Length"] == "10"


def test_full_and_open_ended_video_requests(live):
    port, _, video = live
    status, headers, body = _req(port, "GET", "/video")
    assert (status, body, headers["Accept-Ranges"]) == (200, video.read_bytes(), "bytes")
    status, _, body = _req(port, "GET", "/video", headers={"Range": "bytes=1000-"})
    assert (status, len(body)) == (206, 24)
    assert _req(port, "GET", "/video", headers={"Range": "bytes=5000-"})[0] == 416


def test_other_paths_are_404(live):
    port, _, video = live
    for path in ("/etc/passwd", "/video/", f"/{video.name}", "/../match.mp4", "/video.mp4"):
        assert _req(port, "GET", path)[0] == 404


def test_non_loopback_host_is_refused(live):
    port, marks, _ = live
    before = marks.read_bytes()
    assert _req(port, "GET", "/", headers={"Host": "evil.example"})[0] == 403
    assert _req(port, "GET", "/video", headers={"Host": f"evil.example:{port}"})[0] == 403
    post = _req(port, "POST", "/verdict", {"id": "low_block-HOME-1-2500", "verdict": "right", "note": ""},
                headers={"Host": "evil.example"})
    assert post[0] == 403 and marks.read_bytes() == before


def test_page_embeds_sequences_and_marks(live):
    port, _, _ = live
    status, _, body = _req(port, "GET", "/")
    page = body.decode()
    assert status == 200 and "/*DATA*/null" not in page
    assert "high_press-AWAY-1-10000" in page and '"video_offset_s": 5.0' in page


def test_server_refuses_another_reviewers_file(files):
    csv, video, marks = files
    make_server(video, csv, marks, "ana", 0.0).server_close()
    with pytest.raises(ValueError, match="belongs to reviewer"):
        make_server(video, csv, marks, "bob", 0.0)


def test_cli_score_prints_the_table(files):
    from footlytics.cli import app

    csv, _, marks = files
    doc = review.new_marks("ana", "match.mp4")
    doc["sequences"] = {"low_block-HOME-1-2500": {"verdict": "right", "note": ""},
                        "low_block-AWAY-1-5000": {"verdict": "wrong", "note": ""}}
    review.write_marks(marks, doc)
    r = CliRunner().invoke(app, ["review", "score", "--sequences", str(csv), "--marks", str(marks)])
    assert r.exit_code == 0, r.output
    lb = next(l for l in r.output.splitlines() if l.startswith("low_block")).split()
    assert lb == ["low_block", "3", "2", "1", "1", "0", "0.500", "0"]
    hp = next(l for l in r.output.splitlines() if l.startswith("high_press")).split()
    assert hp[-2:] == ["n/a", "0"]

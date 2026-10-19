"""Local review server: an analyst marks detected sequences right or wrong while watching the video.

Loopback only. Serves the page, one video file at a fixed URL (with Range support so the
browser can seek), and validated JSON POSTs that are written to the marks file at once.
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from .analytics import review

TEMPLATE = Path(__file__).with_name("review.html")
PLACEHOLDER = "/*DATA*/null"
MAX_BODY = 8192
VIDEO_URL = "/video"
CHUNK = 1 << 20
EVIDENCE = ["def_line_x", "compactness_m", "n_within_10m"]
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
          ".mkv": "video/x-matroska"}


def parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Inclusive (start, end) for a single `bytes=` range, or None if unsatisfiable or malformed."""
    m = _RANGE.match(header.strip())
    if not m or (m[1] == "" and m[2] == ""):
        return None
    if m[1] == "":
        n = int(m[2])
        if n == 0:
            return None
        return max(size - n, 0), size - 1
    start = int(m[1])
    end = min(int(m[2]), size - 1) if m[2] else size - 1
    if start >= size or end < start:
        return None
    return start, end


def sequence_rows(sequences: pd.DataFrame) -> list[dict]:
    rows = []
    for sid, r in zip(review.sequence_ids(sequences), sequences.to_dict("records")):
        rows.append({"id": sid, "label": r["label"], "team": r["team"], "period": int(r["period"]),
                     "start_s": float(r["start_s"]), "end_s": float(r["end_s"]),
                     "duration_s": float(r["duration_s"]),
                     **{k: (None if pd.isna(r[k]) else float(r[k])) for k in EVIDENCE}})
    return rows


def render_page(rows: list[dict], marks: dict, labels: list[str]) -> bytes:
    data = {"sequences": rows, "marks": marks, "labels": labels, "teams": list(review.TEAMS),
            "video_url": VIDEO_URL}
    blob = json.dumps(data).replace("</", "<\\/")
    return TEMPLATE.read_text().replace(PLACEHOLDER, blob).encode()


def make_server(video, sequences_csv, marks_path, reviewer: str, video_offset_s: float = 0.0,
                port: int = 0) -> ThreadingHTTPServer:
    video, marks_path = Path(video), Path(marks_path)
    if not video.is_file():
        raise ValueError(f"video {video} not found")
    rows = sequence_rows(pd.read_csv(sequences_csv))
    ids = {r["id"] for r in rows}
    if len(ids) != len(rows):
        raise ValueError("sequence ids are not unique")
    if marks_path.exists():
        marks = review.load_marks(marks_path)
        if (marks["reviewer"], marks["video_offset_s"]) != (reviewer, float(video_offset_s)):
            raise ValueError(f"{marks_path} belongs to reviewer {marks['reviewer']!r} with offset "
                             f"{marks['video_offset_s']}, not {reviewer!r} with {video_offset_s}")
    else:
        marks_path.parent.mkdir(parents=True, exist_ok=True)
        review.write_marks(marks_path, review.new_marks(reviewer, video.name, video_offset_s))
    labels = sorted(set(review.LABELS) | {r["label"] for r in rows})
    lock = threading.Lock()
    ctype = _TYPES.get(video.suffix.lower(), "application/octet-stream")

    def mutate(fn) -> dict:
        with lock:
            doc = review.load_marks(marks_path)
            fn(doc)
            review.validate_marks(doc)
            review.write_marks(marks_path, doc)
            return doc

    def known_id(req: dict) -> str:
        sid = req.get("id")
        if not isinstance(sid, str) or sid not in ids:
            raise ValueError(f"unknown sequence id {sid!r}")
        return sid

    def do_verdict(req: dict) -> dict:
        sid = known_id(req)
        verdict, note = review.check_verdict(req.get("verdict"), req.get("note", ""))
        return mutate(lambda d: d["sequences"].__setitem__(sid, {"verdict": verdict, "note": note}))

    def do_note(req: dict) -> dict:
        sid, note = known_id(req), req.get("note")
        if not isinstance(note, str):
            raise ValueError("note must be text")

        def fn(d):
            if sid not in d["sequences"]:
                raise ValueError("mark the sequence before saving a note")
            d["sequences"][sid]["note"] = note
        return mutate(fn)

    def do_missed(req: dict) -> dict:
        entry = review.check_missed(req)
        return mutate(lambda d: d["missed"].append(entry))

    def do_missed_delete(req: dict) -> dict:
        i = req.get("index")
        if isinstance(i, bool) or not isinstance(i, int):
            raise ValueError("index must be an integer")

        def fn(d):
            if not 0 <= i < len(d["missed"]):
                raise ValueError(f"no missed entry {i}")
            del d["missed"][i]
        return mutate(fn)

    posts = {"/verdict": do_verdict, "/note": do_note, "/missed": do_missed,
             "/missed/delete": do_missed_delete}

    class Handler(BaseHTTPRequestHandler):
        server_version = "footlytics-review"

        def log_message(self, *args):
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def _text(self, status: int, msg: str) -> None:
            self._send(status, msg.encode(), "text/plain; charset=utf-8")

        def _host_ok(self) -> bool:
            port = self.server.server_address[1]
            if self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}"):
                return True
            self._text(403, "forbidden host")
            return False

        def _video(self) -> None:
            size = video.stat().st_size
            header = self.headers.get("Range")
            start, end, status = 0, size - 1, 200
            if header:
                rng = parse_range(header, size)
                if rng is None:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                (start, end), status = rng, 206
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            left = end - start + 1
            try:
                with open(video, "rb") as f:
                    f.seek(start)
                    while left > 0:
                        chunk = f.read(min(CHUNK, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self._host_ok():
                return
            path = urlsplit(self.path).path
            if path == "/":
                with lock:
                    page = render_page(rows, review.load_marks(marks_path), labels)
                self._send(200, page, "text/html; charset=utf-8")
            elif path == VIDEO_URL:
                self._video()
            elif path == "/marks":
                with lock:
                    self._send(200, marks_path.read_bytes(), "application/json")
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self._text(404, "not found")

        def do_POST(self):
            if not self._host_ok():
                return
            handler = posts.get(urlsplit(self.path).path)
            if handler is None:
                return self._text(404, "not found")
            if self.headers.get_content_type() != "application/json":
                return self._text(415, "Content-Type must be application/json")
            try:
                n = int(self.headers.get("Content-Length", ""))
                if not 0 < n <= MAX_BODY:
                    raise ValueError("body size")
                req = json.loads(self.rfile.read(n))
                if not isinstance(req, dict):
                    raise ValueError("body must be an object")
                doc = handler(req)
            except ValueError as e:
                return self._text(400, f"bad request: {e}")
            except Exception as e:  # disk full, file removed: the old file is still intact
                return self._text(500, f"could not save: {e}")
            self._json(200, doc)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server

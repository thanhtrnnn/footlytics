"""Local labelling server for the V.League mini-GT: one page, autosaved to a label file.

Loopback only. Frames are served by manifest key (never by path) and every POST is validated.
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from ..geometry.annotate import HELP, SUGGESTED
from ..geometry.pitch import DEFAULT_PITCH
from . import vleague

TEMPLATE = Path(__file__).with_name("vleague_label.html")
PLACEHOLDER = "/*DATA*/null"
MAX_BODY = 4096
_FRAME_URL = re.compile(r"^/frames/([^/]+)/(\d{7})\.jpg$")


def landmark_names() -> list[str]:
    """SUGGESTED first (in clicking order), then the remaining DEFAULT_PITCH landmarks."""
    rest = [n for n in DEFAULT_PITCH.landmarks() if n not in SUGGESTED]
    return [*SUGGESTED, *rest]


def render_page(annotator: str, session: int) -> bytes:
    data = {"names": landmark_names(), "help": HELP,
            "pitch": {n: [x, y] for n, (x, y) in DEFAULT_PITCH.landmarks().items()},
            "size": list(vleague.SIZE), "annotator": annotator, "session": session}
    blob = json.dumps(data).replace("</", "<\\/")
    return TEMPLATE.read_text().replace(PLACEHOLDER, blob).encode()


def make_server(manifest_path, data_dir, labels_path, annotator: str, session: int,
                relabel_only: bool, port: int = 0) -> ThreadingHTTPServer:
    manifest = json.loads(Path(manifest_path).read_text())
    frames = vleague.manifest_frames(manifest)
    data_dir, labels_path = Path(data_dir), Path(labels_path)
    if labels_path.exists():
        doc = vleague.load_labels(labels_path, manifest)
        if (doc["annotator"], doc["session"]) != (annotator, session):
            raise ValueError(f"{labels_path} belongs to annotator {doc['annotator']!r} session "
                             f"{doc['session']!r}, not {annotator!r} session {session!r}")
    else:
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        vleague._write_atomic(labels_path, vleague.new_labels(annotator, session))
    page = render_page(annotator, session)
    lock = threading.Lock()

    def frames_list() -> list[dict]:
        doc = json.loads(labels_path.read_text())
        return [{"key": k, "split": m["split"], "relabel": m["relabel"],
                 "points": len(doc["frames"].get(k, {}))}
                for k, m in sorted(frames.items()) if m["relabel"] or not relabel_only]

    class Handler(BaseHTTPRequestHandler):
        server_version = "vleague-label"

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

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == "/":
                self._send(200, page, "text/html; charset=utf-8")
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif path == "/labels":
                with lock:
                    self._send(200, labels_path.read_bytes(), "application/json")
            elif path == "/frames-list":
                with lock:
                    self._json(200, frames_list())
            elif m := _FRAME_URL.match(path):
                key = f"{m[1]}/{m[2]}"
                if key not in frames:
                    return self._text(404, "no such frame")
                jpg = data_dir / m[1] / m[2] / "+00.jpg"
                if not jpg.is_file():
                    return self._text(404, "frame file missing")
                self._send(200, jpg.read_bytes(), "image/jpeg")
            else:
                self._text(404, "not found")

        def do_POST(self):
            if urlsplit(self.path).path != "/label":
                return self._text(404, "not found")
            try:
                n = int(self.headers.get("Content-Length", ""))
                if not 0 < n <= MAX_BODY:
                    raise ValueError("body size")
                req = json.loads(self.rfile.read(n))
                if not isinstance(req, dict):
                    raise ValueError("body must be an object")
            except ValueError as e:
                return self._text(400, f"bad request: {e}")
            key, name, uv = req.get("frame"), req.get("landmark"), req.get("uv")
            if not isinstance(key, str) or key not in frames:
                return self._text(400, f"frame {key!r} is not in the manifest")
            if not isinstance(name, str):
                return self._text(400, "landmark must be a name")
            try:
                with lock:
                    doc = vleague.save_label(labels_path, key, name, uv)
            except ValueError as e:
                return self._text(400, str(e))
            self._json(200, {"frame": key, "points": doc["frames"].get(key, {})})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)

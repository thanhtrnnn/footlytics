"""Multi-match Web Tactical Query & Synchronized Video + 2D Radar Workspace (`footlytics workspace serve`).

Serves an interactive club-analyst web workspace over one or more processed match directories:
- Multi-match filtering across clubs, opponents, tactical patterns, and live game-model thresholds.
- Synchronized HTTP Range `.mp4` clip/match streaming + on-demand 2D pitch radar trajectory JSON
  from `tracks.parquet` so every sequence has smooth interactive 2D radar playback even without a
  pre-rendered video clip.
- Live `right` / `wrong` / `unsure` verdict and analyst note persistence back to each match's
  `marks.json`.
- One-click filtered export to Hudl Sportscode / LongoMatch `.xml` and `.csv`.
- Optional bearer / query / HttpOnly-cookie token authentication (`--token`) for club pilot sharing.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import ipaddress
import json
import secrets
import threading
from collections.abc import Sequence
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd

from .analytics import review
from .analytics.pack import (
    LABEL_DISPLAY_VI,
    LABEL_ORDER,
    _period_elapsed_s,
    _resolve_period_starts,
    evidence_summary,
    format_clock,
    format_match_clock,
    resolve_sequence_marks,
    write_sportscode_xml,
)
from .analytics.profiles import (
    BUILTIN_PROFILES,
    TacticalProfile,
    load_profile,
    recompute_match_sequences,
)
from .analytics.query import MatchEntry, discover_matches
from .review_server import _TYPES, CHUNK, MAX_BODY, parse_range
from .state.schema import MatchState, Role, Team
from .viz.radar import hold_gaps

TEMPLATE = Path(__file__).with_name("workspace.html")
PLACEHOLDER = "/*WORKSPACE_DATA*/null"
COOKIE_NAME = "footlytics_ws"

_TEAM_CODE = {Team.HOME.value: 0, Team.AWAY.value: 1, Team.OFFICIAL.value: 2}
_ROLE_CODE = {Role.PLAYER.value: 0, Role.GOALKEEPER.value: 1, Role.REFEREE.value: 2}


def _num_or_none(v: Any) -> float | None:
    if v is None or pd.isna(v):
        return None
    f = float(v)
    return round(f, 3) if np.isfinite(f) else None


def _int_or_none(v: Any) -> int | None:
    if v is None or pd.isna(v):
        return None
    return int(v)


def _index_pack_clips(pack_dirs: Sequence[Path | str]) -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str, int, int], Path]]:
    """Scan exported pack directories for `playlist.csv` -> existing `.mp4` clip paths."""
    by_sid: dict[tuple[str, str], Path] = {}
    by_approx: dict[tuple[str, str, int, int], Path] = {}
    for raw_root in pack_dirs:
        root = Path(raw_root)
        if not root.exists():
            continue
        csvs = [root / "playlist.csv"] if (root / "playlist.csv").is_file() else sorted(root.rglob("playlist.csv"))
        for csv_path in csvs:
            pdir = csv_path.parent
            with contextlib.suppress(OSError, ValueError, KeyError):
                df = pd.read_csv(csv_path)
                if "clip_path" not in df.columns or "match_id" not in df.columns:
                    continue
                for r in df.itertuples():
                    rel = getattr(r, "clip_path", "")
                    if not isinstance(rel, str) or not rel:
                        continue
                    mp4 = (pdir / rel).resolve()
                    if not mp4.is_file():
                        continue
                    mid = str(r.match_id)
                    sid = str(getattr(r, "sequence_id", "") or "")
                    if sid and (mid, sid) not in by_sid:
                        by_sid[(mid, sid)] = mp4
                    lbl = str(getattr(r, "label", "") or "")
                    per = int(getattr(r, "period", 1) or 1)
                    st_round = round(float(getattr(r, "start_s", 0.0) or 0.0))
                    if lbl and (mid, lbl, per, st_round) not in by_approx:
                        by_approx[(mid, lbl, per, st_round)] = mp4
    return by_sid, by_approx


def _find_match_video(entry: MatchEntry, marks_doc: dict[str, Any] | None, video_dir: Path | None) -> Path | None:
    cands: list[Path] = []
    if marks_doc and marks_doc.get("video"):
        vname = str(marks_doc["video"])
        if video_dir is not None:
            cands.append(Path(video_dir) / vname)
        cands.append(entry.dir / vname)
    for ext in (".mp4", ".mkv", ".mov", ".m4v"):
        if video_dir is not None:
            cands.append(Path(video_dir) / f"{entry.match_id}{ext}")
            cands.append(Path(video_dir) / f"{entry.dir.name}{ext}")
        cands.append(entry.dir / f"{entry.match_id}{ext}")
        cands.append(entry.dir / f"{entry.dir.name}{ext}")
    for c in cands:
        if c.is_file():
            return c.resolve()
    return None


class WorkspaceStore:
    """Thread-safe multi-match tactical index, radar slicer, and `marks.json` persistence."""

    def __init__(
        self,
        roots: Sequence[Path | str] | Path | str,
        *,
        pack_dirs: Sequence[Path | str] = (),
        video_dir: Path | str | None = None,
        reviewer: str = "analyst",
        pre_roll_s: float = 3.0,
        post_roll_s: float = 2.0,
        profile: str | Path | TacticalProfile | None = None,
    ):
        entries = discover_matches(roots)
        if not entries:
            raise ValueError("no match directories containing sequences.csv found")
        self.entries: dict[str, MatchEntry] = {e.match_id: e for e in entries}
        self.reviewer = reviewer
        self.pre_roll_s = float(pre_roll_s)
        self.post_roll_s = float(post_roll_s)
        self.video_dir = Path(video_dir) if video_dir is not None else None
        if isinstance(profile, TacticalProfile):
            self.profile = profile
        elif profile is not None:
            self.profile = load_profile(profile)
        else:
            self.profile = BUILTIN_PROFILES["default"]
        self._lock = threading.Lock()
        self._state_cache: dict[str, MatchState] = {}
        self._override_sequences: dict[str, pd.DataFrame] = {}

        self._clip_by_sid, self._clip_by_approx = _index_pack_clips(pack_dirs)
        self.match_videos: dict[str, tuple[Path, float]] = {}
        self.period_starts: dict[str, dict[int, float]] = {}
        self.matches_meta: list[dict[str, Any]] = []
        self._rows_by_uid: dict[str, dict[str, Any]] = {}
        self._uid_order: list[str] = []
        self._build_index()

    def _build_index(self) -> None:
        self.matches_meta = []
        self._rows_by_uid = {}
        self._uid_order = []
        all_rows: list[dict[str, Any]] = []
        for entry in self.entries.values():
            seq_df = self._override_sequences.get(entry.match_id)
            if seq_df is None:
                seq_df = pd.read_csv(entry.sequences_path)
            marks_path = entry.dir / "marks.json"
            marks_doc = review.load_marks(marks_path) if marks_path.is_file() else None
            resolved = resolve_sequence_marks(marks_doc, seq_df)
            p_starts = _resolve_period_starts(entry)
            self.period_starts[entry.match_id] = p_starts

            vid = _find_match_video(entry, marks_doc, self.video_dir)
            vid_offset = float(marks_doc.get("video_offset_s", 0.0)) if marks_doc else 0.0
            if vid is not None:
                self.match_videos[entry.match_id] = (vid, vid_offset)

            q_path = entry.dir / "quality.json"
            q_doc = json.loads(q_path.read_text()) if q_path.is_file() else {}
            has_radar = entry.tracks_path is not None and entry.tracks_path.is_file()

            for (_, row), (sid, verdict, note) in zip(seq_df.iterrows(), resolved):
                uid = f"{entry.match_id}::{sid}"
                side = str(row["team"])
                club = entry.home if side == "home" else (entry.away if side == "away" else "")
                opp = entry.away if side == "home" else (entry.home if side == "away" else "")
                per = int(row["period"])
                s0, s1 = float(row["start_s"]), float(row["end_s"])
                e0 = _period_elapsed_s(s0, per, p_starts)
                e1 = _period_elapsed_s(s1, per, p_starts)
                lbl = str(row["label"])
                clip_file = self._clip_by_sid.get((entry.match_id, sid)) or self._clip_by_approx.get(
                    (entry.match_id, lbl, per, round(s0))
                )
                c0 = max(0.0, s0 - self.pre_roll_s)
                c1 = s1 + self.post_roll_s
                item: dict[str, Any] = {
                    "uid": uid,
                    "sequence_id": sid,
                    "match_id": entry.match_id,
                    "home_team": entry.home or "HOME",
                    "away_team": entry.away or "AWAY",
                    "club": club or side.upper(),
                    "opponent": opp or ("AWAY" if side == "home" else "HOME"),
                    "side": side,
                    "period": per,
                    "match_clock": f"{format_match_clock(e0, per)}-{format_match_clock(e1, per)}",
                    "clock": f"{format_clock(s0)}-{format_clock(s1)}",
                    "period_start_s": p_starts.get(per),
                    "start_s": round(s0, 2),
                    "end_s": round(s1, 2),
                    "clip_start_s": round(c0, 2),
                    "clip_end_s": round(c1, 2),
                    "duration_s": round(float(row["duration_s"]), 2),
                    "label": lbl,
                    "label_vi": LABEL_DISPLAY_VI.get(lbl, lbl),
                    "verdict": verdict,
                    "note": note,
                    "evidence_summary": evidence_summary(row),
                    "def_line_x": _num_or_none(row.get("def_line_x")),
                    "compactness_m": _num_or_none(row.get("compactness_m")),
                    "n_within_10m": _int_or_none(row.get("n_within_10m")),
                    "n_pressers": _int_or_none(row.get("n_pressers")),
                    "flank": str(row.get("flank") or "") if pd.notna(row.get("flank")) else "",
                    "n_attackers": _int_or_none(row.get("n_attackers")),
                    "n_defenders": _int_or_none(row.get("n_defenders")),
                    "in_possession_share": _num_or_none(row.get("in_possession_share")),
                    "ball_dist_m": _num_or_none(row.get("ball_dist_m")),
                    "retreat_mps": _num_or_none(row.get("retreat_mps")),
                    "response": str(row.get("response") or "") if pd.notna(row.get("response")) else "",
                    "clip_url": f"/api/clip/{uid}" if clip_file is not None else "",
                    "video_url": f"/api/video/{entry.match_id}" if vid is not None else "",
                    "video_offset_s": vid_offset,
                    "has_clip": clip_file is not None,
                    "has_full_video": vid is not None,
                    "has_radar": has_radar,
                    "_clip_path": clip_file,
                }
                all_rows.append(item)

            self.matches_meta.append(
                {
                    "match_id": entry.match_id,
                    "home": entry.home or "HOME",
                    "away": entry.away or "AWAY",
                    "fps": entry.fps,
                    "calib_success_rate": q_doc.get("calib_success_rate"),
                    "ball_coverage": q_doc.get("ball_coverage"),
                    "has_radar": has_radar,
                    "has_video": vid is not None,
                    "sequence_count": len(seq_df),
                }
            )

        rank = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
        all_rows.sort(key=lambda r: (rank.get(r["label"], 99), r["match_id"], r["start_s"]))
        for idx, r in enumerate(all_rows, 1):
            r["clip_index"] = idx
            self._rows_by_uid[r["uid"]] = r
            self._uid_order.append(r["uid"])

    def public_snapshot(self) -> dict[str, Any]:
        with self._lock:
            seqs = [
                {k: v for k, v in self._rows_by_uid[u].items() if not k.startswith("_")}
                for u in self._uid_order
            ]
        clubs = sorted({s["club"] for s in seqs if s["club"]})
        labels = [lbl for lbl in LABEL_ORDER if any(s["label"] == lbl for s in seqs)]
        for s in seqs:
            if s["label"] not in labels:
                labels.append(s["label"])
        return {
            "static_mode": False,
            "pack_title": "",
            "dossier_url": "",
            "passcode_sha256": "",
            "reviewer": self.reviewer,
            "matches": self.matches_meta,
            "clubs": clubs,
            "labels": labels,
            "label_vi": LABEL_DISPLAY_VI,
            "sequences": seqs,
            "profile": self.profile.to_dict(),
            "presets": [p.to_dict() for p in BUILTIN_PROFILES.values()],
        }

    def recompute_with_profile(
        self,
        profile: TacticalProfile,
        *,
        persist: bool = False,
    ) -> dict[str, Any]:
        """Re-run tactical sequence detection across loaded matches using custom thresholds."""
        updated: dict[str, pd.DataFrame] = {}
        for mid, entry in self.entries.items():
            if entry.tracks_path is None or not entry.tracks_path.is_file():
                continue
            st = self._get_state(mid)
            updated[mid] = recompute_match_sequences(entry.dir, profile, write=persist, state=st)
        if not updated:
            raise ValueError("no loaded matches have tracks.parquet available for live recompute")
        with self._lock:
            self.profile = profile
            self._override_sequences = updated
            self._build_index()
        return self.public_snapshot()

    def clip_path_for(self, uid: str) -> Path | None:
        row = self._rows_by_uid.get(uid)
        return row["_clip_path"] if row is not None else None

    def video_path_for(self, match_id: str) -> Path | None:
        mv = self.match_videos.get(match_id)
        return mv[0] if mv is not None else None

    def _write_verdict(self, uid: str, verdict_raw: Any, note_raw: Any) -> dict[str, Any]:
        """Validate and persist one verdict. Caller must hold `self._lock`."""
        if uid not in self._rows_by_uid:
            raise ValueError(f"unknown sequence uid {uid!r}")
        verdict, note = review.check_verdict(verdict_raw, note_raw)
        row = self._rows_by_uid[uid]
        entry = self.entries[row["match_id"]]
        marks_path = entry.dir / "marks.json"
        if marks_path.is_file():
            doc = review.load_marks(marks_path)
        else:
            doc = review.new_marks(self.reviewer, f"{entry.match_id}.mp4", 0.0)
        doc["sequences"][row["sequence_id"]] = {"verdict": verdict, "note": note}
        review.validate_marks(doc)
        review.write_marks(marks_path, doc)
        row["verdict"] = verdict
        row["note"] = note
        return {"uid": uid, "verdict": verdict, "note": note}

    def set_verdict(self, uid: str, verdict_raw: Any, note_raw: Any = "") -> dict[str, Any]:
        with self._lock:
            return self._write_verdict(uid, verdict_raw, note_raw)

    def set_note(self, uid: str, note_raw: Any) -> dict[str, Any]:
        if not isinstance(note_raw, str):
            raise TypeError("note must be text")
        with self._lock:
            if uid not in self._rows_by_uid:
                raise ValueError(f"unknown sequence uid {uid!r}")
            row = self._rows_by_uid[uid]
            if not row["verdict"]:
                raise ValueError("mark sequence right/wrong/unsure before saving a note")
            return self._write_verdict(uid, row["verdict"], note_raw)

    def _get_state(self, match_id: str) -> MatchState:
        with self._lock:
            if match_id in self._state_cache:
                return self._state_cache[match_id]
        entry = self.entries.get(match_id)
        if entry is None or entry.tracks_path is None or not entry.tracks_path.is_file():
            raise ValueError(f"no tracks.parquet available for match {match_id!r}")
        st = MatchState.load(entry.dir)
        with self._lock:
            if len(self._state_cache) >= 3 and match_id not in self._state_cache:
                oldest = next(iter(self._state_cache))
                del self._state_cache[oldest]
            self._state_cache[match_id] = st
        return st

    def radar_slice(
        self,
        match_id: str,
        start_s: float,
        end_s: float,
        *,
        target_hz: float = 15.0,
    ) -> dict[str, Any]:
        if not (np.isfinite(start_s) and np.isfinite(end_s)):
            raise ValueError("start_s and end_s must be finite numbers")
        st = self._get_state(match_id)
        fps = float(st.meta.fps or 25.0)
        c0 = max(0.0, float(start_s) - self.pre_roll_s)
        c1 = max(c0 + 0.5, min(float(end_s) + self.post_roll_s, c0 + 90.0))
        f0 = max(0, round(c0 * fps) - 10)
        f1 = round(c1 * fps) + 10
        sub = st.tracks[(st.tracks["frame_idx"] >= f0) & (st.tracks["frame_idx"] <= f1)]
        step = max(1, round(fps / max(target_hz, 1.0)))

        frames_out: list[dict[str, Any]] = []
        if not sub.empty:
            held = hold_gaps(sub, 8)
            by_f = {int(f): g for f, g in held.groupby("frame_idx", observed=True)}
            sf_start = max(0, round(c0 * fps))
            sf_end = round(c1 * fps)
            for f_idx in range(sf_start, sf_end + 1, step):
                t_s = round(f_idx / fps, 3)
                g = by_f.get(f_idx)
                players: list[list[float | int]] = []
                ball_xy: list[float] | None = None
                if g is not None:
                    for r in g.itertuples():
                        if not (np.isfinite(r.x) and np.isfinite(r.y)):
                            continue
                        if r.role == Role.BALL.value:
                            ball_xy = [round(float(r.x), 2), round(float(r.y), 2)]
                        elif r.role in (Role.PLAYER.value, Role.GOALKEEPER.value, Role.REFEREE.value):
                            players.append(
                                [
                                    int(r.track_id),
                                    _TEAM_CODE.get(str(r.team), 3),
                                    _ROLE_CODE.get(str(r.role), 0),
                                    round(float(r.x), 2),
                                    round(float(r.y), 2),
                                ]
                            )
                frames_out.append({"t": t_s, "p": players, "b": ball_xy})

        entry = self.entries[match_id]
        p_starts = self.period_starts.get(match_id, {})
        return {
            "match_id": match_id,
            "home": entry.home or "HOME",
            "away": entry.away or "AWAY",
            "pitch_length": float(st.meta.pitch_length),
            "pitch_width": float(st.meta.pitch_width),
            "seq_start_s": round(float(start_s), 3),
            "seq_end_s": round(float(end_s), 3),
            "clip_start_s": round(c0, 3),
            "clip_end_s": round(c1, 3),
            "period_starts": {str(k): v for k, v in p_starts.items()},
            "frames": frames_out,
        }

    def filter_dataframe(self, params: dict[str, list[str]]) -> pd.DataFrame:
        def _one(key: str) -> str:
            vals = params.get(key)
            return vals[0].strip() if vals and vals[0].strip() else ""

        club = _one("club").lower()
        opp = _one("opponent").lower()
        mid = _one("match_id")
        label = _one("label")
        period = _one("period")
        verdict = _one("verdict")
        response = _one("response")
        flank = _one("flank")
        min_dur = _one("min_duration")
        min_pr = _one("min_pressers")
        max_def = _one("max_def_line")
        max_comp = _one("max_compactness")

        with self._lock:
            rows = [dict(self._rows_by_uid[u]) for u in self._uid_order]

        kept: list[dict[str, Any]] = []
        for r in rows:
            if club and club != "all" and r["club"].lower() != club:
                continue
            if opp and opp != "all" and r["opponent"].lower() != opp:
                continue
            if mid and mid != "all" and r["match_id"] != mid:
                continue
            if label and label != "all" and r["label"] != label:
                continue
            if period and period != "all" and str(r["period"]) != period:
                continue
            if verdict and verdict != "all":
                if verdict == "unmarked" and r["verdict"] != "":
                    continue
                if verdict == "has_clip" and not r["has_clip"]:
                    continue
                if verdict in ("right", "wrong", "unsure") and r["verdict"] != verdict:
                    continue
            if response and response != "all" and r["response"] != response:
                continue
            if flank and flank != "all" and r["flank"] != flank:
                continue
            if min_dur and r["duration_s"] < float(min_dur):
                continue
            if min_pr and (r["n_pressers"] is None or r["n_pressers"] < int(min_pr)):
                continue
            if max_def and (r["def_line_x"] is None or r["def_line_x"] > float(max_def)):
                continue
            if max_comp and (r["compactness_m"] is None or r["compactness_m"] > float(max_comp)):
                continue
            r["clip_path"] = r["clip_url"]
            kept.append({k: v for k, v in r.items() if not k.startswith("_")})

        if not kept:
            return pd.DataFrame(
                columns=[
                    "clip_index", "sequence_id", "match_id", "club", "opponent", "side", "period",
                    "match_clock", "clock", "start_s", "end_s", "clip_start_s", "clip_end_s",
                    "duration_s", "label", "verdict", "note", "evidence_summary", "clip_path",
                ]
            )
        df = pd.DataFrame(kept)
        df["clip_index"] = range(1, len(df) + 1)
        return df


def _is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _secrets_equal(a: str, b: str) -> bool:
    """Constant-time compare that accepts any text (`compare_digest` rejects non-ASCII `str`)."""
    return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def _cookie_digest(secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), b"footlytics-workspace-v1", hashlib.sha256).hexdigest()


def make_workspace_server(
    roots: Sequence[Path | str] | Path | str,
    *,
    pack_dirs: Sequence[Path | str] = (),
    video_dir: Path | str | None = None,
    reviewer: str = "analyst",
    host: str = "127.0.0.1",
    port: int = 0,
    token: str | None = None,
    profile: str | Path | TacticalProfile | None = None,
) -> ThreadingHTTPServer:
    """Create a multi-match Web Tactical Query & 2D Radar HTTP server."""
    clean_token = token.strip() if token and token.strip() else None
    if clean_token is None and not _is_loopback_host(host):
        raise ValueError(
            f"binding to {host or 'all interfaces'!r} exposes the write endpoints to the network: "
            "pass --token (or FOOTLYTICS_WORKSPACE_TOKEN), or bind to 127.0.0.1"
        )
    store = WorkspaceStore(
        roots,
        pack_dirs=pack_dirs,
        video_dir=video_dir,
        reviewer=reviewer,
        profile=profile,
    )
    expected_cookie = _cookie_digest(clean_token) if clean_token else None

    class Handler(BaseHTTPRequestHandler):
        server_version = "footlytics-workspace"

        def log_message(self, *args):
            pass

        def _send(
            self,
            status: int,
            body: bytes,
            ctype: str,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

        def _text(self, status: int, msg: str) -> None:
            self._send(status, msg.encode("utf-8"), "text/plain; charset=utf-8")

        def _authorize(self, query_params: dict[str, list[str]]) -> tuple[bool, dict[str, str]]:
            srv_port = self.server.server_address[1]
            if clean_token is None:
                if host in ("127.0.0.1", "localhost") and self.headers.get("Host") not in (
                    f"127.0.0.1:{srv_port}",
                    f"localhost:{srv_port}",
                ):
                    self._text(403, "forbidden host")
                    return False, {}
                return True, {}

            auth_hdr = self.headers.get("Authorization", "")
            if auth_hdr.startswith("Bearer "):
                # http.server decodes header bytes as latin-1; clients send the token as UTF-8
                bearer = auth_hdr[7:].strip().encode("latin-1", "replace").decode("utf-8", "replace")
                if _secrets_equal(bearer, clean_token):
                    return True, {}

            q_tok = (query_params.get("token") or [""])[0]
            if q_tok and _secrets_equal(q_tok, clean_token):
                cookie_val = f"{COOKIE_NAME}={expected_cookie}; HttpOnly; SameSite=Strict; Path=/"
                return True, {"Set-Cookie": cookie_val}

            raw_cookie = self.headers.get("Cookie", "")
            if raw_cookie and expected_cookie is not None:
                jar = cookies.SimpleCookie()
                with contextlib.suppress(cookies.CookieError):
                    jar.load(raw_cookie)
                morsel = jar.get(COOKIE_NAME)
                if morsel is not None and _secrets_equal(morsel.value, expected_cookie):
                    return True, {}

            self._text(401, "unauthorized: pass ?token=<secret> or Authorization: Bearer <secret>")
            return False, {}

        def _stream_media(self, media_path: Path) -> None:
            size = media_path.stat().st_size
            ctype = _TYPES.get(media_path.suffix.lower(), "application/octet-stream")
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
            with (
                contextlib.suppress(BrokenPipeError, ConnectionResetError),
                open(media_path, "rb") as f,
            ):
                f.seek(start)
                while left > 0:
                    chunk = f.read(min(CHUNK, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            qparams = parse_qs(parsed.query)
            ok, auth_headers = self._authorize(qparams)
            if not ok:
                return
            path = parsed.path

            if path == "/":
                snap = store.public_snapshot()
                blob = json.dumps(snap).replace("</", "<\\/")
                body = TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, blob).encode("utf-8")
                self._send(200, body, "text/html; charset=utf-8", extra_headers=auth_headers)
                return

            if path == "/api/state":
                self._json(200, store.public_snapshot())
                return

            if path == "/api/radar":
                try:
                    mid = (qparams.get("match_id") or [""])[0]
                    s0 = float((qparams.get("start_s") or ["0"])[0])
                    s1 = float((qparams.get("end_s") or [str(s0 + 5.0)])[0])
                    payload = store.radar_slice(mid, s0, s1)
                except (ValueError, KeyError, OverflowError) as e:
                    self._text(400, f"bad radar query: {e}")
                    return
                self._json(200, payload)
                return

            if path.startswith("/api/clip/"):
                uid = path[len("/api/clip/") :]
                mp4 = store.clip_path_for(uid)
                if mp4 is None or not mp4.is_file():
                    self._text(404, "clip not found")
                    return
                self._stream_media(mp4)
                return

            if path.startswith("/api/video/"):
                mid = path[len("/api/video/") :]
                vid = store.video_path_for(mid)
                if vid is None or not vid.is_file():
                    self._text(404, "match video not found")
                    return
                self._stream_media(vid)
                return

            if path == "/api/export/playlist.csv":
                try:
                    df = store.filter_dataframe(qparams)
                except ValueError as e:
                    self._text(400, f"invalid filter: {e}")
                    return
                csv_cols = [
                    "clip_index", "sequence_id", "match_id", "club", "opponent", "side", "period",
                    "match_clock", "clock", "start_s", "end_s", "clip_start_s", "clip_end_s",
                    "duration_s", "label", "verdict", "note", "evidence_summary", "clip_path",
                ]
                csv_bytes = df[csv_cols].to_csv(index=False).encode("utf-8")
                self._send(
                    200,
                    csv_bytes,
                    "text/csv; charset=utf-8",
                    extra_headers={"Content-Disposition": 'attachment; filename="playlist.csv"'},
                )
                return

            if path == "/api/export/sportscode.xml":
                import tempfile

                try:
                    df = store.filter_dataframe(qparams)
                except ValueError as e:
                    self._text(400, f"invalid filter: {e}")
                    return
                with tempfile.TemporaryDirectory() as td:
                    xml_p = write_sportscode_xml(df, Path(td) / "sportscode.xml")
                    xml_bytes = xml_p.read_bytes()
                self._send(
                    200,
                    xml_bytes,
                    "application/xml; charset=utf-8",
                    extra_headers={"Content-Disposition": 'attachment; filename="sportscode.xml"'},
                )
                return

            if path == "/api/export/profile.yaml":
                yaml_bytes = store.profile.to_yaml().encode("utf-8")
                self._send(
                    200,
                    yaml_bytes,
                    "application/x-yaml; charset=utf-8",
                    extra_headers={"Content-Disposition": f'attachment; filename="{store.profile.name}.yaml"'},
                )
                return

            if path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return

            self._text(404, "not found")

        def do_POST(self) -> None:
            parsed = urlsplit(self.path)
            ok, _ = self._authorize(parse_qs(parsed.query))
            if not ok:
                return
            if parsed.path not in ("/api/verdict", "/api/note", "/api/recompute"):
                self._text(404, "not found")
                return
            if self.headers.get_content_type() != "application/json":
                self._text(415, "Content-Type must be application/json")
                return
            try:
                n = int(self.headers.get("Content-Length", ""))
                if not 0 < n <= MAX_BODY:
                    raise ValueError("invalid body length")
                req = json.loads(self.rfile.read(n))
                if not isinstance(req, dict):
                    raise TypeError("JSON body must be an object")
                if parsed.path == "/api/verdict":
                    res = store.set_verdict(str(req.get("uid", "")), req.get("verdict"), req.get("note", ""))
                elif parsed.path == "/api/note":
                    res = store.set_note(str(req.get("uid", "")), req.get("note", ""))
                else:
                    persist = bool(req.get("persist", False))
                    prof_raw = req.get("profile", req)
                    if not isinstance(prof_raw, dict):
                        raise TypeError("profile must be an object")
                    prof_clean = {k: v for k, v in prof_raw.items() if k != "persist"}
                    prof = TacticalProfile.from_dict(prof_clean, base=store.profile)
                    res = store.recompute_with_profile(prof, persist=persist)
            except (ValueError, TypeError) as e:
                self._text(400, f"bad request: {e}")
                return
            except Exception as e:  # noqa: BLE001
                self._text(500, f"could not save: {e}")
                return
            self._json(200, res)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.workspace_store = store  # type: ignore[attr-defined]
    return server

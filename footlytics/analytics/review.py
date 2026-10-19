"""Analyst marks on detected sequences: the precision and recall numbers the labels are tuned against.

Marks are stored in match-state seconds (the sequences CSV's clock), never video seconds,
so they survive a re-encoded or re-cut video; the page adds `video_offset_s` to seek.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd

VERDICTS = ("right", "wrong", "unsure")
LABELS = ("low_block", "high_press")
TEAMS = ("HOME", "AWAY")
VERSION = 1


def sequence_ids(sequences: pd.DataFrame) -> list[str]:
    """Stable id per row: it survives re-sorting and reloads because it uses the start frame."""
    return [f"{r.label}-{r.team}-{int(r.period)}-{int(r.start_frame)}" for r in sequences.itertuples()]


def new_marks(reviewer: str, video: str, video_offset_s: float = 0.0) -> dict:
    return {"version": VERSION, "reviewer": reviewer, "video": video,
            "video_offset_s": float(video_offset_s), "sequences": {}, "missed": []}


def check_verdict(verdict, note) -> tuple[str, str]:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {', '.join(VERDICTS)}")
    if not isinstance(note, str):
        raise ValueError("note must be text")
    return verdict, note


def check_missed(entry) -> dict:
    """Return a clean missed-sequence entry or raise ValueError saying what is wrong."""
    if not isinstance(entry, dict):
        raise ValueError("missed entry must be an object")
    label, team, note = entry.get("label"), entry.get("team"), entry.get("note", "")
    if not isinstance(label, str) or not label:
        raise ValueError("label must be text")
    if team not in TEAMS:
        raise ValueError(f"team must be one of {', '.join(TEAMS)}")
    times = []
    for k in ("start_s", "end_s"):
        v = entry.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{k} must be a number")
        times.append(float(v))
    if times[1] <= times[0]:
        raise ValueError("end_s must be after start_s")
    if not isinstance(note, str):
        raise ValueError("note must be text")
    return {"label": label, "team": team, "start_s": times[0], "end_s": times[1], "note": note}


def validate_marks(doc) -> dict:
    if not isinstance(doc, dict) or doc.get("version") != VERSION:
        raise ValueError(f"marks file must be an object with version {VERSION}")
    if not isinstance(doc.get("reviewer"), str) or not isinstance(doc.get("video"), str):
        raise ValueError("reviewer and video must be text")
    off = doc.get("video_offset_s")
    if isinstance(off, bool) or not isinstance(off, (int, float)) or not math.isfinite(off):
        raise ValueError("video_offset_s must be a number")
    if not isinstance(doc.get("sequences"), dict) or not isinstance(doc.get("missed"), list):
        raise ValueError("sequences must be an object and missed a list")
    for m in doc["sequences"].values():
        if not isinstance(m, dict):
            raise ValueError("each sequence mark must be an object")
        check_verdict(m.get("verdict"), m.get("note"))
    for entry in doc["missed"]:
        check_missed(entry)
    return doc


def load_marks(path) -> dict:
    return validate_marks(json.loads(Path(path).read_text()))


def write_marks(path, doc: dict) -> None:
    """Atomic write: a crash or full disk leaves the previous file intact."""
    path = Path(path)
    tmp = path.with_name(f".tmp-{path.name}")
    try:
        with open(tmp, "w") as f:
            f.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def score_marks(marks: dict, sequences: pd.DataFrame) -> dict:
    """Per label: counts of verdicts, precision = right / (right + wrong), and missed entries.

    Marks for ids that are not in `sequences` (a stale file) are ignored.
    """
    labels = dict.fromkeys(list(sequences["label"]) + [m["label"] for m in marks.get("missed", [])])
    out = {l: {"n": 0, "marked": 0, "right": 0, "wrong": 0, "unsure": 0,
               "precision": None, "missed": 0} for l in labels}
    for sid, label in zip(sequence_ids(sequences), sequences["label"]):
        s = out[label]
        s["n"] += 1
        m = marks.get("sequences", {}).get(sid)
        if m:
            s["marked"] += 1
            s[m["verdict"]] += 1
    for m in marks.get("missed", []):
        out[m["label"]]["missed"] += 1
    for s in out.values():
        judged = s["right"] + s["wrong"]
        s["precision"] = s["right"] / judged if judged else None
    return out

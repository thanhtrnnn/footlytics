"""Calibration assist: find white pitch-line intersections on a frame so a person can label them.

Output: an annotated image (numbered lines and candidate points) and a JSON with candidates plus an
empty calibration template. The analyst copies the candidate ids they recognise into the template
with the matching pitch vertex id or landmark name (see footlytics/calibration.py).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import cv2
import numpy as np


def _pitch_mask(hsv: np.ndarray) -> np.ndarray:
    grass = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(grass)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return cv2.erode(mask, np.ones((7, 7), np.uint8))


def _merge_segments(segs: np.ndarray, angle_deg: float = 2.5, rho_px: float = 10.0, min_len: float = 120.0) -> list[dict]:
    lines: list[dict] = []
    for x1, y1, x2, y2 in segs:
        th = np.arctan2(y2 - y1, x2 - x1) % np.pi
        rho = float(np.array([-np.sin(th), np.cos(th)]) @ np.array([x1, y1]))
        length = float(np.hypot(x2 - x1, y2 - y1))
        for ln in lines:
            dth = min(abs(ln["th"] - th), np.pi - abs(ln["th"] - th))
            if dth < np.deg2rad(angle_deg) and abs(ln["rho"] - rho) < rho_px:
                ln["segs"].append((x1, y1, x2, y2)); ln["len"] += length
                break
        else:
            lines.append({"th": th, "rho": rho, "segs": [(x1, y1, x2, y2)], "len": length})
    lines = [ln for ln in lines if ln["len"] > min_len]
    lines.sort(key=lambda ln: -ln["len"])
    for ln in lines:
        pts = np.array(ln["segs"], dtype=np.float32).reshape(-1, 2)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        ln["p"] = (float(x0), float(y0)); ln["d"] = (float(vx), float(vy))
        ln["extent"] = [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]
    return lines


def detect_pitch_intersections(frame: np.ndarray, person_boxes: np.ndarray | None = None) -> dict:
    """Return {"lines": [...], "points": [{"id", "x", "y", "lines": [i, j]}]} for one BGR frame."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pitch = _pitch_mask(hsv)
    keep = np.full((h, w), 255, np.uint8)
    if person_boxes is not None:
        for x1, y1, x2, y2 in np.asarray(person_boxes).astype(int):
            cv2.rectangle(keep, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), 0, -1)
    white = cv2.inRange(hsv, (0, 0, 140), (180, 80, 255)) & pitch & keep
    segs = cv2.HoughLinesP(white, 1, np.pi / 360, threshold=50, minLineLength=50, maxLineGap=12)
    segs = segs.reshape(-1, 4) if segs is not None else np.zeros((0, 4), int)
    lines = _merge_segments(segs)
    points = []
    for (i, a), (j, b) in itertools.combinations(enumerate(lines), 2):
        d1, d2 = np.array(a["d"]), np.array(b["d"])
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 0.05:
            continue
        t = np.linalg.solve(A, np.array(b["p"]) - np.array(a["p"]))
        q = np.array(a["p"]) + d1 * t[0]
        if 0 <= q[0] < w and 0 <= q[1] < h and pitch[int(q[1]), int(q[0])] > 0:
            points.append({"id": len(points), "x": float(q[0]), "y": float(q[1]), "lines": [i, j]})
    return {"lines": [{"id": i, "p": ln["p"], "d": ln["d"], "extent": ln["extent"], "len": ln["len"]} for i, ln in enumerate(lines)],
            "points": points}


def annotate(frame: np.ndarray, res: dict) -> np.ndarray:
    vis = frame.copy()
    for ln in res["lines"]:
        p, d = np.array(ln["p"]), np.array(ln["d"])
        a, b = (p - d * 3000).astype(int), (p + d * 3000).astype(int)
        cv2.line(vis, tuple(a), tuple(b), (255, 0, 255), 1)
        e = ln["extent"]
        cv2.putText(vis, f"L{ln['id']}", (int((e[0] + e[2]) / 2), int((e[1] + e[3]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for pt in res["points"]:
        cv2.circle(vis, (int(pt["x"]), int(pt["y"])), 5, (0, 0, 255), -1)
        cv2.putText(vis, f"P{pt['id']}", (int(pt["x"]) + 6, int(pt["y"]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return vis


def write_assist(frame: np.ndarray, out_prefix: str | Path, person_boxes: np.ndarray | None = None, frame_index: int = 0) -> tuple[Path, Path]:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    res = detect_pitch_intersections(frame, person_boxes)
    img_path = out_prefix.with_suffix(".jpg")
    json_path = out_prefix.with_suffix(".json")
    cv2.imwrite(str(img_path), annotate(frame, res))
    payload = {
        "frame": frame_index,
        "candidates": [{"id": p["id"], "x": round(p["x"], 1), "y": round(p["y"], 1), "lines": p["lines"]} for p in res["points"]],
        "lines": res["lines"],
        "template": {"frame": frame_index, "points": []},
        "how_to": "Open the .jpg, pick candidate ids you recognise, add {\"vertex\": <1-32 or landmark name>, \"x\", \"y\"} entries to template.points, save as calib.json and pass --calib.",
    }
    json_path.write_text(json.dumps(payload, indent=1))
    return img_path, json_path

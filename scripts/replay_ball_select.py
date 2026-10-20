"""Replay the ball selection offline on candidates captured by `capture_ball_candidates.py`.

It calls the pipeline's own functions (`on_pitch`, `BallSelector.flags`, `link_ball_rows`), so a
replay gives the numbers a bench run would, minus the detector and calibration. Ball numbers are
the bench's: on each GT-ball frame our box centre within 20 px of the GT centre is a hit, a box
farther away is wrong, no box is a miss.

    python scripts/replay_ball_select.py cache/ball_cands_train.pkl [key=value ...]

`key=value` overrides `PipelineConfig` fields (ball_link_break_cost=0.4 ...).
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import fields

import numpy as np

from footlytics.geometry.pitch import Pitch
from footlytics.pipeline.radar import BallSelector, PipelineConfig, link_ball_rows, on_pitch
from footlytics.state.schema import Role

HIT_PX = 20.0


def load(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def replay_clip(clip: dict, cfg: PipelineConfig, ball_floor: float = 0.0) -> dict[int, tuple]:
    """frame -> (x, y, w, h, conf) of the picked ball box, as `run_frames` would write it."""
    pitch = Pitch(105.0, 68.0)
    fps = clip["fps"]
    sel = BallSelector(fps, cfg.ball_max_speed, cfg.ball_slack_m, cfg.ball_reacquire_s,
                       still_s=cfg.ball_still_s)
    log = []
    for i in range(clip["n_frames"]):
        dets, is_b, xy = clip["frames"][i]
        if xy is None or not len(dets):
            continue
        if ball_floor:       # the detector's `ball_conf`: candidates below it never exist
            ok = ~(is_b & (dets[:, 4] < ball_floor))
            dets, is_b, xy = dets[ok], is_b[ok], xy[ok]
        roles = [Role.BALL.value if b else Role.PLAYER.value for b in is_b]
        keep = on_pitch(xy.astype(float), roles, pitch, cfg.pitch_margin_m, cfg.ball_margin_m)
        dets, xy, is_b = dets[keep], xy[keep].astype(float), is_b[keep]
        bi, people = np.flatnonzero(is_b), np.flatnonzero(~is_b)
        if len(bi):
            still, decoy = sel.flags(i, xy[bi], xy[people])
            log.append((i, xy[bi].copy(), dets[bi].astype(float).copy(), still, decoy))
    rows, _ = link_ball_rows(log, fps, 1, cfg)
    return {r["frame_idx"]: (r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"], r["det_conf"])
            for r in rows}


def score(clip: dict, picks: dict) -> dict:
    n = hit = wrong = 0
    for f, (x, y, w, h) in clip["gt_ball"].items():
        n += 1
        p = picks.get(f)
        if p is None:
            continue
        d = np.hypot(p[0] + p[2] / 2 - (x + w / 2), p[1] + p[3] / 2 - (y + h / 2))
        hit += d <= HIT_PX
        wrong += d > HIT_PX
    return {"n": n, "hit": hit, "wrong": wrong}


def pooled(data: dict, cfg: PipelineConfig, **kw) -> dict:
    t = {"n": 0, "hit": 0, "wrong": 0}
    for clip in data["clips"].values():
        s = score(clip, replay_clip(clip, cfg, **kw))
        for k in t:
            t[k] += s[k]
    return {**t, "recall": t["hit"] / t["n"], "wrong_share": t["wrong"] / t["n"],
            "none": 1 - (t["hit"] + t["wrong"]) / t["n"]}


def parse_overrides(args: list[str]) -> dict:
    types = {f.name: f.type for f in fields(PipelineConfig)}
    out = {}
    for a in args:
        k, v = a.split("=")
        out[k] = {"bool": lambda s: s.lower() in ("1", "true"), "int": int}.get(
            str(types[k]), float)(v) if str(types[k]) in ("bool", "int", "float") else float(v)
    return out


if __name__ == "__main__":
    data = load(sys.argv[1])
    cfg = PipelineConfig(**{"ball_link": True, **parse_overrides(sys.argv[2:])})
    r = pooled(data, cfg)
    print(f"recall {r['recall']:.3f}  wrong {r['wrong_share']:.3f}  none {r['none']:.3f}  (n={r['n']})")

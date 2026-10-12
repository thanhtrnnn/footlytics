"""SoccerNet Game State Reconstruction (GSR) <-> Match State.

GSR pitch coordinates are the Match State's: metres from the centre spot, x along the length
(negative = the goal on the camera's left), y negative towards the far touchline. Positions
pass through unchanged. What needs care is identity:

- frames: GSR images are 1-based (`000001.jpg`), Match State `frame_idx` is 0-based;
- image ids: `f"{split digit}{clip:03d}{frame:06d}"`, e.g. "2033000001";
- teams: GSR names a team by the goal it defends in the clip ("left"/"right"), not home/away.
  Our side is decided from our own positions (`team_sides`), never from ground truth.

GS-HOTA matches on the bottom-middle point only, so predictions carry it in all three
bottom-point slots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...state.schema import MatchMeta, MatchState, Role, Team

SPLIT_DIGIT = {"train": 1, "valid": 2, "test": 3, "challenge": 4}
CATEGORY_ID = {"player": 1, "goalkeeper": 2, "referee": 3, "ball": 4, "other": 7}
MIN_VERSION = (1, 3)

_ROLE = {"player": Role.PLAYER.value, "goalkeeper": Role.GOALKEEPER.value,
         "referee": Role.REFEREE.value, "ball": Role.BALL.value}
_TEAM = {"left": Team.HOME.value, "right": Team.AWAY.value}
_PLAYING = (Role.PLAYER.value, Role.GOALKEEPER.value)


def image_id(split: str, seq: str, frame_idx: int) -> str:
    return f"{SPLIT_DIGIT[split]}{int(seq.split('-')[-1]):03d}{frame_idx + 1:06d}"


def check_version(labels: dict) -> str:
    v = str(labels["info"]["version"])
    if tuple(int(p) for p in v.split(".")) < MIN_VERSION:
        raise ValueError(f"GSR labels v{v} for {labels['info'].get('name')}: need >= 1.3 "
                         "(earlier versions have the old bbox_pitch and team labels)")
    return v


def gt_to_state(labels: dict) -> MatchState:
    """Ground truth as a Match State: left team -> HOME, right -> AWAY, officials -> OFFICIAL."""
    info = labels["info"]
    version = check_version(labels)
    fps = float(info.get("frame_rate", 25))
    frame_of = {img["image_id"]: int(Path(img["file_name"]).stem) - 1 for img in labels["images"]}
    rows = []
    for a in labels["annotations"]:
        if a.get("supercategory") != "object" or a.get("bbox_pitch") is None:
            continue
        attrs = a["attributes"]
        role = _ROLE.get(attrs.get("role"), Role.OTHER.value)
        if role == Role.REFEREE.value:
            team = Team.OFFICIAL.value
        else:
            team = _TEAM.get(attrs.get("team"), Team.UNKNOWN.value)
        jersey = attrs.get("jersey")
        f = frame_of[a["image_id"]]
        bi, bp = a.get("bbox_image") or {}, a["bbox_pitch"]
        rows.append(dict(
            frame_idx=f, period=1, timestamp=f / fps, track_id=int(a["track_id"]), role=role,
            team=team, jersey=float(jersey) if jersey not in (None, "") else np.nan,
            bbox_x=bi.get("x"), bbox_y=bi.get("y"), bbox_w=bi.get("w"), bbox_h=bi.get("h"),
            det_conf=1.0, x=bp["x_bottom_middle"], y=bp["y_bottom_middle"],
            z=0.0 if role == Role.BALL.value else np.nan))
    meta = MatchMeta(match_id=info["name"], fps=fps, source_type="broadcast",
                     provenance={"source": "gsr-ground-truth", "gsr_version": version})
    return MatchState(meta, pd.DataFrame(rows))


def team_sides(tracks: pd.DataFrame) -> dict[str, str]:
    """HOME/AWAY -> "left"/"right" for one clip, from our own positions.

    A team defends the goal its keeper stands in front of, so a goalkeeper sighting decides;
    without one, the team's mean x. With both teams the lower x is left; a lone team goes by
    the sign of its x.
    """
    people = tracks[tracks["role"].isin(_PLAYING)
                    & tracks["team"].isin([Team.HOME.value, Team.AWAY.value])]
    xs: dict[str, float] = {}
    for team, g in people.groupby("team", observed=True):
        gk = g[g["role"] == Role.GOALKEEPER.value]
        xs[str(team)] = float((gk if len(gk) else g)["x"].mean())
    if len(xs) == 2:
        home_left = xs[Team.HOME.value] <= xs[Team.AWAY.value]
        return {Team.HOME.value: "left" if home_left else "right",
                Team.AWAY.value: "right" if home_left else "left"}
    return {t: ("left" if x < 0 else "right") for t, x in xs.items()}


def _num(v) -> float | None:
    return float(v) if v is not None and np.isfinite(v) else None


def state_to_predictions(state: MatchState, split: str, use_jersey: bool = False) -> dict:
    """A Match State as one GSR predictions file (`{"predictions": [...]}`).

    Team and jersey are null for anyone not playing, as the evaluator requires; jersey is
    null throughout unless `use_jersey` (we have no shirt-number reader yet).
    """
    seq = state.meta.match_id
    sides = team_sides(state.tracks)
    preds = []
    for r in state.tracks.itertuples(index=False):
        x, y = _num(r.x), _num(r.y)
        if x is None or y is None:
            continue
        role = str(r.role) if str(r.role) in CATEGORY_ID else "other"
        team = sides.get(str(r.team)) if role in _PLAYING else None
        jersey = (str(int(r.jersey)) if use_jersey and role == Role.PLAYER.value
                  and _num(r.jersey) is not None else None)
        box = [_num(v) for v in (r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h)]
        preds.append({
            "id": str(len(preds)), "image_id": image_id(split, seq, int(r.frame_idx)),
            "video_id": seq.split("-")[-1], "track_id": int(r.track_id),
            "supercategory": "object", "category_id": CATEGORY_ID[role],
            "confidence": _num(r.det_conf) if _num(r.det_conf) is not None else 1.0,
            "bbox_image": (dict(zip(("x", "y", "w", "h"), box))
                           if all(v is not None for v in box) else None),
            "bbox_pitch": {f"{a}_bottom_{s}": v for s in ("left", "middle", "right")
                           for a, v in (("x", x), ("y", y))},
            "attributes": {"role": role, "team": team, "jersey": jersey},
        })
    return {"predictions": preds}

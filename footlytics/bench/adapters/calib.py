"""Our cameras -> SN-Calibration prediction files; GSR pitch lines -> SN-Calibration labels.

GSR pitch annotations already use the SN-Calibration line format (line name -> points
normalised to [0, 1]), so a GSR clip becomes calibration ground truth frame by frame.
"""
from __future__ import annotations

import json
from pathlib import Path

CAMERA_KEYS = ("pan_degrees", "tilt_degrees", "roll_degrees", "position_meters",
               "x_focal_length", "y_focal_length", "principal_point", "radial_distortion",
               "tangential_distortion", "thin_prism_distortion")


def _plain(v):
    if hasattr(v, "tolist"):                  # numpy scalar or array -> builtin
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return float(v)


def camera_json(cam: dict) -> dict:
    return {k: _plain(cam[k]) for k in CAMERA_KEYS}


def write_camera(path: str | Path, cam: dict) -> None:
    Path(path).write_text(json.dumps(camera_json(cam)))


#: GSR spells one class differently from the evaluator's pitch model, which has a trailing space
#: in "Goal left post left " (upstream typo); the evaluator raises KeyError on the GSR spelling.
_EVALUATOR_NAMES = {"Goal left post left": "Goal left post left "}

_SWAP = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


def mirror(lines: dict) -> dict:
    """Labels seen through the centre spot: left <-> right, top <-> bottom."""
    return {" ".join(_SWAP.get(w, w) for w in name.split(" ")): v for name, v in lines.items()}


def gsr_line_frames(labels: dict) -> dict[int, dict]:
    """0-based frame -> SN-Calibration lines, for frames whose pitch is labelled."""
    frame_of = {img["image_id"]: int(Path(img["file_name"]).stem) - 1
                for img in labels["images"] if img.get("has_labeled_pitch")}
    out = {}
    for a in labels["annotations"]:
        if a.get("supercategory") == "pitch" and a["image_id"] in frame_of:
            lines = {_EVALUATOR_NAMES.get(k, k): v for k, v in (a.get("lines") or {}).items() if v}
            if lines:
                out[frame_of[a["image_id"]]] = lines
    return out

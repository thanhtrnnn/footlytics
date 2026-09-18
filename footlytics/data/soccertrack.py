"""Loader for SoccerTrack v2 -- ten full matches shot on BePro panoramic cameras.

This is the closest public dataset to what Footlytics is being built for, so it
is the benchmark that actually counts. Fixed panoramic rig spanning the whole
pitch, 3840x1504 at 25 fps, with complete per-frame ground truth: all 22 players
in every frame, plus pitch coordinates, jersey identities, roles and team sides.

    https://huggingface.co/datasets/atomscott/soccertrack-v2   (CC-BY 4.0)

The image -> pitch chain
------------------------
Not a single homography. The panorama is a genuinely distorted view, and the
dataset ships explicit fisheye coefficients alongside the homography:

    raw pixel
      -> cv2.fisheye.undistortPoints(K, D, P=Knew)
      -> H^-1
      -> metres, CORNER origin
      -> subtract (L/2, W/2)
      -> metres, centre origin  [our convention]

Verified on the released ground truth: 99.5% of annotated feet land inside the
pitch rectangle through this chain, versus 0-3% through any other combination.

Two traps
---------
* The dataset README states the origin is the centre circle. The shipped
  homography does not: it outputs a CORNER origin (x 0..105, y 0..68). Taking
  the README at face value puts every position 52.5 m out, and nothing about
  the resulting numbers looks obviously wrong. Hence `to_centre_origin`.
* `H` maps pitch -> undistorted image, so image -> pitch needs `H^-1`.
  Applying H the wrong way round yields values in the thousands, which is at
  least loud rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..state.schema import MatchMeta, MatchState, Role, Team, TeamInfo

MOT_COLUMNS = ["frame", "track_id", "x", "y", "w", "h", "conf", "cls", "vis", "extra"]

PANORAMA_SIZE = (3840, 1504)
FPS = 25.0

# Pitch dimensions are a PER-VENUE MEASUREMENT, not a constant.
#
# The length is confirmed: the dataset's own middle-line annotation projects to
# x = 52.49 +- 0.006 through this calibration, so 105 m is right.
#
# The width is NOT 68 m, and assuming it was produced a real, silent error:
# 2.14% of ground-truth positions fell outside the touchline, systematically and
# by up to 11 m, and the projected touchline visibly sat inside the painted one.
# Player occupancy spans 90.5 m of the 105 m length (86%) but 72.4 m of an
# assumed 68 m width (106%) -- impossible, and enough on its own to prove the
# number wrong. Scaling by the length coverage implies roughly 84 m.
#
# The width was then read directly off the video: projecting candidate
# touchlines onto a brightened frame, the painted near touchline lands on
# y = 76, between y = 74 and y = 78. Occupancy scaling had suggested 84, which
# overshot -- a reminder that an inferred number should be checked against the
# pixels whenever the pixels are available.
#
# Uncertainty is asymmetric. The near touchline is sharp, but at the FAR
# touchline the projection compresses several metres into a few pixels:
# y = -3, 0, +3 and +6 are visually indistinguishable there. So the far edge is
# only known to a few metres, and any metric involving the far side of the pitch
# inherits that. This is inherent to filming from one touchline, and is why
# TRACAB uses stereo pairs on both sides.
PITCH_L = 105.0
PITCH_W = 76.0


@dataclass
class SoccerTrackCalibration:
    """The dataset's own calibration: fisheye undistortion plus a homography."""

    K: np.ndarray
    D: np.ndarray
    Knew: np.ndarray
    H: np.ndarray                      # pitch -> undistorted image
    to_centre_origin: bool = True
    #: Source resolution, so this can stand in for geometry.Calibration in the
    #: pipeline, which checks it against the video before processing.
    image_size: tuple[int, int] = (3840, 1906)

    @classmethod
    def load(cls, raw_dir: str | Path, match_id: str | int) -> "SoccerTrackCalibration":
        d = Path(raw_dir)
        z = np.load(d / f"{match_id}_camera_intrinsics.npz", allow_pickle=True)
        H = np.load(d / f"{match_id}_homography.npy")
        return cls(K=z["K"], D=z["D"], Knew=z["Knew"], H=H)

    def image_to_pitch(self, pts) -> np.ndarray:
        import cv2

        p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        und = cv2.fisheye.undistortPoints(p, self.K, self.D, P=self.Knew).reshape(-1, 2)
        hom = np.hstack([und, np.ones((len(und), 1))]) @ np.linalg.inv(self.H).T
        out = hom[:, :2] / hom[:, 2:3]
        if self.to_centre_origin:
            out = out - np.array([PITCH_L / 2, PITCH_W / 2])
        return out

    def feet_to_pitch(self, bboxes) -> np.ndarray:
        """Ground-plane position from the bottom centre of each box."""
        b = np.asarray(bboxes, np.float64).reshape(-1, 4)
        feet = np.stack([b[:, 0] + b[:, 2] / 2.0, b[:, 1] + b[:, 3]], axis=1)
        return self.image_to_pitch(feet)


def load_mot_groundtruth(
    mot_txt: str | Path,
    calib: SoccerTrackCalibration,
    match_id: str = "SOCCERTRACK",
    fps: float = FPS,
    max_frames: Optional[int] = None,
) -> MatchState:
    """Build a MatchState from the released MOT boxes -- perfect perception.

    Useful as an upper bound: run analytics on this and whatever it produces is
    what our own pipeline would produce with a flawless detector and tracker.
    Any gap between the two is ours, not football's.

    Team labels are NOT in the MOT file (its class column is -1); they live in
    the much larger GSR JSONs. Everything here is left as UNKNOWN rather than
    guessed.
    """
    df = pd.read_csv(mot_txt, names=MOT_COLUMNS)
    if max_frames is not None:
        df = df[df["frame"] < max_frames]

    xy = calib.feet_to_pitch(df[["x", "y", "w", "h"]].to_numpy())
    out = pd.DataFrame({
        "frame_idx": df["frame"].astype("int32"),
        "period": 1,
        "timestamp": df["frame"] / fps,
        "track_id": df["track_id"].astype("int32"),
        "role": Role.PLAYER.value,
        "team": Team.UNKNOWN.value,
        "jersey": np.nan,
        "bbox_x": df["x"], "bbox_y": df["y"], "bbox_w": df["w"], "bbox_h": df["h"],
        "det_conf": 1.0,
        "x": xy[:, 0], "y": xy[:, 1], "z": np.nan,
    })

    meta = MatchMeta(
        match_id=match_id, fps=fps,
        pitch_length=PITCH_L, pitch_width=PITCH_W,
        source_type="fixed_panoramic",
        competition="SoccerTrack v2 (university, Japan)",
        home=TeamInfo("Team A", "A"), away=TeamInfo("Team B", "B"),
        provenance={"dataset": "atomscott/soccertrack-v2", "annotation": "mot ground truth"},
    )
    state = MatchState(meta, out)
    from ..analytics.kinematics import add_kinematics
    state = MatchState(meta, add_kinematics(state.tracks, fps))
    return state


def _removed_infer_teams_from_kickoff():
    """Deliberately removed. See the note below.

    This used to split tracks into two teams by which half of the pitch they
    occupied at frame 0. On the released MOT clips it produced a 17/5 split,
    because those clips are four-minute excerpts from mid-match, not kick-offs.

    The lesson is worth keeping: at any moment other than a genuine kick-off,
    position carries almost no information about team. During an attack most of
    both teams are in the same third. Balancing the split to 11/11 would not
    have helped either -- it would have produced a confidently wrong 11/11.

    Real team labels come from exactly two places:
      * the dataset's GSR JSONs (gsr/<match>/<half>.json), which carry team side
        per tracklet, or
      * footlytics.perception.teams.TeamClassifier run on the actual video,
        which is what the product does.
    """
    raise NotImplementedError(_removed_infer_teams_from_kickoff.__doc__)


def score_against_mot(
    state: MatchState,
    mot_txt: str | Path,
    calib: SoccerTrackCalibration,
    max_match_dist: float = 5.0,
) -> dict:
    """Score a pipeline MatchState against SoccerTrack v2's MOT ground truth.

    Frame-indexed, so unlike the Alfheim scorer no timestamp alignment is needed.
    And because every one of the 22 players is annotated in every frame, both
    recall AND precision are meaningful here -- which is exactly what Alfheim,
    with only one team wearing sensors, could never give us.
    """
    from scipy.optimize import linear_sum_assignment

    gt = pd.read_csv(mot_txt, names=MOT_COLUMNS)
    gt_xy = calib.feet_to_pitch(gt[["x", "y", "w", "h"]].to_numpy())
    gt = gt.assign(px=gt_xy[:, 0], py=gt_xy[:, 1])

    det = state.players
    errors, recalls, precisions = [], [], []
    gt_to_track: dict[int, int] = {}
    switches = 0

    gt_by_frame = {f: g for f, g in gt.groupby("frame")}
    for frame_idx, d in det.groupby("frame_idx"):
        g = gt_by_frame.get(int(frame_idx))
        if g is None or g.empty or d.empty:
            continue
        G = g[["px", "py"]].to_numpy()
        Dt = d[["x", "y"]].to_numpy()
        dist = np.linalg.norm(Dt[:, None, :] - G[None, :, :], axis=-1)
        cost = np.where(dist <= max_match_dist, dist, 1e6)
        ri, ci = linear_sum_assignment(cost)
        n = 0
        for a, b in zip(ri, ci):
            if cost[a, b] >= 1e6:
                continue
            n += 1
            errors.append(float(dist[a, b]))
            gid = int(g["track_id"].to_numpy()[b])
            tid = int(d["track_id"].to_numpy()[a])
            if gid in gt_to_track and gt_to_track[gid] != tid:
                switches += 1
            gt_to_track[gid] = tid
        recalls.append(n / len(G))
        precisions.append(n / len(Dt))

    if not errors:
        return {"error": "nothing matched -- check the calibration or the frame indexing"}

    e = np.array(errors)
    return {
        "matched": len(e),
        "recall": float(np.mean(recalls)),
        "precision": float(np.mean(precisions)),
        "err_median_m": float(np.median(e)),
        "err_p90_m": float(np.percentile(e, 90)),
        "err_p99_m": float(np.percentile(e, 99)),
        "id_switches": switches,
        "gt_players_seen": len(gt_to_track),
    }

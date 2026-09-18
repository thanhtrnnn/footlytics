"""First real measurement: our detector against SoccerTrack v2 ground truth.

Everything else in this repo is checked against synthetic data or against the
released annotations replayed as if they were our output. This is the first time
our own perception touches real pixels and is scored on them.

Detection only, sampled frames -- tracking quality needs consecutive frames and
is measured separately. Because every one of the 22 players is annotated in
every frame, BOTH recall and precision are meaningful here.
"""
import sys, time; sys.path.insert(0, ".")
import numpy as np, pandas as pd, cv2
from scipy.optimize import linear_sum_assignment
from footlytics.perception.detect import Detector, DetectorConfig
from footlytics.data.soccertrack import SoccerTrackCalibration, MOT_COLUMNS, PITCH_L, PITCH_W

CLIP = "data/raw/soccertrack/clips/117092.mp4"
D = "data/raw/soccertrack/117092"
FRAMES = list(range(200, 5800, 140))          # 40 frames spread over the clip
MAX_MATCH_M = 3.0

cal = SoccerTrackCalibration.load(D, 117092)
gt = pd.read_csv(f"{D}/117092.txt", names=MOT_COLUMNS)
gt_xy = cal.feet_to_pitch(gt[["x", "y", "w", "h"]].to_numpy())
gt = gt.assign(px=gt_xy[:, 0], py=gt_xy[:, 1])

det = Detector("weights/yolo_v8x6_finetuned.pt",
               DetectorConfig(tile=1280, overlap=0.25, conf=0.25,
                              ball_conf=0.10, device="mps", half=False))

rows, t0 = [], time.time()
cap = cv2.VideoCapture(CLIP)
for i, f in enumerate(FRAMES):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, bgr = cap.read()
    if not ok:
        continue
    d = det.detect(bgr[:, :, ::-1])
    g = gt[gt.frame == f]
    if not len(g):
        continue
    G = g[["px", "py"]].to_numpy()
    if len(d):
        P = cal.feet_to_pitch(d[:, :4])
        on = np.array([np.isfinite(p).all() and abs(p[0]) <= PITCH_L / 2 + 6
                       and abs(p[1]) <= PITCH_W / 2 + 6 for p in P])
        P = P[on]
    else:
        P = np.empty((0, 2))
    n_match, errs = 0, []
    if len(P):
        dist = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=-1)
        cost = np.where(dist <= MAX_MATCH_M, dist, 1e6)
        ri, ci = linear_sum_assignment(cost)
        for a, b in zip(ri, ci):
            if cost[a, b] < 1e6:
                n_match += 1
                errs.append(dist[a, b])
    rows.append(dict(frame=f, n_det=len(P), n_gt=len(G), matched=n_match,
                     recall=n_match / len(G), precision=n_match / max(len(P), 1),
                     err=np.median(errs) if errs else np.nan))
    if i % 5 == 0:
        print(f"  {i+1}/{len(FRAMES)} frame {f}: {len(P)} det / {len(G)} gt, "
              f"{n_match} matched", flush=True)
cap.release()

r = pd.DataFrame(rows)
el = time.time() - t0
print("\n" + "=" * 62)
print(f"{len(r)} frames, {el/max(len(r),1):.1f}s per frame on MPS")
print(f"  detections per frame : median {r.n_det.median():.0f}  (ground truth 22)")
print(f"  RECALL               : {r.recall.mean():.1%}")
print(f"  PRECISION            : {r.precision.mean():.1%}")
print(f"  position error       : median {r.err.median():.2f} m")
print(f"  frames with >=20/22  : {(r.matched >= 20).mean():.0%}")
print(f"  worst frame          : {r.recall.min():.0%} recall")
r.to_csv("data/out/detector_score_real.csv", index=False)
print("\nwrote data/out/detector_score_real.csv")

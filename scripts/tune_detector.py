"""Recall is the bottleneck (66% on real footage). What cheaply raises it?

Position error is already 0.18 m, so nothing here is about accuracy -- it is
purely about the players the detector never sees. On a floodlit night match with
48-pixel players the plausible levers are the confidence gate, image brightness,
and how much the tiling zooms each player.
"""
import sys, time; sys.path.insert(0, ".")
import numpy as np, pandas as pd, cv2
from scipy.optimize import linear_sum_assignment
from footlytics.perception.detect import Detector, DetectorConfig
from footlytics.data.soccertrack import SoccerTrackCalibration, MOT_COLUMNS, PITCH_L, PITCH_W

D = "data/raw/soccertrack/117092"
FRAMES = list(range(300, 5800, 500))
cal = SoccerTrackCalibration.load(D, 117092)
gt = pd.read_csv(f"{D}/117092.txt", names=MOT_COLUMNS)
g_xy = cal.feet_to_pitch(gt[["x", "y", "w", "h"]].to_numpy())
gt = gt.assign(px=g_xy[:, 0], py=g_xy[:, 1])

cap = cv2.VideoCapture("data/raw/soccertrack/clips/117092.mp4")
frames = {}
for f in FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, b = cap.read()
    if ok: frames[f] = b[:, :, ::-1].copy()
cap.release()
print(f"{len(frames)} frames loaded\n")

def evaluate(det, brighten):
    rec, prec, errs = [], [], []
    for f, img in frames.items():
        im = cv2.convertScaleAbs(img, alpha=brighten, beta=8) if brighten != 1.0 else img
        d = det.detect(im)
        G = gt[gt.frame == f][["px", "py"]].to_numpy()
        P = cal.feet_to_pitch(d[:, :4]) if len(d) else np.empty((0, 2))
        if len(P):
            ok = np.array([np.isfinite(p).all() and abs(p[0]) <= PITCH_L/2+6
                           and abs(p[1]) <= PITCH_W/2+6 for p in P])
            P = P[ok]
        n = 0
        if len(P) and len(G):
            dist = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=-1)
            cost = np.where(dist <= 3.0, dist, 1e6)
            ri, ci = linear_sum_assignment(cost)
            for a, b in zip(ri, ci):
                if cost[a, b] < 1e6:
                    n += 1; errs.append(dist[a, b])
        rec.append(n / max(len(G), 1)); prec.append(n / max(len(P), 1))
    return np.mean(rec), np.mean(prec), (np.median(errs) if errs else np.nan)

print(f"{'tile':>6}{'conf':>7}{'bright':>8}{'recall':>9}{'precision':>11}{'err_m':>8}{'sec/fr':>8}")
print("-" * 57)
for tile in (1280, 960):
    for conf in (0.25, 0.15, 0.08):
        for bright in (1.0, 2.0):
            det = Detector("weights/yolo_v8x6_finetuned.pt",
                           DetectorConfig(tile=tile, overlap=0.25, conf=conf,
                                          ball_conf=conf, device="mps", half=False))
            t0 = time.time(); r, p, e = evaluate(det, bright); dt = (time.time()-t0)/len(frames)
            print(f"{tile:>6}{conf:>7.2f}{bright:>8.1f}{r:>8.1%}{p:>10.1%}"
                  f"{e:>8.2f}{dt:>8.1f}", flush=True)

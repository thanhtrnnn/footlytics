"""Synthetic-camera check for the calibration math.

Builds a known pinhole camera above a pitch, projects the landmarks into it,
then asks Calibration to recover the mapping from the pixels alone.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from footlytics.geometry.pitch import DEFAULT_PITCH
from footlytics.geometry.homography import Calibration, fit_homography, apply_homography

W, H_PX = 3840, 2160
rng = np.random.default_rng(0)


def make_camera(cam_pos, fx=2600.0):
    C = np.array(cam_pos, float)
    fwd = -C / np.linalg.norm(C)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    K = np.array([[fx, 0, W / 2], [0, fx, H_PX / 2], [0, 0, 1.0]])

    def project(pitch_xy):
        P = np.asarray(pitch_xy, float).reshape(-1, 2)
        Xw = np.hstack([P, np.zeros((len(P), 1))])
        Xc = (Xw - C) @ R.T
        uv = (Xc @ K.T)
        return uv[:, :2] / uv[:, 2:3]
    return project


lm = DEFAULT_PITCH.landmarks()
names = sorted(lm)
pitch_pts = np.array([lm[n] for n in names])

print("=" * 62)
print("1. exact pinhole camera, 8 well-spread landmarks")
project = make_camera([0, -75, 20])
use = ["corner_LT", "corner_RT", "corner_LB", "corner_RB",
       "halfway_T", "halfway_B", "pen_spot_L", "pen_spot_R"]
img = {n: tuple(project([lm[n]])[0]) for n in use}
cal = Calibration.from_correspondences(img, lm, (W, H_PX))
# recovery is judged on ALL 35 landmarks, not just the 8 we fitted
truth_px = project(pitch_pts)
rec = cal.image_to_pitch(truth_px)
err = np.linalg.norm(rec - pitch_pts, axis=1)
print(f"   held-out landmarks: mean {err.mean():.2e} m,  max {err.max():.2e} m")
assert err.max() < 1e-6, "exact homography should recover the plane to machine precision"

print("\n2. same, but with 3 px of click noise on the operator's landmarks")
img_noisy = {n: tuple(project([lm[n]])[0] + rng.normal(0, 3.0, 2)) for n in use}
cal_n = Calibration.from_correspondences(img_noisy, lm, (W, H_PX))
err_n = np.linalg.norm(cal_n.image_to_pitch(truth_px) - pitch_pts, axis=1)
print(f"   held-out landmarks: mean {err_n.mean():.3f} m,  max {err_n.max():.3f} m")
print(f"   verdict: {cal_n.verdict(lm)[1]}")

print("\n3. more landmarks should beat fewer, under the same noise")
for k in (4, 6, 10, 20, 35):
    sub = names[:: max(len(names) // k, 1)][:k]
    if len(sub) < 4:
        continue
    noisy = {n: tuple(project([lm[n]])[0] + rng.normal(0, 3.0, 2)) for n in sub}
    c = Calibration.from_correspondences(noisy, lm, (W, H_PX))
    e = np.linalg.norm(c.image_to_pitch(truth_px) - pitch_pts, axis=1)
    print(f"   {len(sub):2d} landmarks -> mean {e.mean():.3f} m, max {e.max():.3f} m")

print("\n4. stitched panorama (barrel-ish seam warp) -- does TPS earn its keep?")
def stitch_warp(px):
    """Fake a stitch: squeeze x towards the seams, bow y near frame edges."""
    px = np.asarray(px, float).reshape(-1, 2).copy()
    u = (px[:, 0] - W / 2) / (W / 2)
    px[:, 0] += 55.0 * np.sin(np.pi * u)
    px[:, 1] += 30.0 * u ** 2
    return px

wide = ["corner_LT", "corner_RT", "corner_LB", "corner_RB", "halfway_T", "halfway_B",
        "pen_area_LT_front", "pen_area_LB_front", "pen_area_RT_front", "pen_area_RB_front",
        "pen_spot_L", "pen_spot_R", "centre_spot", "goal_area_LT_front", "goal_area_RB_front"]
img_w = {n: tuple(stitch_warp(project([lm[n]]))[0]) for n in wide}
warped_truth_px = stitch_warp(project(pitch_pts))

plain = Calibration.from_correspondences(img_w, lm, (W, H_PX), use_tps=False)
tps = Calibration.from_correspondences(img_w, lm, (W, H_PX), use_tps=True)
e_plain = np.linalg.norm(plain.image_to_pitch(warped_truth_px) - pitch_pts, axis=1)
e_tps = np.linalg.norm(tps.image_to_pitch(warped_truth_px) - pitch_pts, axis=1)
print(f"   homography only : mean {e_plain.mean():.3f} m, max {e_plain.max():.3f} m")
print(f"   homography + TPS: mean {e_tps.mean():.3f} m, max {e_tps.max():.3f} m")
assert e_tps.mean() < e_plain.mean(), "TPS should reduce residual on a non-pinhole view"

print("\n5. round-trip pitch -> image -> pitch")
rt = cal.image_to_pitch(cal.pitch_to_image(pitch_pts))
print(f"   max round-trip error: {np.abs(rt - pitch_pts).max():.2e} m")
assert np.abs(rt - pitch_pts).max() < 1e-6

print("\n6. feet-vs-centre of bounding box")
player_xy = np.array([[10.0, 5.0]])
feet_px = project(player_xy)[0]
head_px = project(player_xy)[0].copy(); head_px[1] -= 95        # ~1.8 m tall here
bbox = [feet_px[0] - 25, head_px[1], 50, feet_px[1] - head_px[1]]
via_feet = cal.feet_to_pitch([bbox])[0]
via_centre = cal.image_to_pitch([[bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]])[0]
print(f"   truth        {player_xy[0]}")
print(f"   via feet     {via_feet.round(3)}  err {np.linalg.norm(via_feet - player_xy[0]):.3f} m")
print(f"   via centre   {via_centre.round(3)}  err {np.linalg.norm(via_centre - player_xy[0]):.3f} m")

print("\n7. save / load round-trip")
p = cal.save("calib/_synthetic_test.json")
back = Calibration.load(p)
assert np.allclose(back.H, cal.H)
tps.save("calib/_synthetic_tps.json")
back_tps = Calibration.load("calib/_synthetic_tps.json", pitch_landmarks=lm)
assert np.allclose(back_tps.image_to_pitch(warped_truth_px), tps.image_to_pitch(warped_truth_px))
print("   both plain and TPS calibrations survive a save/load cycle")

print("\n" + "=" * 62)
print("geometry OK")

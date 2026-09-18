"""On a genuinely fixed camera the bridge must be a no-op: KLT step ~ identity, no invalid frames.

Uses the Alfheim panorama (three stationary cameras stitched). Skipped until fetched with
`python scripts/fetch_alfheim.py --minutes 2 --view panorama`.
"""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = sorted((ROOT / "data/raw/alfheim").glob("alfheim_*_panorama.mp4")) if (ROOT / "data/raw/alfheim").exists() else []


@pytest.mark.skipif(not CANDIDATES, reason="Alfheim panorama not fetched")
def test_anchored_camera_is_identity_on_a_fixed_rig():
    import cv2

    from footlytics.geometry.camera_motion import AnchoredCamera

    cap = cv2.VideoCapture(str(CANDIDATES[0]))
    ok, anchor = cap.read()
    assert ok
    cam = AnchoredCamera(anchor)
    h, w = anchor.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2)
    centre = np.array([[w / 2, h / 2]], np.float64).reshape(-1, 1, 2)
    invalid, corner_shift, centre_shift = 0, [], []
    for _ in range(300):
        ok, fr = cap.read()
        if not ok:
            break
        H = cam.update(fr)                       # no player mask: the harder case
        if H is None:
            invalid += 1
            continue
        corner_shift.append(float(np.abs(cv2.perspectiveTransform(corners, H) - corners).max()))
        centre_shift.append(float(np.abs(cv2.perspectiveTransform(centre, H) - centre).max()))
    cap.release()
    assert invalid == 0
    # Corners extrapolate a 4450 px panorama; the pitch is in the middle. Before keyframe
    # tracking this drifted 7-25 px between re-anchors (median 8.4 px); now it is noise.
    assert np.median(corner_shift) < 1.5, f"median corner drift {np.median(corner_shift):.2f} px"
    assert np.percentile(corner_shift, 95) < 3.0, f"p95 corner drift {np.percentile(corner_shift, 95):.2f} px"
    assert max(centre_shift) < 2.0, f"centre drift {max(centre_shift):.2f} px"

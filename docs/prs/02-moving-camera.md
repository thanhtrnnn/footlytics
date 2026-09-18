# PR 2: moving-camera bridge (AnchoredCamera + MovingCalibration)

Branch `feat/moving-camera` (2 commits on top of `feat/packaging`). Addresses the README's "Honest limits": *a fixed homography is only valid for a fixed camera*.

> Landed in this repository with the migration to the Match State architecture.
> These four PRs were written against `scalliontor/Footlytics`; kept here as the
> record of what was bridged across from the V0 prototype and what each change measured.

## What
- `footlytics/geometry/camera_motion.py`: `AnchoredCamera` relates every frame to the anchor frame the landmarks were clicked on: a KLT step (RANSAC, 12 inliers / 0.5 ratio), a scene-change check on the frame difference, and global SIFT re-anchoring (half resolution, 30 inliers / 0.5 ratio) every 50 frames and after any cut, retried every 10 frames while invalid. Frames that cannot be related return `None`. `MovingCalibration` warps feet points back into anchor pixels before `Calibration.image_to_pitch`, so TPS still applies and `Calibration` is untouched.
- `pipeline.run(..., camera=None)`: with a camera, the player boxes are masked out of the motion estimate, invalid frames are skipped (tracks coast) and counted; the report gains `frames_calibrated`, `frames_uncalibrated`, `calib_success_rate`, `reanchors`. With `camera=None` the code path is the existing one.
- `calib/brazil_france_*.json`: a public 720p tactical-cam clip calibrated with 6 of your landmark names, verdict "good" (median 0.24 m, max 0.48 m at `corner_RT`).
- Tests: synthetic pan+zoom tracking, cut invalidation and recovery, periodic re-anchor limits drift, MovingCalibration round trip, and an end-to-end run on the clip (skipped without clip/weights).

## Measured
- 720p tactical cam, 3 minutes (earlier prototype, same algorithm): a 34 s bench close-up rejected, 81.2% of frames calibrated, 139 re-anchors; SIFT inliers 139-384 on wide views vs 2-5 on the close-up, which set the 30-inlier threshold.
- 30 s through this pipeline with the x6 weights: 750/750 frames calibrated, 28 re-anchors, median 21 players per frame, `validate()` clean, team split "sound".
- On a fixed rig (Alfheim stitched panorama, 4450x2000, 300 frames, no player mask): chaining (t-1)->t drifted 7-25 px between re-anchors (median 8.4 px at the corners). The second commit tracks from the last keyframe instead and only chains once the camera has moved beyond one LK step: median 0.75 px, p95 2.1 px at the corners, 1.35 px max at the centre, no invalid frames. `tests/test_fixed_camera_regression.py` asserts this.

## Open
Their pitch-space tracker made 71 raw tracks on the 30 s clip (ByteTrack in the prototype: 47), stitched to 55 for ~25 people; HSV appearance on ~30 px crops does not separate teammates. Detector quality on 720p remains the binding constraint, as your README says for the panorama.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

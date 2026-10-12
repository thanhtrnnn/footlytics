# Benchmark V.League mini-GT Implementation Plan (plan 4 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score our calibration on our own target footage. The spec's V.League mini-GT is 3
matches × 15 frames, half of the frames held out and frozen once labelled. The user chose one
annotation pass, so inter-annotator error is not measured. It reports:
- landmark error in metres;
- the share of frames whose error is under 1 m;
- frame-to-frame jitter in metres.

**Architecture:**
- Frames are sampled on the M2 from the three local match videos. For each sampled frame we keep
  a 25-frame window around it, resized to 1920×1080, under `data/vleague/`. The window is there to
  measure jitter.
- A person labels pitch landmarks on the centre frames in a browser page served by a small
  local server (Python standard library, localhost only, on the M2). Every change is saved
  straight into the repo's label file, so no work can be lost and there is no export step. The
  labels are committed under `bench/vleague/labels/`.
- A new task kind `vleague` lives in its own module, `footlytics/bench/vleague.py`, and registers
  itself with the runner. It runs plan 2's `ClipCalibrator` over each window and compares our
  image→pitch homography at the centre frame with the labelled landmarks. It measures jitter
  against independently estimated image motion.

**Tech Stack:** numpy, OpenCV (KLT optical flow), plan 2 calibrator, `http.server` from the standard library, plain HTML/JS/SVG (no framework, nothing leaves the machine).

**Spec:** `docs/superpowers/specs/2026-10-11-benchmark-design.md`. Relevant parts: the suites row
"V.League mini-GT", the validity rule "V.League labels", and build-order step 4.

## Global Constraints

- **Spec wording:** "V.League mini-GT: 3 matches x 15 frames, half held out | landmark error (m)
  on landmarks the solver did not use, % frames under 1 m, frame-to-frame jitter (m)". Our
  calibrator is automatic and never sees the labels, so every labelled landmark is one "the
  solver did not use".
- **Labels:** scored on held-out landmarks with one annotation pass; inter-annotator error is
  unavailable. The held-out half is frozen after label review.
- **Footage:** FPT Play broadcasts, internal use only. Videos and frames stay out of git (`/data/`
  is gitignored). Only labels and the manifest (frame indices, video sha256) are committed.
- **Frozen frames:** they are scored only by the milestone suite and never used for tuning.
- **Shared host:** files go under `~/footlytics-bench/` only, after the disk floor is checked. The
  V.League set is about 300 MB of JPEGs. The GPU queue is shared with the parallel session
  football-e0, so check before running.
- **File ownership:** football-e0 currently owns `footlytics/bench/{metrics,runner,evaluators,results}.py`.
  This plan touches them in exactly one place: a one-line `from . import vleague` at the end of
  `runner.py`, which registers the kind. That line must be agreed with football-e0 before the
  commit, or added after its branch merges.
- **Pitch coordinates:** `DEFAULT_PITCH.landmarks()` (105 × 68, centre origin, `T` = far
  touchline, y negative), the same convention as the calibrator's `H_img_to_pitch`. Both the
  labels' true coordinates and the calibrator assume 105 × 68, so errors are in **model metres**
  (agreement with the standard pitch model), not measured distance on a V.League pitch whose size
  we have not measured. Every reported number says "model metres".
- **Tests:** `uv run --extra dev --extra calib pytest`.

## Facts verified while planning (2026-10-12)

- **Videos** (repo root on the M2, gitignored):
  - `CÔNG AN TP HỒ CHÍ MINH - HÀ NỘI - LPBANK V.LEAGUE 1 - 2026-27.mp4`: 2560×1440, 60 fps,
    383,700 frames.
  - `SHB ĐÀ NẴNG - HẢI PHÒNG - LPBANK V.LEAGUE 1 - 2025-26.mp4`: 2560×1440, 50 fps, 323,870 frames.
  - `NINH BÌNH - PVF-CAND - LPBANK V.LEAGUE 1 - 2025-26.mp4`: 2560×1440, 60 fps, 371,578 frames.
- `DEFAULT_PITCH.landmarks()` has 35 named points: corners, halfway ends, centre spot and
  circle, penalty and goal-area corners, penalty spots and arcs, and goal posts.
  `geometry/annotate.py` has `SUGGESTED` (15 names in clicking order) and `HELP` (descriptions).
- The existing hand calibration `calib/vn_cahcm_hanoi_frame4500.json` has 12 named points (format
  `{"H", "image_size", "named_points", "use_tps", "notes"}`). It is a single frame, so it is not
  part of this set.
- `geometry/calib_assist._pitch_mask(hsv) -> uint8 mask` gives the green-pitch mask used to reject
  close-ups.
- `geometry/homography.Calibration.from_correspondences(points, landmarks, image_size)` fits a
  homography from named points. `apply_homography(H, pts)`.
- Plan 2 pieces are reused:
  - `bench.runner._calibrator(paths)`, `ClipCalibrator(fit_keyframes, every, gate_px,
    tracker_factory)`, `static_overlay_mask`.
  - `SoccerMasterCalibrator.fit(frames_rgb, which=..., **kw)` and its `last_errors`.
  - `runner.kind`, `TaskContext`, `require_vram`.
  - `remote.Remote.ship_dir(name, archive)` writes to `ext/<name>` only.

## Review Focus

- **A labelled frame where the calibrator produced no homography** (a close-up slipped through, or every keyframe was rejected) → the frame counts as a miss: no error number, the "frames under 1 m" denominator still includes it, and it is reported as `uncalibrated`. (Task 3 test.)
- **Labels whose frozen file changed after freezing** → the milestone task fails loudly ("frozen labels changed"), not silently scoring new labels. (Task 4 test.)
- **A label using a landmark name that is not in `DEFAULT_PITCH`** (typo, or an older vocabulary) → the label loader rejects the file with the name, before any scoring. (Task 2 test.)
- **Jitter on a window that contains a camera cut** → pairs whose image-motion estimate fails (too few KLT inliers) are skipped and counted, not scored as huge jitter. (Task 3 test.)
- **Re-running the sampler** → the same frames are chosen (deterministic), and existing window JPEGs are not rewritten, so labels never point at a different picture. (Task 1 test.)

## File structure

```
footlytics/bench/vleague.py          sampling, manifest, label I/O, metrics, the `vleague` task kind
footlytics/bench/vleague_label.html  labelling page template (filled by `bench vleague-label`)
footlytics/bench/cli.py              + vleague-sample, vleague-label, ship-data commands
footlytics/bench/remote.py           + Remote.ship_data(name, archive) -> data/<name>
footlytics/bench/runner.py           + one line at the end: `from . import vleague  # registers the kind` (agreed with football-e0)
footlytics/bench/suites/quick.yaml, milestone.yaml   + vleague tasks
bench/vleague/manifest.json          frame indices, splits, video sha256 (committed)
bench/vleague/labels/*.json          labels (committed; frozen file hash pinned in milestone.yaml)
tests/test_bench_vleague.py
```

---

### Task 1: Frame sampling and the manifest

**Files:** Create `footlytics/bench/vleague.py` (sampling part). Modify `footlytics/bench/cli.py` (the `vleague-sample` command). Test `tests/test_bench_vleague.py`.

**Interfaces:**
- Produces:
  - `WINDOW = 12` (frames each side of the labelled frame), `SIZE = (1920, 1080)`, `PER_MATCH = 15`.
  - `pitch_share(frame_bgr) -> float`.
  - `choose_frames(n_frames, fps, is_pitch: Callable[[int], bool], k=PER_MATCH, step_s=30.0) -> list[int]`:
    deterministic. Candidates sit every `step_s` seconds, starting `WINDOW` frames in. It keeps
    candidates where `is_pitch(idx)` is true, then picks `k` evenly spaced by rank.
  - `assign_splits(keys: list[str]) -> dict[str, dict]`: by rank within each match, even ranks
    are `dev` and odd ranks `frozen`. `relabel` is true for rank % 4 == 1, about 3-4 per match,
    about 11 in total.
  - `extract_windows(video, frames, out_dir, size=SIZE)` writes
    `out_dir/<idx:07d>/<offset:+03d>.jpg` for offsets -12..+12. It skips files that already exist.
  - `sample_match(video, match_id, data_dir) -> dict` (the manifest entry).
  - `write_manifest(entries, path)`.
  - Manifest schema:
    `{"version": 1, "size": [1920, 1080], "window": 12, "matches": {match_id: {"video": name, "sha256": ..., "fps": f, "frames": {"<idx:07d>": {"split": "dev"|"frozen", "relabel": bool}}}}}`.
  - Frame key: `f"{match_id}/{idx:07d}"`.

- [ ] **Step 1: Failing tests:**
  - `choose_frames` is deterministic and evenly spread. With `n_frames=60_000`, `fps=50` and
    `is_pitch` rejecting every third candidate, it returns 15 sorted, distinct indices inside
    `[WINDOW, n-WINDOW)`, and calling it again gives the same list.
  - `assign_splits` gives exactly 8 dev and 7 frozen for 15 keys of one match, and between 3 and
    4 relabels.
  - `pitch_share` is about 1 on a green synthetic frame and about 0 on a grey one.
  - `extract_windows` on a tiny synthetic video, written with `cv2.VideoWriter` (40 frames, MJPG
    `.avi`), writes 25 files per chosen frame at `SIZE`. Run it a second time: no file is rewritten (compare
    `st_mtime_ns` before and after).
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.**
  - `choose_frames` steps from `WINDOW` to `n - WINDOW` every `round(step_s * fps)` frames,
    filters with `is_pitch`, and then picks `k` with
    `np.linspace(0, len(c) - 1, k).round().astype(int)` without duplicates.
  - `pitch_share` is the mean of `_pitch_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)) > 0`.
  - `is_pitch` for real videos seeks with `cv2.CAP_PROP_POS_FRAMES`, reads a frame, and returns
    `pitch_share >= 0.35`.
  - `extract_windows` seeks once per window to `idx - WINDOW`, reads 25 frames sequentially, and
    writes them with `cv2.resize(..., SIZE, INTER_AREA)` and `cv2.imwrite(..., [cv2.IMWRITE_JPEG_QUALITY, 92])`.
  - `sample_match` also computes the video's sha256 (streamed) so the manifest pins the footage.
  - CLI: `footlytics bench vleague-sample VIDEO... --data data/vleague --manifest bench/vleague/manifest.json`.
    Match ids come from the file name: the two team names, slugified to ASCII with accents
    removed, joined by `-`, e.g. `cong-an-tp-ho-chi-minh-ha-noi`.
- [ ] **Step 4: Run → pass.** Commit `feat(bench): V.League frame sampling with jitter windows`.

### Task 2: Labelling server, page and label I/O

**Files:** Create `footlytics/bench/vleague_label.html` (template) and `footlytics/bench/vleague_label.py` (server). Modify `vleague.py` (label I/O) and `cli.py` (the `vleague-label` command). Test in `tests/test_bench_vleague.py`.

**Interfaces:**
- Produces:
  - `load_labels(path, manifest) -> dict`. It validates the version, that every landmark name is
    in `DEFAULT_PITCH.landmarks()` (otherwise `ValueError` naming the bad name), that coordinates
    fall inside `SIZE`, and that frame keys exist in the manifest.
  - `save_label(path, frame_key, landmark, uv | None) -> dict`. It sets (or, for None, removes)
    one point and writes the whole file atomically (write a temp file in the same directory, then
    `os.replace`). It validates the name and the coordinates the same way as `load_labels`.
  - `make_server(manifest_path, data_dir, labels_path, annotator, session, relabel_only, port=0)
    -> http.server.ThreadingHTTPServer`, bound to `127.0.0.1` only. Routes:
    - `GET /` serves the page;
    - `GET /frames/<match>/<idx>.jpg` serves the centre frame from `data_dir` (only keys in the
      manifest; anything else is 404, so no path traversal);
    - `GET /labels` returns the current label file;
    - `POST /label` with `{"frame", "landmark", "uv"}` (uv null deletes) calls `save_label` and
      returns the updated frame's points.
  - Label schema:
    `{"version": 1, "annotator": str, "session": int, "image_size": [1920, 1080], "frames": {frame_key: {landmark: [u, v]}}}`.
    Coordinates are in 1920×1080 pixels.

- [ ] **Step 1: Failing tests** (behaviour, through the server the way the page uses it):
  - Start `make_server(...)` on port 0 in a thread over a 2-frame fake manifest. POST a point for
    `pen_area_LT_front` at `[812.5, 403.0]`, then read the label file from disk and assert the
    exact entry `{"pen_area_LT_front": [812.5, 403.0]}` under that frame. POST `uv: null` and
    assert the landmark is gone from the file.
  - POST an unknown landmark name → HTTP 400 with the name in the body, and the file unchanged.
  - POST a point outside 1920×1080 → 400.
  - `GET /frames/<a key not in the manifest>.jpg` and `GET /frames/../../etc/passwd` → 404.
  - With `relabel_only=True`, `GET /labels`' companion manifest view (`GET /frames-list`) lists
    only the flagged frames.
  - `load_labels` accepts the file the server wrote, rejects an unknown landmark name (named in
    the error), a point outside the image, and a frame key not in the manifest.
  - Restarting the server on the same label file keeps every point (the page reloads from it).
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.**
  - Server: `ThreadingHTTPServer` on `127.0.0.1`, a `BaseHTTPRequestHandler` with the four routes
    above plus `GET /frames-list` (frame keys, split, relabel flag, and per-frame point counts).
    The page template is read from `vleague_label.html` and served with the landmark names
    (`SUGGESTED` first, then the rest of `DEFAULT_PITCH`), the `HELP` text, and each landmark's
    pitch coordinates injected as JSON in place of the placeholder `/*DATA*/null`.
  - Page (inline CSS and JS, no external requests):
    - Left column: frames with key, split, and the number of points placed.
    - Centre: the frame on a canvas at fit-to-width. **Mouse wheel zooms around the cursor**
      (up to 8×), dragging the empty canvas pans, and a **4× loupe** follows the cursor near the
      selected point. Click places the selected landmark; dragging a point moves it; right-click
      deletes it. Each change POSTs to `/label` immediately, and a small "saved" / "save failed"
      indicator shows the result.
    - Right column: landmark names with `HELP` tooltips, and an **inline SVG pitch diagram**
      (105 × 68) that marks every landmark and highlights the selected one, so left/right and
      top/bottom are never guessed from a name.
    - Keyboard: `n`/`p` for next/previous frame, `1`-`9` for the suggested names, `Del` to remove
      the selected point, `0` to reset zoom.
    - A fixed guide line at the top: "Click the centre of the line intersection. Goal posts: the
      base of the post on the goal line. Label every clearly visible landmark, at least 3."
  - CLI: `footlytics bench vleague-label --manifest bench/vleague/manifest.json --data data/vleague --labels bench/vleague/labels/labels-A-s1.json --annotator A --session 1 [--relabel-only] [--port 8765]`.
    It creates the label file if missing, starts the server, prints `http://127.0.0.1:<port>/`,
    and runs until Ctrl-C.
- [ ] **Step 4: Run → pass.** Manually open the page once with the 2-frame fake set and check that
  zoom, loupe, diagram highlight and autosave work (screenshot in the report). Commit
  `feat(bench): V.League labelling server with autosave, zoom and pitch diagram`.

### Task 3: Metrics: landmark error, frames under 1 m, jitter, labelling error

**Files:** `vleague.py` (metrics part); tests.

**Interfaces:**
- Produces:
  - `landmark_errors(H_img_to_pitch, points: dict[str, (u, v)], landmarks) -> dict[str, float]`
    (metres).
  - `frame_error(H, points, landmarks) -> float | None`, the median over landmarks; None when H is
    None.
  - `image_motion(prev_gray, gray) -> np.ndarray | None`: KLT plus RANSAC homography from prev to
    current. None when fewer than 30 inliers or an inlier ratio below 0.5.
  - `jitter_m(Hs: list, grays: list, pitch_pts: np.ndarray, sources: list[str]) -> dict`.
    For each consecutive pair with both H and a motion estimate M: take x_t = inv(H_t) applied to
    P, predict x_{t+1} = M x_t, and take P' = H_{t+1} x_{t+1}; the pair's jitter is the mean of
    |P' − P| over the points. Returns `{"switch_m", "between_m", "pairs_switch", "pairs_between",
    "pairs_skipped"}`. A pair is a *switch* when either frame's source is `keyframe` (the
    calibration jumps to a fresh solve there); otherwise it is *between* keyframes, where both
    calibrations come from the camera tracker and jitter is expected near zero by construction.
    Reporting the two separately keeps the number honest about what it measures.
  - `labelling_error_m(a: dict, b: dict, landmarks, image_size) -> float | None`. On the
    landmarks both sets labelled (at least 4), fit `Calibration.from_correspondences` on A and
    map both A's and B's pixels to the pitch. The result is the median distance in metres.

- [ ] **Step 1: Failing tests** (synthetic, exact):
  - **Landmark error:** points produced by projecting landmarks through a known camera
    (`sn_camera.make_camera`, at 1920×1080) give an error of about 0 with the true H, and about
    the shift with H offset by 1 m in x.
  - **Uncalibrated frame:** `frame_error(None, ...)` returns None.
  - **Jitter, pure pan:** a textured synthetic image panned by 3 px per frame across 6 frames
    (`np.roll` on a noise texture), with Hs that are consistent with that pan (H_t = H_0 ∘
    shift(−3t)), gives jitter below 0.05 m.
  - **Jitter, perturbed:** the same set with one H moved 0.5 m and its source set to `keyframe`
    gives `switch_m` above 0.1 m and `between_m` below 0.05 m (the perturbation lands only in the
    switch pairs).
  - **Jitter, cut:** a window with a cut (the frame replaced by a different texture) counts that
    pair in `pairs_skipped`, not in the mean.
  - **Labelling error:** identical label sets give 0. B shifted 2 px on every point gives a
    small positive value below 0.5 m at broadcast scale.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** with `cv2.goodFeaturesToTrack` (`maxCorners=600`, `qualityLevel=0.01`,
  `minDistance=8`), `cv2.calcOpticalFlowPyrLK`, and `cv2.findHomography(..., RANSAC, 2.0)`, at
  half resolution for speed, rescaled. Points use `apply_homography`.
- [ ] **Step 4: Run → pass.** Commit `feat(bench): V.League metrics (landmark error, jitter, labelling error)`.

### Task 4: The `vleague` task kind, data shipping, suites

**Files:** `vleague.py` (the task part); `remote.py` (`ship_data`); `cli.py` (`ship-data`);
`runner.py` (the one-line import, agreed with football-e0); `suites/quick.yaml` and
`suites/milestone.yaml`. Tests.

**Interfaces:**
- Consumes: `runner.kind`, `TaskContext`, `_calibrator`, `require_vram`, `VRAM_MIB`, `TRACKER`;
  plan 2 `ClipCalibrator`, `static_overlay_mask`; Tasks 1-3.
- Produces:
  - Kind `vleague`. Task keys: `split: dev|frozen`, `labels: [paths relative to the repo]`,
    `relabels: [paths]` (optional), `frozen_sha256: {path: sha}` (milestone only), `every: 10`,
    `gate_px: 25`.
  - `frozen_digest(labels, manifest) -> str`: sha256 of a canonical JSON (sorted keys, no
    whitespace) of **only the frozen frames' entries**. Development labels can still be corrected
    after freezing without breaking the milestone; a change to any frozen frame fails it.
  - Per frame: `error_m`, `n_landmarks`, `source` (keyframe/bridged/none) and the jitter dict.
  - Aggregate:
    - `median_error_m` (median of frame medians, calibrated frames only);
    - `share_under_1m` (calibrated frames under 1 m divided by **all** labelled frames);
    - `uncalibrated` (count);
    - `jitter_switch_m` and `jitter_between_m` (pair-weighted means over all windows);
    - `jitter_pairs_skipped`;
    - `labelling_error_m` (median over relabelled frames, only when relabels are given; omitted
      for this single-annotation run).
    - `frames`.
  - `Remote.ship_data(name, archive)`: the same atomic tar-replace as `ship_dir`, but into
    `data/<name>`. The name goes through `_safe`.

- [ ] **Step 1: Failing tests** (fakes as in plan 2's runner tests: a `FakeCal` whose `fit`
  returns a CameraFit with a known camera, and `TRACKER` set to a fake):
  - The `vleague` task on a 2-frame fake set (windows of tiny JPEGs, labels built from the known
    camera) gives `median_error_m` of about 0, `share_under_1m == 1.0`, and frames == 2.
  - A labelled frame with no camera (the fake fit returns None for every keyframe) is counted in
    `uncalibrated`, and `share_under_1m == 0.5`.
  - For the milestone, a changed **frozen** frame's point makes `frozen_digest` mismatch and the
    task fail with "frozen labels changed"; changing a **dev** frame's point does not.
  - `split: dev` scores only dev frames.
  - `Remote.ship_data` builds an atomic replace under `data/<name>` and rejects unsafe names.
  - `bench ship-data vleague --remote h` tars `data/vleague` (the windows only, not HTML pages)
    and calls `ship_data`.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.**
  - For each labelled frame in the split, read its 25 window JPEGs from
    `paths.data / "vleague" / match / idx`.
  - Build the overlay mask from the window.
  - Run `ClipCalibrator(fit, every, gate_px, tracker_factory=runner.TRACKER).run(frames, mask)`,
    with the fit counting `last_errors` as in `gsr_calibration`.
  - Take H at offset 0 (list index `WINDOW`) and compute `frame_error`.
  - Compute `jitter_m` over the window. Its pitch points are the labelled landmarks' true
    coordinates on that frame.
  - Labels come from `load_labels`. Frozen digests (`frozen_digest`) are checked first.
  - Add `from . import vleague  # noqa: E402,F401 - registers the "vleague" task kind` as the last
    line of `runner.py`. `vleague.py` imports from `runner` and the import comes after every
    runner definition, so it does not cycle at import time.
  - Suites:
    - `quick.yaml` gets `vleague-dev` (`kind: vleague`, `split: dev`, `labels: [bench/vleague/labels/labels-A-s1.json]`,
      `relabels: [bench/vleague/labels/labels-A-s2.json]`).
    - `milestone.yaml` gets `vleague-frozen` with `split: frozen` and a `frozen_sha256` entry.
      The real sha is filled in Task 5 step 4, when the labels freeze; until then the entry is
      the literal `PENDING`, which the task treats as "not frozen yet" and fails with a clear
      message.
- [ ] **Step 4: Run → pass**, then the full suite. Commit `feat(bench): vleague task kind, data shipping, suites`.

### Task 5: Sample, label, freeze, run (operational, needs the user)

- [ ] **Step 1 (agent, M2):** `uv run footlytics bench vleague-sample "CÔNG AN TP HỒ CHÍ MINH - HÀ NỘI - LPBANK V.LEAGUE 1 - 2026-27.mp4" "SHB ĐÀ NẴNG - HẢI PHÒNG - LPBANK V.LEAGUE 1 - 2025-26.mp4" "NINH BÌNH - PVF-CAND - LPBANK V.LEAGUE 1 - 2025-26.mp4"`.
  Check: 45 frames, 3 × 25 window JPEGs each, about 300 MB, and every centre frame shows the
  pitch. A contact sheet of the 45 centre frames goes to the scratchpad for the user to glance
  at. Commit `bench/vleague/manifest.json`.
- [ ] **Step 2 (user):** label session 1 on all 45 frames:
  `uv run footlytics bench vleague-label --labels bench/vleague/labels/labels-A-s1.json --annotator A --session 1`,
  then open the printed localhost address. Label every clearly visible landmark (at least 3 per
  frame), clicking the centre of each line intersection; zoom in for far-side points. Everything
  saves as you go; stop with Ctrl-C and resume any time.
- [x] **Step 3 (user):** second annotation pass explicitly skipped. No `labels-A-s2.json` is used;
  inter-annotator error is omitted from the benchmark results.
- [x] **Step 4 (agent):** resolve every live label warning, run `load_labels` on session 1, commit
  `labels-A-s1.json`, and pin `frozen_digest(session-1 labels)` in `milestone.yaml`'s
  `frozen_sha256`. From then on the frozen frames' labels never change; dev labels may still be
  corrected.
- [ ] **Step 5 (agent):** check with football-e0 that the GPU is free. Then run
  `uv run footlytics bench ship-data vleague --remote iec4090_2`, the `quick` suite (vleague-dev),
  then the `milestone` suite (vleague-frozen). Record the numbers under the spec's reference
  points, labelled "model metres": median landmark error, share under 1 m, and jitter at keyframe
  switches and between keyframes. State that inter-annotator error was not measured. Commit.

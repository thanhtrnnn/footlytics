# Footlytics

Football match analytics for Vietnamese clubs. Video of a match goes in; tracked
player positions on a scale pitch model come out, and everything else is built
on top of that.

Target user: V.League 1 and 2 clubs, who currently have no access to the
tracking-data analysis that is routine in European leagues.

---

## Getting started

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
uv run pytest                     # synthetic tests run offline; the rest skip until you fetch data
```

Weights go in `weights/`: `yolo_v8x6_finetuned.pt` from
[`xleprime/SoccerMaster`](https://huggingface.co/xleprime/SoccerMaster) (195 MB, ungated) and,
for the ball, `football-ball-detection.pt` from the roboflow/sports release (mirrored on
Hugging Face as `martinjolif/yolo-football-ball-detection`). Both are gitignored.

```bash
bash scripts/fetch_sample.sh      # one public 720p tactical-cam clip -> data/clips/
```

Calibrating a new camera is three commands. Every venue needs this once; a moving camera
needs it once per clip, on whichever frame shows the most pitch:

```bash
# 1. propose numbered line intersections on the anchor frame
uv run footlytics calib-assist data/clips/sample_3min.mp4 --frame 752 --out data/out/assist

# 2. open data/out/assist.jpg, and for each candidate you recognise write
#    "landmark_name": [px, py] into assist.json -> template.image_points
#    (names come from geometry/pitch.py: corner_LT, halfway_T, pen_spot_L, ...)
uv run footlytics calibrate data/out/assist.json --out calib/myvenue.json

# 3. check it before trusting it -- the overlay should sit on the painted lines
uv run footlytics drift-check data/clips/sample_3min.mp4 --calib calib/myvenue.json \
    --frames 0 --frames 752 --frames 1500
```

Then run the pipeline:

```bash
uv run footlytics run data/clips/sample_3min.mp4 \
    --calib calib/brazil_france_3min_frame752.json --anchor-frame 752 \
    --ball-model weights/football-ball-detection.pt --out data/out/mc_3min
```

Outputs, next to each other in `--out`: `tracks.parquet` + `meta.json` (the Match State),
`report.json`, `quality.json` + `quality.md`, and `radar.mp4`. Add `--no-camera` for a fixed
rig, `--stride N` to process every Nth frame, `--max-frames N` for a quick look.

---

## The one architectural idea

Everything talks through **Match State**: every player and the ball, at 10–25 Hz,
in pitch metres, with team/role/jersey, plus a derived event stream.

```
      video ──► perception ──►  MATCH STATE  ──► analytics ──► reports / app
                                (metres, not pixels)
```

Perception is a swappable box. Analytics never sees a pixel. This means:

* the tracking model can be replaced — by a better one, by a commercial feed, or
  by Bepro's own FIFA-certified tracking if you end up reselling — without a
  single analytics function changing;
* every tactical metric is a pure function of a DataFrame, so it is testable
  without a GPU;
* the pretty 3D reconstruction is just another renderer over the same state.

The schema is deliberately kloppy-compatible (metric pitch, origin at the centre
spot), so the PySport ecosystem — mplsoccer, existing xT and pitch-control
implementations — works on our data, and so does interop with TRACAB /
SkillCorner / StatsBomb feeds later.

---

## Why a fixed camera changes the engineering

Broadcast pipelines are complicated because the camera pans, tilts, zooms and
shows about a third of the pitch. Much of SoccerNet's machinery exists to fight
that. A fixed full-pitch rig removes those problems instead of solving them:

| | broadcast | fixed full-pitch |
|---|---|---|
| calibration | per-frame neural net, drifts, fails in scrambles | **once per venue**, exact |
| players visible | ~1/3 of the pitch | all 22, always |
| main cause of ID switches | players leaving frame | mostly gone |
| ball height | unrecoverable | recoverable by triangulation (3-cam) |
| cost | players are large | players are ~40 px → must tile the detector |

So this repo does not port SoccerMaster's pipeline. It takes SoccerMaster's
*models* and builds the fixed-camera pipeline properly.

---

## What is built

```
footlytics/
  state/schema.py        Match State: the contract. Parquet + JSON metadata, self-validating.
  geometry/pitch.py      Pitch model, 43 named landmarks derived from the Laws of the Game.
  geometry/homography.py DLT homography + optional thin-plate spline. Calibration grading.
  geometry/annotate.py   In-notebook landmark clicker, and the projected-pitch visual check.
  geometry/calib_assist.py  Finds pitch line intersections so the clicker has candidates.
  geometry/camera_motion.py AnchoredCamera + MovingCalibration: the moving-camera bridge.
  geometry/keypoints.py  Automatic calibration from a 32-vertex YOLO-pose model. Not wired in.
  geometry/lines.py      Classical painted-line detection. EXPERIMENTAL — fails on hard footage.
  geometry/lines_nn.py   Runs SoccerMaster's learned heads locally on Apple Silicon (MPS).
  perception/detect.py   Tiled YOLO detection (SoccerMaster's finetuned YOLOv8x6) + ball pass.
  perception/track.py    Kalman + Hungarian tracking IN PITCH METRES, with appearance gating.
  perception/teams.py    Kit-colour descriptors, 2-means, per-track voting, clash diagnostics.
  perception/identity.py Tracklet stitching and roster-constrained team assignment.
  analytics/kinematics.py Smoothed speed/acceleration, distance/HSR/sprint summary.
  analytics/ball.py      Fills short ball gaps between two sightings. Never extrapolates.
  analytics/tactics.py   Attacking direction, team block, formation, PPDA, transitions.
  viz/radar.py           The 2D tactical radar: frames and mp4.
  viz/overlay.py         Video and radar side by side.
  pipeline/radar.py      video → MatchState, one pass.
  pipeline/quality.py    quality.json + quality.md: the numbers the blockers are scored on.
  cli.py                 run / calib-assist / calibrate / drift-check.
scripts/test_*.py        Synthetic-truth tests for geometry, tracking, teams, rendering.
scoccer.ipynb            The Colab (A100) driver, end to end.
```

Bootstrap uses `yolo_v8x6_finetuned.pt` from
[`xleprime/SoccerMaster`](https://huggingface.co/xleprime/SoccerMaster) — a single
ungated 195 MB file, so no conda/tracklab install is needed to get a radar.

---

## Measured behaviour

Everything below is from `scripts/test_*.py` against synthetic ground truth, so
the numbers are honest about the code and optimistic about reality.

**Calibration** (`test_geometry.py`)
* exact pinhole recovery: 1e-13 m
* 3 px of click noise, 8 landmarks: 0.09 m mean
* 4 clustered landmarks: 1.04 m — 35 spread landmarks: 0.06 m. **Click more points.**
* stitched panorama, one homography 1.25 m → with TPS 0.22 m
* **mapping the box centre instead of the feet: 6.7 m of error.** The single
  easiest way to corrupt every number downstream.

**Tracking** (`test_tracker.py`, 22 players + ball, 1 minute)

| detections | ID switches, motion only | with appearance |
|---|---|---|
| near-perfect | 103 | **10** |
| 0.2 m noise | 164 | **10** |
| 0.35 m noise, 5% dropout | 243 | **77** |
| 0.6 m noise, 25% dropout | 3561 | 3095 |

Two conclusions. Appearance features are not optional — 100% of motion-only
switches happen with another player within 2 m, which geometry cannot resolve.
And past ~0.6 m of detection error, no tracker change helps: **detector quality
is the binding constraint.**

**SoccerMaster's learned line detector runs on an M2 Pro — and also fails here.**
Both checkpoints load with zero missing keys and inference costs 272 ms/frame on
MPS with 32 GB unified memory; an earlier claim that this "needs a GPU" was
wrong. The head predicts two Gaussian blobs at each line's ENDPOINTS, not a drawn
line — reading those blobs as failure was an error on my part. On the correct
reading it still fails here, verified against a real broadcast frame: 9 lines at
0.70–0.86 with coherent geometry there (goalposts as vertical segments, crossbar
joining their tops), versus 4 lines at 0.21–0.47 on our night panorama, mostly a
single endpoint stuck on the image border. That test also proves the integration
is correct — the weakness is the footage. Squashing the 2:1 panorama into the
model's square input is worse — every class drops under 0.01, because a
full-pitch panorama looks nothing like the broadcast frames it was trained on.
The learned and classical methods fail for the same reason: fisheye curvature,
floodlit contrast and full-pitch framing are all far outside the broadcast
distribution. `backbone.pt` also turned out to contain the complete fine-tuned
tower, so the 1.6 GB base SigLIP2 download is unnecessary.

**Automatic pitch-line detection does not work yet, and that is a finding.**
The premise was that a fixed camera makes calibration easy enough for classical
vision. On the SoccerTrack v2 night match it failed, and failed misleadingly:
developed on one frame it recovered the width exactly, then returned 87.5, 63.0,
46.0 and 47.5 m on four other frames of the same clip — a 41 m spread — reporting
high confidence throughout. Four approaches all failed: raw peak height, robust
quantiles, peak prominence with a width constraint, and multi-frame consensus.
Consensus fails because the false detections are static structures — a fence,
floodlight masts, a neighbouring pitch's markings — so they reproduce perfectly
frame to frame. **Stable false positives are indistinguishable from true ones by
agreement alone.**

What works is *independent* evidence: players cannot stand outside the pitch, so
their positional extent is a hard lower bound derived from entirely different
data. `cross_check` uses it, and it correctly rejects the 47 m answer against a
72.4 m observed player span. Measuring the pitch by hand once per venue remains
the reliable route.

**Pitch dimensions are a measurement, not a default.** On SoccerTrack v2 the
loader assumed the IFAB 68 m width. It was wrong: the ground is ~76 m wide, and
the error was invisible in aggregate but put 2.14% of positions outside the
touchline — systematically, by up to 11 m, for whole passages of play. Projecting
the model pitch back onto the video showed the modelled touchline sitting well
inside the painted one.

`validate()` now catches this with no ground truth needed. Players fill a pitch
similarly along both axes, so the fraction of the configured length they span
calibrates what to expect across the width. Here they spanned 86% of the length
but **106% of the width** — impossible, and proof on its own. Fixing the width
took off-pitch positions from 2.14% to 0.51%.

Two lessons worth keeping: an inferred number (occupancy scaling suggested 84 m)
should be checked against the pixels when the pixels are available — reading the
touchline off the frame gave 76 m. And accuracy is **not uniform across the
pitch**: at the far touchline the projection compresses several metres into a few
pixels, so y = −3, 0, +3 and +6 are visually indistinguishable there. Filming
from one touchline means the far side is always the weak side.

**Against real football** (SoccerTrack v2 match 117092, 6,000 frames of hand-annotated
ground truth)
* occupancy across pitch thirds 27 / 41 / 33% and channels 23 / 47 / 28% — the
  midfield-heavy, box-to-box distribution football actually has
* top speed median 7.13 m/s (range 3.96-8.55); the 3.96 outlier is the goalkeeper
* this is what caught the distance bug below

**Identity resolution** (`test_identity.py`, real ground truth deliberately fragmented)

| fragmentation | fragments | tracklets out | welded | players recovered |
|---|---|---|---|---|
| 5 cuts/player | 129 | 22 | 0 | 22/22 |
| 10 cuts/player | 233 | 22 | 0 (2 without appearance) | 22/22 |
| 20 cuts/player | 418 | 32 | 0 (6 without appearance) | 12/22 |

"Welded" means one tracklet containing two real players — the failure that
matters, because it silently averages two people into one set of statistics.
Appearance eliminates it up to moderate fragmentation.

**Jersey numbers, read on only a handful of frames** (`test_identity.py`). Shirt
numbers are sparse by nature — legible only when a player faces the camera —
but a number does not decay with time the way a motion gate does, so it joins
fragments that trajectory reasoning never could.

| fragmentation | motion only | + jersey numbers |
|---|---|---|
| 20 cuts, 5% legible | 29 tracklets, 2 welded, 13/22 | 23, **0 welded**, **21/22** |
| 40 cuts, 5% legible | 81, 19 welded, 0/22 | 37, **3 welded**, 9/22 |
| 40 cuts, 15% legible | 81, 19 welded, 0/22 | 35, **1 welded**, 10/22 |

The ordering is the whole trick. Using numbers *afterwards*, to join leftovers,
left welding untouched at 19 — the damage was already done during motion
stitching, and a welded tracklet's number evidence is mixed beyond repair. Used
*before*, as a veto that refuses to merge fragments carrying different confident
numbers, welding drops from 19 to 1. Same evidence, opposite outcome.

One trap worth stating: a shirt number is **not** a unique identity — both teams
field a number 10. Identity is the pair (team, number), and appearance is
checked too, so a wrong team label cannot on its own cause a bad merge.

Team assignment respecting the eleven-a-side quota *over time* held **100% of
frames at exactly 11v11**, where a single global split managed **0%** — the
global rule balances total playing time, not headcount, and returned something
other than 11/11 in 57.7% of realistic cases.

**Team assignment** (`test_teams.py`, per-track voting)
* red vs blue: 100% · red vs red: 68% · white vs white: 64%
* `diagnose()` catches gross failure (clash, degenerate split) but cannot
  certify the rest — confirm with `cluster_montage()` before anything ships.

---

## Moving-camera bridge

The fixed-camera path above is the product. Clubs today mostly have a tactical cam or a
broadcast feed that pans, zooms and cuts to the bench, so there is a bridge for that footage:

* `geometry/camera_motion.py` — `AnchoredCamera` relates every frame to the frame the
  landmarks were clicked on (KLT step, scene-change check, periodic global SIFT re-anchor).
  Frames it cannot relate (a bench close-up) are skipped and counted, never mapped.
  `MovingCalibration` warps detections back into anchor pixels before the usual
  `Calibration` mapping, so TPS still applies. On a fixed rig the step is the identity.
* `footlytics run CLIP --calib venue.json [--anchor-frame N] [--ball-model W]` — the pipeline
  with the bridge, a dedicated 1920 px ball model (the x6 weights expose only `person`),
  `quality.json` next to `tracks.parquet`, and a radar mp4. `--no-camera` for a fixed rig.
* `footlytics calib-assist CLIP --frame N` proposes numbered line intersections to label;
  `footlytics calibrate` builds and grades the calibration; `footlytics drift-check` overlays
  the pitch model on chosen frames and lists the frames the camera could not be related.

Measured on a public 720p tactical-cam clip (Brazil vs France): a 34 s bench cut rejected,
81% of frames calibrated over 3 minutes; 30 s through this pipeline: 750/750 frames
calibrated, median 21 players plus 2 officials per frame with the x6 weights, clean
`validate()`, the match ball in 77% of frames with the dedicated model (85% of the calibrated
frames over 3 minutes) and never more than 5 m from where it was a frame earlier, 506 s of
compute per match minute on an M2 without the ball model, ~550 s with it (x6 on MPS is the
cost; yolo11n was 2.5x faster at the same players-per-frame). The ball model also finds the
spare balls lying beyond the touchline, and a still spare ball usually out-scores the match
ball; the detector therefore keeps several candidates and the pipeline keeps the one the ball
could have reached from its last sighting. Details and the blocker table in
`docs/feasibility.md` (Vietnamese).

## Roadmap

**Phase 0 — radar on real footage** *(you are here)*
Bepro demo / public full-pitch video → tracked radar + physical numbers. This is
the demo that gets a V.League club to let you install a camera.

**Phase 1 — trustworthy identity**
Jersey numbers (SoccerMaster's Qwen2.5-VL-7B module; the 72B variant does not fit
one A100), ReID embeddings, goalkeeper/referee handling. Turns anonymous track
ids into named players, which is what makes a report sellable.

**Phase 2 — analytics that a coach argues with**
Formation detection over time, pressing/PPDA, defensive line height, pitch
control, xT. All pure functions of Match State.

**Phase 3 — 90 minutes at production scale**
Off Colab. ~135k frames per camera per match; chunked and resumable on a rented
persistent GPU.

**Phase 4 — the coach app, then capture hardware**
Web replay, clip export, Vietnamese-language reports. Hardware only once clubs
are paying.

---

## Honest limits

* **Colab is for prototyping and finetuning, not production.** Sessions die; a
  match is 400k+ frames across three cameras.
* **A fixed homography is only valid for a fixed camera.** On broadcast footage
  the calibration in this repo is wrong the moment the camera pans. Per-frame
  calibration (SoccerMaster's `KeypointsDetection` head) is the Phase-1 fix.
* **Sprint thresholds (HSR > 5.5 m/s, sprint > 7.0 m/s) are senior-football
  conventions, not physics.** Lower them for youth football.
* **Numbers go to coaches only after `validate()` is clean and the team montage
  has been eyeballed.** A tracking bug looks exactly like a tactical insight.
* **Never scale a clip's distance to 90 minutes.** A 4-minute passage is not a
  match and players do not sustain its intensity.
* **Distance must be integrated from smoothed positions.** Summing raw
  frame-to-frame displacement overstated it by 1.9x on real annotated football,
  because jitter never cancels — every wobble adds. Synthetic tests could not
  catch this; only real data did.

---

## Test data

No club footage yet, so the pipeline is validated against open datasets.

**Alfheim / Tromsø IL** — `python scripts/fetch_alfheim.py --minutes 2 --view panorama`
Three *stationary* cameras covering the whole pitch, available individually or
as a stitched panorama: the same optical situation as a Bepro rig. Ships ZXY
body-sensor positions at 20 Hz, so our output can be scored in real metres.
Fully open, no registration. Verified from the bitstream: 1280x960 @ 30 fps per
camera, 943 chunks (~47 min) per half.
*Caveats:* only Tromsø players wore sensors, so ground truth covers ~11 players
and cannot validate team assignment; and it is 2013 footage, so detection rates
here are a pessimistic floor.
Both the video chunk filenames and the sensor rows carry wall-clock timestamps,
which is what makes frame-accurate scoring possible.

**SoccerTrack v2** — <https://huggingface.co/datasets/atomscott/soccertrack-v2>
**The primary benchmark.** Ten full matches (934 min) on fixed **BePro panoramic
cameras**, 3840x1504 @ 25 fps, with complete per-frame ground truth: all 22
players in every frame, pitch coordinates, jersey identities, roles, team sides
and ball action events. CC-BY 4.0, gated (request access on the page).
Loader: `footlytics/data/soccertrack.py`. Start with `mot/clips/<id>.mp4` — a
4-minute clip with paired MOT ground truth, small enough to iterate on.

*The image-to-pitch chain is not a single homography* — the dataset ships fisheye
coefficients alongside it, because a panorama genuinely is not a pinhole view:

    raw px -> cv2.fisheye.undistortPoints(K, D, P=Knew) -> H^-1 -> metres

Verified: 99.5% of annotated feet land inside the pitch through this chain, and
0-3% through any other combination. Two traps: the dataset README says the origin
is the centre circle but the shipped homography outputs a **corner** origin (a
52.5 m error that looks entirely plausible downstream); and `H` maps pitch->image,
so image->pitch needs `H^-1`.

**Soccer Factory** (SoccerMaster's own, ungated Dropbox) — 7,000 clips with
per-frame boxes, roles, jersey numbers and camera K/R. Broadcast-style, so it
is for *finetuning the detector*, not for validating fixed-camera geometry.

---

## Lineage and licence

This repository is `thanhtrnnn/footlytics`. It started as a flat V0 prototype for moving
cameras (`AnchoredCamera`, a dedicated ball pass, quality reports, a CLI) and was rebuilt on
the Match State architecture from [`scalliontor/Footlytics`](https://github.com/scalliontor/Footlytics)
(`upstream`), which contributed the data contract, the Laws-derived pitch model, tiled
detection, tracking in pitch metres, identity resolution and the analytics layer. The V0
prototype is in this repository's history, before the migration commit; `docs/prs/` records
which pieces were bridged across and what each one measured.

`ultralytics` is AGPL-3.0. Research use only until it is licensed or swapped for an
Apache-2.0 detector — see the licence row in `docs/feasibility.md`.

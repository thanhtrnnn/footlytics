# Benchmark design

Date: 2026-10-11. Status: approved in conversation, awaiting written-spec review.
Revised 2026-10-11: host moved to `iec4090_2` (see Host).

## Purpose

Two jobs, one harness:

1. **Engineering compass.** One command scores every pipeline stage after a change, so a
   calibration, detection or identity change is judged by a number instead of a contact sheet.
2. **External credibility.** The standard metric on the standard broadcast dataset, comparable
   to published results, for clubs and investors.

Analyst acceptance (precision of tactical sequences against an analyst's labels) is out of
scope here; the harness is built so it can be added as another suite later.

## Decisions

- **Approach:** a thin harness around the official evaluators. Our output is converted to
  SoccerNet formats and scored by their code; we do not reimplement HOTA or the calibration
  metric, and we do not port the pipeline into TrackLab.
- **Everything runs on the lab 4090 `iec4090_2`.** The M2 only triggers runs and pulls
  results. Data and weights live on that host only, under `~/footlytics-bench/`.
- **Primary score is GS-HOTA with jersey matching off.** The pipeline has no shirt-number OCR;
  with jersey on, every player track is a miss plus a false positive and the score measures
  the missing feature. The official jersey-on score is reported next to it, labelled.
- **The test split is for milestones only**, never for tuning thresholds.

## Host

`iec4090_2` (ssh alias for `iec@10.170.100.152`). `iec4090` is not used: its NVIDIA driver
libraries and kernel module disagree until its admin reboots it, and we have no sudo there.

- **Hardware:** RTX 4090 24 GB (driver 555.42.06), 20 cores, 31 GB RAM, one 915 GB disk.
- **Shared lab account.** Other people log in as `iec` and run their own jobs (on 2026-10-11:
  42 sessions, three other tmux sessions, a 2.6 GB process on the GPU). So:
  - everything we create lives under `~/footlytics-bench/`, and our tmux sessions are named
    `footlytics-<run-id>`; we never touch other files, sessions, Docker volumes or images;
  - GPU memory is not ours alone: a run checks free VRAM before it starts and fails cleanly
    with that reason instead of dying mid-clip;
  - seconds per match minute is noisy, so every row also records the other processes' GPU
    memory and the load average at the start and end of the run.
- **Python:** system `python3` 3.12 already has torch 2.7.1+cu128 and torchvision 0.22.1 with
  CUDA working (installed in the shared account's `~/.local`, so another user could change it).
  The pipeline venv is built with `--system-site-packages` and must not install a second torch:
  it is filled with `pip`, because `uv pip` ignores system site-packages and would pull a new
  torch. Checked after install and at the start of every run: `torch.__version__ ==
  "2.7.1+cu128"` and CUDA available. `uv` lives at `~/footlytics-bench/bin/uv`. The evaluators
  get their own venv.
- **Disk:** 41 GB free on 2026-10-11, after a user-approved prune of Docker build cache and
  dangling images. The disk is shared: downloads and runs check free space first and stop
  below a 1.5 GB floor. Budget:

  | Item | Size |
  |---|---|
  | GSR valid zip | 11.2 GB |
  | GSR test zip (milestone only) | 8.9 GB |
  | SN-Calibration valid + test | 1.0 GB |
  | SoccerMaster + detector weights, venvs | about 5 GB |
  | Run outputs (predictions, logs) | about 1 GB per nightly run, pruned to the last 5 |

  GSR images are read straight out of the zips (the frames are stored JPEGs) and only the
  label files are extracted, so the data is not held twice. GSR train and challenge are not
  downloaded.
- **SSH:** key login is not set up yet; commands ride on a ControlMaster session the user opens
  with a password. When it drops, the run itself is unaffected (it lives in tmux on the host);
  only polling and the result pull stop, and `footlytics bench --resume <run-id>` picks them up
  after the user reconnects. Suggested permanent fix: `ssh-copy-id` with the `iec220` key, plus
  `IdentityFile` and `ServerAliveInterval 60` in `~/.ssh/config`.

## Suites

| Suite | Data | Metrics |
|---|---|---|
| quick | 6 pinned GSR valid clips | GS-HOTA (jersey off) with DetA/AssA, team accuracy, ball recall and px error, tactical-feature error (m), seconds per match minute |
| quick | 60 pinned SN-Calibration valid frames | JaC@5 x completeness, median reprojection px |
| quick | 6 pinned GSR valid clips, calibrated frame by frame (keyframes + camera tracker) | JaC@5 x completeness on the GSR pitch-line labels, share of labelled frames calibrated, keyframe acceptance |
| quick | V.League mini-GT: 3 matches x 15 frames, half held out | landmark error (m) on landmarks the solver did not use, % frames under 1 m, frame-to-frame jitter (m) |
| quick | synthetic tests; possession rule on SoccerTrack v2 GT positions (when access arrives) | pass/fail; possession accuracy vs `ballStatus` |
| nightly | all 58 GSR valid clips | as quick, plus run-to-run noise floor (same commit twice) |
| milestone | 49 GSR test clips | GS-HOTA jersey off, re-scored baseline, official jersey-on GS-HOTA |
| milestone | SN-Calibration test | JaC@5 x completeness, compared with test-split papers (PnLCalib, SoccerMaster) |

GSR valid comes from only 3 games (ids 2, 3, 5), so its 58 clips are diverse in situations but
not in stadiums or broadcasters; SN-Calibration valid spans 55 games.

Median reprojection px is ours (auxiliary): for each labelled point of a straight ground-plane
line, the distance to the predicted camera's projection of that line, at 960x540, taking the
better of the labels and their mirror as the official evaluator does.

Every row records git SHA, suite, host, per-clip and aggregate numbers, seconds per match
minute, and the host contention readings above.

Reference points (all GSR): baseline 22.26 test / 18.05 valid (arXiv 2404.11335); challenge
winners about 64 on the hidden challenge split (2024, 2025 reports); SoccerMaster 64.1 on test
(arXiv 2512.11016). Calibration: PnLCalib 51.8 and SoccerMaster 56.2 on the test split.

Measured on this harness (2026-10-11, run `20261011-113649-harness-3b32aa6`): GSR ground truth
through our adapters scores 100.00 on all 58 valid clips; the sn-gamestate baseline predictions
re-score to **23.09** jersey-on on the 49 test clips (published 22.26; those predictions date from
March 2024, before the v1.3 labels) and **38.34 with jersey off**, the baseline to beat on our
primary metric.

Calibration, measured on this harness (2026-10-12; SoccerMaster heads + PnLCalib solver at
PnLCalib's thresholds with line refinement; clips solve every 10th frame and bridge with the
camera tracker). All rows carry `leakage: unverified` and are internal until SoccerMaster's
training games are known:

| Set | Score (JaC@5 x completeness) | JaC@5 | Completeness | Median px |
|---|---|---|---|---|
| SN-Calibration valid, 60 quick frames | 54.4 | 71.0 | 76.7 % | 2.07 |
| SN-Calibration test, 3,141 frames (milestone) | **48.8** | 71.5 | 68.2 % | 1.92 |
| GSR valid, 6 quick clips, every frame | 56.8 | 56.8 | 100 % | |

Published on SN-Calibration test: PnLCalib 51.8, SoccerMaster 56.2. A third of our test frames
get no camera (too few visible keypoints for the solver), which caps the score; raising
completeness is the obvious lever. Clip scores vary widely by clip (44 to 86 on the quick set,
72.5 on the two tuning clips).

## Validity rules

- **Teams:** GSR scores team as left/right of the camera. HOME/AWAY is mapped to left/right per
  clip from our own output, never from ground truth: a team defends the goal its goalkeeper
  stands in front of, so the goalkeepers' mean x decides; a team with no goalkeeper sighting
  falls back to its mean x. GSR x is camera-relative, like the Match State's.
- **Calibration output:** the SN-Calibration evaluator needs full camera parameters (it scores
  goal posts and crossbars off the ground plane), so the calibrator exports PnLCalib's full
  camera, not only the ground homography.
- **Leakage:** SoccerMaster was pretrained on SoccerNet-v2 matches. Before quoting any external
  number, check which GSR / SN-Calibration games overlap its training games and report the
  score on the non-overlapping subset alongside. Its training games are not published, so until
  they are, calibration rows carry `leakage: unverified` and their numbers are internal only.
  The SN-Calibration task scores the non-overlapping subset as soon as a list is given.
- **V.League labels:** scored on held-out landmarks; about 10 frames labelled twice (two people
  or two sessions) to measure labelling error; the held-out half is frozen once labelled.
- **Noise:** the nightly suite runs the same commit twice; a change smaller than that spread is
  reported as no change.
- **Footage:** SoccerNet frames are copyrighted broadcast; used for internal scoring only.

## Architecture

```
M2:    footlytics bench --suite quick --remote iec4090_2
         -> rsync the current commit to ~/footlytics-bench/src/, start the run in tmux session
            footlytics-<run-id>, poll, pull results back
       footlytics bench --resume <run-id>    re-attach polling and pull after an SSH drop

iec4090_2:  ~/footlytics-bench/{src,data,weights,runs,venv,venv-eval}
            footlytics/bench/
         suites/{quick,nightly,milestone}.yaml   pinned clip and frame ids
         datasets.py      locate or download GSR (>= v1.3), SN-Calibration, V.League mini-GT
         adapters/gsr.py  Match State -> GSR predictions (role, left/right team, null jersey);
                          GSR ground truth -> Match State (for ball and tactical metrics)
         adapters/calib.py  our camera -> SN-Calibration prediction files
         evaluators.py    official evaluators as subprocesses in their own venv
                          (sn-trackeval is MIT, sn-calibration has no licence: run, never imported)
         metrics.py       ball recall/px, tactical-feature error, landmark error, jitter, s/min
         runner.py        suite loop: pipeline -> adapters -> evaluators -> metrics -> results

bench/results/results.jsonl   one row per run, committed
bench/results/SCOREBOARD.md   regenerated from the JSONL, committed
```

A clip that crashes is recorded as failed with its reason and the suite continues; a run with
any failed clip is marked partial and is not compared with complete runs.

## Proving the harness

Before any number is trusted:

1. GSR ground truth through our adapter scores close to 100 GS-HOTA.
2. The published baseline outputs re-score to 22.26 on test with jersey on. Source: the JSON
   predictions shipped in sn-gamestate (`examples_predictions/SoccerNetGS-test.zip`, pinned by
   sha256); the Zenodo 11065177 tracker state is a pickle and is only the fallback.
3. Unit tests: adapter round trip on a tiny sample, the left/right mapping, each metric on
   synthetic data.

## Build order

1. Host environment on `iec4090_2` (uv in the home dir, pipeline venv on the system torch,
   evaluator venv), GSR valid and SN-Calibration valid downloads, and the two harness checks
   above. The baseline re-score is on test, so the GSR test zip is downloaded here too; its
   labels are extracted and its images stay unread until a milestone.
2. Calibration benchmark on SN-Calibration, together with keyframed per-frame calibration
   (calibrate every N frames with one RANSAC setting, gate at 25 px reprojection, bridge with
   the camera tracker with the score bug and channel logo masked). GSR clips are broadcast with
   moving cameras, so hand landmarks per clip are not an option.
3. Full-pipeline GSR runs: GS-HOTA, teams, ball, tactical features.
4. V.League mini-GT once two more matches are on the host.

## Blockers and constraints

- `iec4090_2` is a shared machine: disk, GPU memory and timing are subject to other users
  (handled under Host). Runs depend on the user's SSH session until key login is set up.
- The SoccerMaster code repository has no licence: internal R&D only. Weights are Apache-2.0.
- SoccerTrack v2 is gated; the possession check waits for access.

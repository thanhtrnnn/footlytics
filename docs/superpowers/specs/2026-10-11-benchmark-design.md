# Benchmark design

Date: 2026-10-11. Status: approved in conversation, awaiting written-spec review.

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
- **Everything runs on the 4090** (`ssh iec4090`). The M2 only triggers runs and pulls results.
  Data and weights live on the 4090 only.
- **Primary score is GS-HOTA with jersey matching off.** The pipeline has no shirt-number OCR;
  with jersey on, every player track is a miss plus a false positive and the score measures
  the missing feature. The official jersey-on score is reported next to it, labelled.
- **The test split is for milestones only**, never for tuning thresholds.

## Suites

| Suite | Data | Metrics |
|---|---|---|
| quick | 6 pinned GSR valid clips | GS-HOTA (jersey off) with DetA/AssA, team accuracy, ball recall and px error, tactical-feature error (m), seconds per match minute |
| quick | 60 pinned SN-Calibration valid frames | JaC@5 x completeness, median reprojection px |
| quick | V.League mini-GT: 3 matches x 15 frames, half held out | landmark error (m) on landmarks the solver did not use, % frames under 1 m, frame-to-frame jitter (m) |
| quick | synthetic tests; possession rule on SoccerTrack v2 GT positions (when access arrives) | pass/fail; possession accuracy vs `ballStatus` |
| nightly | all 58 GSR valid clips | as quick, plus run-to-run noise floor (same commit twice) |
| milestone | 49 GSR test clips | GS-HOTA jersey off, re-scored baseline, official jersey-on GS-HOTA |
| milestone | SN-Calibration test | JaC@5 x completeness, compared with test-split papers (PnLCalib, SoccerMaster) |

Every row records git SHA, suite, host, per-clip and aggregate numbers, and seconds per match
minute.

Reference points (all GSR): baseline 22.26 test / 18.05 valid (arXiv 2404.11335); challenge
winners about 64 on the hidden challenge split (2024, 2025 reports); SoccerMaster 64.1 on test
(arXiv 2512.11016). Calibration: PnLCalib 51.8 and SoccerMaster 56.2 on the test split.

## Validity rules

- **Teams:** GSR scores team as left/right of the camera. HOME/AWAY is mapped to left/right per
  clip from the teams' mean x after direction normalisation, computed from our own output,
  never from ground truth.
- **Calibration output:** the SN-Calibration evaluator needs full camera parameters (it scores
  goal posts and crossbars off the ground plane), so the calibrator exports PnLCalib's full
  camera, not only the ground homography.
- **Leakage:** SoccerMaster was pretrained on SoccerNet-v2 matches. Before quoting any external
  number, check which GSR / SN-Calibration games overlap its training games and report the
  score on the non-overlapping subset alongside.
- **V.League labels:** scored on held-out landmarks; about 10 frames labelled twice (two people
  or two sessions) to measure labelling error; the held-out half is frozen once labelled.
- **Noise:** the nightly suite runs the same commit twice; a change smaller than that spread is
  reported as no change.
- **Footage:** SoccerNet frames are copyrighted broadcast; used for internal scoring only.

## Architecture

```
M2:    footlytics bench --suite quick --remote iec4090
         -> rsync the current commit to the 4090, start the run in tmux, poll, pull results back

4090:  footlytics/bench/
         suites/{quick,nightly,milestone}.yaml   pinned clip and frame ids
         datasets.py      locate or download GSR (>= v1.3), SN-Calibration, V.League mini-GT
         adapters/gsr.py  Match State -> GSR predictions (role, left/right team, null jersey);
                          GSR ground truth -> Match State (for ball and tactical metrics)
         adapters/calib.py  our camera -> SN-Calibration prediction files
         evaluators.py    official evaluators as subprocesses in their own venv
                          (sn-trackeval, sn-calibration; GPL code is run, never imported)
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
2. The published baseline outputs (Zenodo 11065177) re-score to 22.26 on test with jersey on.
3. Unit tests: adapter round trip on a tiny sample, the left/right mapping, each metric on
   synthetic data.

## Build order

1. 4090 environment (uv, CUDA torch), datasets, and the two harness checks above.
2. Calibration benchmark on SN-Calibration, together with keyframed per-frame calibration
   (calibrate every N frames with one RANSAC setting, gate at 25 px reprojection, bridge with
   the camera tracker with the score bug and channel logo masked). GSR clips are broadcast with
   moving cameras, so hand landmarks per clip are not an option.
3. Full-pipeline GSR runs: GS-HOTA, teams, ball, tactical features.
4. V.League mini-GT once two more matches are on the 4090.

## Blockers and constraints

- The 4090 GPU is unusable until its admin reboots or reloads the NVIDIA kernel module (driver
  libraries 595.99.02, loaded module 595.91.07). Steps that need no GPU can proceed.
- The SoccerMaster code repository has no licence: internal R&D only. Weights are Apache-2.0.
- SoccerTrack v2 is gated; the possession check waits for access.

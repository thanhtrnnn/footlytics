# Benchmark scoreboard

Regenerated from `results.jsonl` by `footlytics bench scoreboard`; do not edit.
`partial` runs had a failed clip or task and are not comparable with complete runs.

## calib-tune

| finished (UTC) | commit | status | host | tune-clip-every-10 | tune-clip-every-25 | tune-low-thresholds-refine | tune-pnl-thresholds | tune-pnl-thresholds-refine |
|---|---|---|---|---|---|---|---|---|
| 2026-10-11 16:16 | 8ddcb58 | complete | iec4090_2 | score 72.46 | score 65.09 | score 45.35 | score 48.79 | score 49.00 |
| 2026-10-11 14:13 | 8ddcb58 | failed | iec4090_2 | error | error | error | error | error |
| 2026-10-11 14:06 | 18af644 | partial | iec4090_2 | error | error | error | score 48.79 | error |
| 2026-10-11 13:55 | 2832d80 | partial | iec4090_2 | error | error | score 45.35 | score 48.79 | score 49.00 |

## harness

| finished (UTC) | commit | status | host | gsr-baseline-test | gsr-gt-roundtrip-valid |
|---|---|---|---|---|---|
| 2026-10-11 11:39 | 3b32aa6 | complete | iec4090_2 | hota_jersey_on 23.09 ✓ | hota_jersey_off 100.00 ✓ |

## milestone

| finished (UTC) | commit | status | host | gsr-baseline-test | gsr-pipeline-test | sncalib-test |
|---|---|---|---|---|---|---|
| 2026-10-11 20:28 | f6de620 | complete | iec4090_2 | hota_jersey_on 23.09 | hota_jersey_off 49.11 | score 48.78 |
| 2026-10-11 17:51 | 1879173 | complete | iec4090_2 |  |  | score 48.78 |

## nightly

| finished (UTC) | commit | status | host | gsr-calibration-valid-all | gsr-pipeline-valid-all | gsr-pipeline-valid-all-repeat |
|---|---|---|---|---|---|---|
| 2026-10-12 09:57 | 2fe73f3 | complete | iec4090_2 | score 59.42 | hota_jersey_off 45.69 | hota_jersey_off 45.69 |

## quick

| finished (UTC) | commit | status | host | gsr-calibration-valid-6 | gsr-pipeline-valid-6 | sncalib-valid-60 |
|---|---|---|---|---|---|---|
| 2026-10-12 05:17 | d2500aa | complete | iec4090_2 | score 56.70 | hota_jersey_off 42.38 | score 54.43 |
| 2026-10-11 19:10 | e5bea5b | complete | iec4090_2 | score 56.70 | hota_jersey_off 42.38 | score 54.43 |
| 2026-10-11 17:12 | 6fea8be | complete | iec4090_2 | score 56.75 |  | score 54.43 |

## vleague

| finished (UTC) | commit | status | host | vleague-dev | vleague-frozen |
|---|---|---|---|---|---|
| 2026-10-13 05:07 | edbbaa2 | complete | iec5090 | failed 0.00 | failed 0.00 |
| 2026-10-12 22:52 | fccd3d4 | complete | iec4090 | failed 0.00 | failed 0.00 |
| 2026-10-12 19:51 | fccd3d4 | failed | iec4090 | error | error |

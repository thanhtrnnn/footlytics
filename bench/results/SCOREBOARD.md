# Benchmark scoreboard

Regenerated from `results.jsonl` by `footlytics bench scoreboard`; do not edit.
`partial` runs had a failed clip or task and are not comparable with complete runs.

## calib-tune

| finished (UTC) | commit | status | host | tune-clip-every-10 | tune-clip-every-25 | tune-low-thresholds-refine | tune-pnl-thresholds | tune-pnl-thresholds-refine |
|---|---|---|---|---|---|---|---|---|
| 2026-10-11 16:16 | 8ddcb58 | complete | iec4090_2 | calibrated_share 1.00 | calibrated_share 1.00 | completeness 75.88 | completeness 67.34 | completeness 67.34 |
| 2026-10-11 14:13 | 8ddcb58 | failed | iec4090_2 | error | error | error | error | error |
| 2026-10-11 14:06 | 18af644 | partial | iec4090_2 | error | error | error | completeness 67.34 | error |
| 2026-10-11 13:55 | 2832d80 | partial | iec4090_2 | error | error | completeness 75.88 | completeness 67.34 | completeness 67.34 |

## harness

| finished (UTC) | commit | status | host | gsr-baseline-test | gsr-gt-roundtrip-valid |
|---|---|---|---|---|---|
| 2026-10-11 11:39 | 3b32aa6 | complete | iec4090_2 | hota_jersey_on 23.09 ✓ | hota_jersey_off 100.00 ✓ |

## milestone

| finished (UTC) | commit | status | host | gsr-baseline-test | gsr-pipeline-test | sncalib-test |
|---|---|---|---|---|---|---|
| 2026-10-11 20:28 | f6de620 | complete | iec4090_2 | assa_jersey_off 43.22 | assa_jersey_off 48.86 | completeness 68.23 |
| 2026-10-11 17:51 | 1879173 | complete | iec4090_2 |  |  | completeness 68.23 |

## quick

| finished (UTC) | commit | status | host | gsr-calibration-valid-6 | gsr-pipeline-valid-6 | sncalib-valid-60 |
|---|---|---|---|---|---|---|
| 2026-10-11 19:10 | e5bea5b | complete | iec4090_2 | calibrated_share 1.00 | assa_jersey_off 45.77 | completeness 76.67 |
| 2026-10-11 17:12 | 6fea8be | complete | iec4090_2 | calibrated_share 1.00 |  | completeness 76.67 |

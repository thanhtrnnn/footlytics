# PR 4: small fixes found while porting (independent of PRs 1-3)

Branch `fix/small-upstream` (1 commit directly on `main`). Each item was verified in the code, not inferred.

> Landed in this repository with the migration to the Match State architecture.
> These four PRs were written against `scalliontor/Footlytics`; kept here as the
> record of what was bridged across from the V0 prototype and what each change measured.

| Where | Problem | Fix |
|---|---|---|
| `perception/track.py` `Track.confirmed` | compared `hits` against a literal 3 while `active()` used `cfg.min_hits`; the two diverge as soon as `min_hits` changes | `Track.min_hits` set from `TrackerConfig.min_hits` at creation |
| `perception/track.py` `resolve_roles` | `winner[short] = winner[short]` is a self-assignment, so `min_frames` had no effect | tracks under `min_frames` keep their per-frame roles; long tracks are voted |
| `analytics/tactics.py` `ppda` | docstring promises a front-60% `press_zone`; the body never applies it and the 12-class event feed has no location | `press_zone=None`; passing a value raises `NotImplementedError` with the reason; docstring says WHOLE PITCH |
| `config.py` | `SIGLIP_MODEL` referenced nowhere (`teams.py` uses an HSV histogram) | removed |
| `.gitignore` | `/weights/` and `calib/_synthetic*` listed twice | deduped |

Not changed, worth a look by the author: `geometry/lines_nn.py` has `NUM_LINES = 24` but `LINE_CLASSES` lists 23 names, so heatmap channel 23 has no label; `scoccer.ipynb` installs `supervision`, which nothing imports.

Tests: `tests/test_small_fixes.py` (3 tests) plus the synthetic harnesses unchanged.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

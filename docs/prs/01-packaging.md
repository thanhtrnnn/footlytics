# PR 1: packaging (uv, pyproject) and pytest wrappers around the synthetic harnesses

Branch `feat/packaging` (1 commit on top of `main`).

> Landed in this repository with the migration to the Match State architecture.
> These four PRs were written against `scalliontor/Footlytics`; kept here as the
> record of what was bridged across from the V0 prototype and what each change measured.

## What
- `pyproject.toml`: runtime deps pinned to floors (numpy, pandas, scipy, opencv, matplotlib, torch, ultralytics, imageio, huggingface_hub, pyarrow, requests, typer, scikit-learn), `dev` extra with pytest and ruff, `footlytics` console script, pytest `testpaths`.
- `scripts/test_tracker.py`, `scripts/test_teams.py`: functions stay importable; the printed tables only run under `__main__`. Output unchanged when run as scripts.
- `tests/test_upstream_harness.py`: the README numbers as assertions with headroom: exact pinhole recovery, 3 px click noise < 0.2 m, TPS beats plain homography on a stitched view, feet vs box centre; tracker appearance cuts ID switches; teams 11v11 balanced split on distinct kits; viz render and save/load; identity harness runs when SoccerTrack v2 is present.

## Why
`uv venv --python 3.11 && uv pip install -e ".[dev]" && pytest` is now the whole setup. A regression in geometry, tracking or teams fails a test instead of only changing a printed table.

## Verification
`pytest -q` on an M2: 5 passed, 1 skipped (SoccerTrack v2 gated).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

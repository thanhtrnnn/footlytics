"""bench.flicker: every metric on a tiny synthetic table whose answer is known by construction."""
import json

import numpy as np
import pandas as pd
import pytest

from footlytics.bench import flicker
from footlytics.bench.flicker import compute, evaluate

FPS = 25.0


def rows(role, tid, frames, x, y=0.0, team="home", shot=1, live=True, interpolated=False, conf=0.9):
    frames = np.asarray(list(frames))
    n = len(frames)
    return pd.DataFrame(dict(
        frame_idx=frames, period=1, timestamp=frames / FPS, track_id=tid, role=role, team=team,
        det_conf=conf, x=np.broadcast_to(np.asarray(x, float), n).copy(),
        y=np.broadcast_to(np.asarray(y, float), n).copy(), interpolated=interpolated,
        live_wide=live, shot=shot))


def player(tid, frames, x, y=0.0, **kw):
    return rows("player", tid, frames, x, y, **kw)


def ball(tid, frames, x, y=0.0, **kw):
    return rows("ball", tid, frames, x, y, team="unknown", **kw)


def table(*parts):
    return pd.concat(parts, ignore_index=True)


def metrics(*parts, every=10):
    return compute(table(*parts), FPS, every)


def walk(n, step):
    """x positions of a player moving `step` m a frame for n frames."""
    return step * np.arange(n)


# ------------------------------------------------------------------ player steps


def test_player_step_shares_split_by_keyframe_phase():
    x = np.cumsum(np.r_[0.0, np.full(100, 0.3)])      # 7.5 m/s for 100 steps (frames 1..100)
    x[50:] += 0.7                                     # the step into frame 50 is 1.0 m = 25 m/s
    m = metrics(player(1, range(101), x))
    assert m["player_steps"] == 100
    assert m["player_step_gt12"] == pytest.approx(1 / 100)
    assert m["player_step_gt20"] == pytest.approx(1 / 100)
    # frame 50 is a keyframe (50 % 10 == 0): 10 keyframe steps, 90 between
    assert m["player_step_gt12_kf"] == pytest.approx(1 / 10)
    assert m["player_step_gt12_between"] == 0.0
    # the same jump one frame later lands in the between phase
    x2 = np.cumsum(np.r_[0.0, np.full(100, 0.3)])
    x2[51:] += 0.7
    m2 = metrics(player(1, range(101), x2))
    assert m2["player_step_gt12_kf"] == 0.0
    assert m2["player_step_gt12_between"] == pytest.approx(1 / 90)


def test_a_step_of_exactly_the_limit_and_a_gap_are_not_counted():
    m = metrics(player(1, [0, 1, 2], [0.0, 0.48, 0.96]),     # 12.0 m/s: not above 12
                player(2, [0, 1, 5], [0.0, 0.1, 9.0]))       # a 4-frame gap is not a step
    assert m["player_steps"] == 3
    assert m["player_step_gt12"] == 0.0


def test_raw_acceleration_p99_is_the_step_change_times_fps_squared():
    x = np.cumsum(np.r_[0.0, np.full(100, 0.3)])
    x[50:] += 0.7
    m = metrics(player(1, range(101), x))
    # steps 0.3, 1.0, 0.3 around frame 50: |change| 0.7 m/frame -> 0.7 * 25^2 = 437.5 m/s^2
    assert m["player_accel_p99"] == pytest.approx(437.5)
    assert metrics(player(1, range(101), walk(101, 0.3)))["player_accel_p99"] == pytest.approx(0, abs=1e-6)


def test_steps_do_not_cross_a_shot_and_replay_frames_are_ignored():
    cut = metrics(player(1, [0, 1], [0.0, 30.0], shot=[1, 2]))
    assert cut["player_steps"] == 0
    live = [True, True, False, False, True, True]
    m = metrics(player(1, range(6), [0.0, 0.1, 50.0, 90.0, 0.2, 0.3], live=live))
    assert m["live_frames"] == 4 and m["player_step_gt12"] == 0.0 and m["player_steps"] == 2


# ------------------------------------------------------------------ common mode


def _crowd(n_frames=50, shift_at=None, shift=1.0, who=range(6)):
    parts = []
    for i in range(6):
        x = np.full(n_frames, 5.0 * i)
        if shift_at is not None and i in who:
            x[shift_at:] += shift
        parts.append(player(i + 1, range(n_frames), x))
    return parts


def test_common_mode_step_is_the_median_across_players_and_p99_is_on_keyframes():
    m = metrics(*_crowd(shift_at=20))                  # everyone moves 1 m together at frame 20
    assert m["common_mode_frames"] == 49
    assert m["common_mode_gt05"] == pytest.approx(1 / 49)
    assert m["common_mode_gt05_kf"] == pytest.approx(1 / 4)          # keyframes 10, 20, 30, 40
    assert m["common_mode_gt05_between"] == 0.0
    assert m["common_mode_p99_kf"] == pytest.approx(0.97)            # [0, 1, 0, 0] interpolated
    assert m["common_mode_p99_between"] == pytest.approx(0.0)


def test_one_player_jumping_is_not_common_mode():
    m = metrics(*_crowd(shift_at=20, shift=3.0, who=[2]))
    assert m["common_mode_gt05"] == 0.0 and m["player_step_gt12"] > 0


def test_a_frame_with_fewer_than_five_matched_players_has_no_common_mode():
    parts = _crowd(n_frames=10)[:4]
    assert metrics(*parts)["common_mode_frames"] == 0


# ------------------------------------------------------------------ still jitter


def test_still_player_jitter_is_the_detrended_rms_of_a_window():
    wobble = 0.05 * (-1.0) ** np.arange(60)            # +-5 cm every frame, two windows
    m = metrics(player(1, range(60), wobble), player(2, range(60), walk(60, 0.1)))
    assert m["still_windows_total"] == 4 and m["still_windows"] == 2     # the walker drifts 3 m
    assert m["still_sigma_p50"] == pytest.approx(0.05, abs=0.003)
    assert m["still_sigma_p90"] == pytest.approx(0.05, abs=0.003)


def test_a_slow_steady_drift_is_detrended_away_and_gaps_break_windows():
    drift = 0.01 * np.arange(30)                       # 0.29 m over the window: still, and a straight line
    m = metrics(player(1, range(30), drift))
    assert m["still_windows"] == 1 and m["still_sigma_p50"] == pytest.approx(0.0, abs=1e-9)
    # 29 frames then a hole then 29: no window of 30 contiguous frames
    m = metrics(player(1, list(range(29)) + list(range(30, 59)), 0.0))
    assert m["still_windows"] == 0 and m["still_sigma_p50"] is None


# ------------------------------------------------------------------ ball


def test_ball_step_gt40_counts_real_one_frame_steps_only():
    x = np.cumsum(np.r_[0.0, np.full(9, 0.5)])         # 12.5 m/s
    x[5:] += 1.5                                       # the step into frame 5 is 2.0 m = 50 m/s
    m = metrics(ball(7, range(10), x), player(1, range(10), 0.0))
    assert m["ball_steps"] == 9 and m["ball_step_gt40"] == pytest.approx(1 / 9)
    # steps touching an interpolated row are not detections and are left out of this share
    filled = ball(7, range(10), x)
    filled["interpolated"] = [False, False, False, False, True, True, False, False, False, False]
    m = compute(table(filled, player(1, range(10), 0.0)), FPS, 10)
    assert m["ball_step_gt40"] == 0.0 and m["ball_steps"] == 6   # 3 of the 9 steps touch a filled row


def test_beyond_gate_counts_same_id_steps_above_the_link_gate_including_interpolated():
    # 1.6 m in a frame is 40 m/s: above the new gate (1.49 m at 25 fps), below the old (2.4 m)
    far = ball(7, [0, 1, 2], [0.0, 1.6, 3.2])
    m = metrics(far, player(1, range(3), 0.0))
    assert m["ball_beyond_gate"] == 2 and m["ball_beyond_gate_legacy"] == 0
    # a different id is a segment break the linker declared: not a violation of it
    broke = table(ball(7, [0, 1], [0.0, 1.6]), ball(8, [2], [3.2]))
    assert metrics(broke, player(1, range(3), 0.0))["ball_beyond_gate"] == 1
    # a gap is judged on the elapsed time: 4 m in 4 frames (25 m/s) is fine, 8 m is not
    assert metrics(ball(7, [0, 4], [0.0, 4.0]), player(1, range(5), 0.0))["ball_beyond_gate"] == 0
    leap = metrics(ball(7, [0, 4], [0.0, 8.0]), player(1, range(5), 0.0))
    assert leap["ball_beyond_gate"] == 1
    # rows from interpolate_ball carry shot 0 in old states: still counted, and reported apart
    filled = ball(7, [0, 1, 2], [0.0, 7.0, 14.0])
    filled["interpolated"] = [False, True, False]
    filled["shot"] = [1, 0, 1]
    m = metrics(filled, player(1, range(3), 0.0))
    assert m["ball_beyond_gate"] == 2 and m["ball_beyond_gate_interpolated"] == 2


def test_ball_presence_and_gap_lengths():
    b = table(ball(7, range(0, 5), 0.0), ball(7, range(10, 15), 0.0))
    m = metrics(b, player(1, range(20), 0.0))
    assert m["ball_presence"] == pytest.approx(10 / 20)
    assert m["ball_gaps"] == 1 and m["ball_gap_p50"] == 5.0
    assert m["ball_gap_hist"]["2-5"] == 1 and m["ball_gap_hist"]["1"] == 0
    # interpolated rows count as present, but not as detected
    f = ball(7, range(0, 5), 0.0)
    f["interpolated"] = [False, True, True, False, False]
    m = metrics(f, player(1, range(5), 0.0))
    assert m["ball_presence"] == 1.0 and m["ball_presence_detected"] == pytest.approx(3 / 5)
    assert m["ball_interpolated_share"] == pytest.approx(2 / 5)


def test_gaps_do_not_span_a_shot_change():
    b = table(ball(7, range(0, 5), 0.0, shot=1), ball(7, range(20, 25), 0.0, shot=2))
    m = metrics(b, player(1, range(25), 0.0))
    assert m["ball_gaps"] == 0


# ------------------------------------------------------------------ ids, teams, holes


def test_births_per_minute_and_median_id_life():
    f = range(1500)                                       # one minute at 25 fps
    m = metrics(*[player(i, f, 5.0 * i) for i in range(1, 5)], player(5, range(700, 1500), 30.0))
    assert m["minutes"] == pytest.approx(1.0)
    assert m["births_per_min"] == pytest.approx(1.0)      # id 5; the first four were there at the start
    assert m["id_life_median_s"] == pytest.approx(60.0)   # ids live 60, 60, 60, 60 and 32 s
    assert m["id_life_p10_s"] == pytest.approx(32 + 0.4 * 28)         # lives 32, 60, 60, 60, 60


def test_blinks_are_one_to_three_missing_frames_inside_an_id():
    f = list(range(300)) + [302] + list(range(303, 400))                      # 2 frames missing: a blink
    g = list(range(100)) + list(range(105, 400))                              # 5 missing: a loss, not a blink
    h = list(range(100)) + list(range(101, 400))                              # 1 missing: a blink
    parts = [player(1, f, 0.0), player(2, g, 5.0), player(3, h, 10.0)]
    m = metrics(*parts)
    n_rows = len(f) + len(g) + len(h)
    # f (2 missing) and h (1 missing) blink; g (5 missing) was lost, which is not a blink
    assert m["blinks_per_player_min"] == pytest.approx(2 / (n_rows / FPS / 60))


def test_mass_rebirth_frames_are_counted():
    old = [player(i, range(0, 50), 5.0 * i) for i in range(1, 7)]
    new = [player(10 + i, range(50, 100), 5.0 * i) for i in range(1, 7)]
    m = metrics(*old, *new)
    assert m["mass_birth_frames"] == 1
    # births spread over many frames are not a mass rebirth
    spread = [player(i, range(10 * i, 100), 5.0 * i) for i in range(1, 7)]
    assert metrics(*spread)["mass_birth_frames"] == 0


def test_more_than_eleven_on_a_team():
    full = [player(i, range(10), 3.0 * i) for i in range(1, 12)]                       # 11 home
    twelfth = player(12, [4], 99.0)                                                   # one frame of 12
    m = metrics(*full, twelfth)
    assert m["frames_over_11"] == 1 and m["frames_over_11_share"] == pytest.approx(1 / 10)
    assert metrics(*full)["frames_over_11"] == 0
    # 11 home + 11 away is fine
    away = [player(100 + i, range(10), 3.0 * i + 1, team="away") for i in range(1, 12)]
    assert metrics(*full, *away)["frames_over_11"] == 0


def test_same_team_pairs_inside_0_6_m_for_ten_frames_are_duplicates():
    twice = [player(1, range(12), 0.0), player(2, range(12), 0.3)]
    short = [player(3, range(9), 20.0), player(4, range(9), 20.3)]
    other_team = [player(5, range(12), 40.0), player(6, range(12), 40.2, team="away")]
    far = [player(7, range(12), 60.0), player(8, range(12), 61.0)]
    m = metrics(*twice, *short, *other_team, *far)
    assert m["duplicate_pairs"] == 1 and m["duplicate_pair_frames"] == 12
    # the same ids close in different frames only are not a pair
    apart = [player(1, range(12), 0.0), player(2, range(12), np.where(np.arange(12) % 2, 0.3, 30.0))]
    assert metrics(*apart)["duplicate_pairs"] == 0


def test_frames_with_more_than_four_referees():
    refs = [rows("referee", 50 + i, range(3), 10.0 * i, team="official") for i in range(5)]
    m = metrics(*refs[:4], player(1, range(3), 0.0))
    assert m["frames_over_4_referees"] == 0
    m = metrics(*refs, player(1, range(3), 0.0))
    assert m["frames_over_4_referees"] == 3


def test_radar_holes_longer_than_eight_frames_inside_one_shot():
    a = player(1, list(range(50)) + list(range(70, 100)), 0.0)               # 20 frames missing
    m = metrics(a)
    assert m["radar_holes"] == 1 and m["radar_hole_frames"] == 20
    assert m["radar_holes_per_min"] == pytest.approx(1 / m["minutes"])
    # 8 frames missing is not a hole; nor is a gap that a cut explains
    assert metrics(player(1, list(range(50)) + list(range(58, 100)), 0.0))["radar_holes"] == 0
    cut = player(1, list(range(50)) + list(range(70, 100)), 0.0, shot=[1] * 50 + [2] * 30)
    assert metrics(cut)["radar_holes"] == 0


def test_duplicate_frames_are_frames_where_every_player_stayed_put():
    parts = []
    for i in range(6):
        x = np.arange(10, dtype=float)
        x[3:] -= 1.0                                    # no movement into frame 3
        parts.append(player(i + 1, range(10), x + 10 * i))
    m = metrics(*parts)
    assert m["duplicate_frame_share"] == pytest.approx(1 / 9)


# ------------------------------------------------------------------ thresholds, io, display


def test_evaluate_marks_each_threshold_and_leaves_missing_values_unscored():
    checks = evaluate({"player_step_gt12": 0.004, "player_step_gt20": 0.01, "id_life_median_s": 20.0,
                       "ball_beyond_gate": 0})
    assert checks["player_step_gt12"]["passed"] is True
    assert checks["player_step_gt20"]["passed"] is False
    assert checks["id_life_median_s"]["passed"] is True            # >= 20
    assert checks["ball_beyond_gate"]["passed"] is True
    assert checks["still_sigma_p50"]["passed"] is None and checks["still_sigma_p50"]["value"] is None
    assert set(checks) == set(flicker.THRESHOLDS)
    assert flicker.THRESHOLDS["player_step_gt12"] == ("<=", 0.005)


def test_run_dir_writes_flicker_json_and_a_table(tmp_path, capsys):
    t = table(player(1, range(30), walk(30, 0.2)), ball(9, range(30), walk(30, 0.4)))
    d = tmp_path / "match1"
    d.mkdir()
    t.to_parquet(d / "tracks.parquet")
    (d / "meta.json").write_text(json.dumps({"fps": FPS}))
    res = flicker.run_dir(d, every=5)
    saved = json.loads((d / "flicker.json").read_text())
    assert saved["match"] == "match1" and saved["metrics"]["every"] == 5 and saved["metrics"]["fps"] == FPS
    assert saved["total"] == res["total"] and "player_step_gt12" in saved["checks"]
    out = flicker.render([res])
    assert "match1" in out and "player_step_gt12" in out and "PASS" in out
    # without a meta.json the fps comes from the timestamps
    (d / "meta.json").unlink()
    assert flicker.run_dir(d, write=False)["metrics"]["fps"] == FPS


def test_cli_flicker_writes_json_prints_the_table_and_exits_zero_on_failures(tmp_path):
    from typer.testing import CliRunner

    from footlytics.cli import app

    dirs = []
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        # a 50 m/s player: fails the step thresholds
        table(player(1, range(30), walk(30, 2.0))).to_parquet(d / "tracks.parquet")
        (d / "meta.json").write_text(json.dumps({"fps": FPS}))
        dirs.append(d)
    r = CliRunner().invoke(app, ["flicker", str(dirs[0]), str(dirs[1]), "--every", "5"])
    assert r.exit_code == 0, r.output
    assert "player_step_gt12" in r.output and "FAIL" in r.output
    for d in dirs:
        assert json.loads((d / "flicker.json").read_text())["metrics"]["every"] == 5


def _saved_state(path, steps_m):
    path.mkdir()
    table(player(1, range(40), walk(40, steps_m)), ball(9, range(40), walk(40, 0.2))
          ).to_parquet(path / "tracks.parquet")
    (path / "meta.json").write_text(json.dumps({"fps": FPS}))
    return path


def test_flicker_task_kind_reports_pooled_numbers_and_never_gates(tmp_path):
    from footlytics.bench import runner
    from footlytics.bench.results import headline, render_scoreboard

    good = _saved_state(tmp_path / "good", 0.2)          # 5 m/s
    bad = _saved_state(tmp_path / "bad", 2.0)            # 50 m/s
    out = runner.KINDS["flicker"]({"states": [str(good), str(bad)], "every": 10}, None)
    a = out["aggregate"]
    assert out["failed"] == {} and set(out["clips"]) == {"good", "bad"}
    # pooled over steps, not averaged over states: one state's 39 steps of 78 are impossible
    assert a["flicker_player_step_gt12"] == pytest.approx(0.5)
    assert a["flicker_passed"] < a["flicker_total"] <= 10      # a threshold without a value is unscored
    assert headline({"kind": "flicker", "status": "complete", "aggregate": a}) == (
        "flicker_passed", a["flicker_passed"], None)
    row = {"run_id": "r1", "suite": "flick", "git_sha": "abcdef1234567", "host": "h",
           "started_at": "2026-11-27T10:00:00Z", "finished_at": "2026-11-27T10:03:00Z",
           "status": "complete", "tasks": {"flicker-saved": {"kind": "flicker", "status": "complete",
                                                              "aggregate": a}}}
    md = render_scoreboard([row])
    assert "Flicker (reported, not gating)" in md and "| flicker-saved |" in md and "50.00%" in md


def test_flicker_task_kind_fails_one_unreadable_state_alone(tmp_path):
    from footlytics.bench import runner

    good = _saved_state(tmp_path / "good", 0.2)
    out = runner.KINDS["flicker"]({"states": [str(good), str(tmp_path / "missing")]}, None)
    assert list(out["failed"]) == ["missing"] and list(out["clips"]) == ["good"]
    with pytest.raises(RuntimeError):
        runner.KINDS["flicker"]({"states": [str(tmp_path / "missing")]}, None)


def test_flicker_suite_is_valid():
    from footlytics.bench import runner

    s = runner.load_suite("flicker")
    assert s["tasks"] and {t["kind"] for t in s["tasks"]} <= set(runner.KINDS)
    assert all(t["states"] and not t.get("check") for t in s["tasks"])      # reported, never gated


def test_pooling_clips_keeps_ids_frames_and_shots_apart():
    from footlytics.bench.runner import flicker_aggregate

    a = table(player(1, range(30), walk(30, 0.2)))
    b = table(player(1, range(30), walk(30, 0.2) + 50.0))     # same ids and frames, elsewhere
    pooled = flicker_aggregate([a, b], FPS, 10)
    assert pooled["flicker_person_ids"] == 2 and pooled["flicker_player_steps"] == 58
    assert pooled["flicker_player_step_gt12"] == 0.0           # no jump between the clips
    assert flicker_aggregate([], FPS, 10) == {}


def test_flat_keeps_numbers_only():
    m = metrics(player(1, range(5), 0.0), ball(7, range(5), 0.0))
    f = flicker.flat(m)
    assert f["flicker_player_steps"] == 4 and "flicker_ball_gap_hist" not in f
    assert all(isinstance(v, (int, float)) for v in f.values() if v is not None)

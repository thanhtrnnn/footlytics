"""Per-track state smoothing: noise, coasting, robustness, manoeuvres, the schema contract."""
import numpy as np
import pandas as pd

from footlytics.perception.state_smooth import (
    StateSmoothConfig, depth_sensitivity, is_smoothed, measurement_sigma, smooth_player_states,
)
from footlytics.pipeline.stability import step_stats, synthetic_response
from footlytics.state.schema import MatchMeta, MatchState

FPS = 30.0


def track(xy, tid=1, role="player", foot_row=650.0, frames=None, shot=0, live=True, box_x=900.0):
    xy = np.asarray(xy, float)
    n = len(xy)
    f = np.arange(n) if frames is None else np.asarray(frames)
    return pd.DataFrame({
        "frame_idx": f, "period": 1, "timestamp": f / FPS, "track_id": tid, "role": role,
        "team": "home", "jersey": np.nan, "bbox_x": box_x + 0.1 * np.arange(n) % 7,
        "bbox_y": foot_row - 150.0, "bbox_w": 60.0, "bbox_h": 150.0, "det_conf": 0.9,
        "x": xy[:, 0], "y": xy[:, 1], "z": np.nan, "shot": shot, "live_wide": live})


def run(df, **kw):
    out, rep = smooth_player_states(df, FPS, StateSmoothConfig(**kw))
    return out.sort_values(["track_id", "frame_idx"]).reset_index(drop=True), rep


def test_depth_sensitivity_matches_the_measured_points():
    assert abs(float(depth_sensitivity(np.array(0.30))) - 0.23) < 0.01
    assert abs(float(depth_sensitivity(np.array(0.90))) - 0.03) < 0.005


def test_measurement_sigma_grows_with_distance_from_camera():
    cfg = StateSmoothConfig()
    s = measurement_sigma(np.array([0.3, 0.5, 0.7, 0.9]) * 1080.0, 1080.0, cfg)
    assert (np.diff(s) < 0).all() and s[0] > 4 * s[-1] and s[-1] >= cfg.meas_floor


def test_a_still_player_is_quiet_and_the_measurement_is_kept():
    rng = np.random.default_rng(0)
    raw = np.array([10.0, -5.0]) + rng.normal(0, 0.2, (200, 2))
    out, rep = run(track(raw))
    assert np.std(out["x"]) < 0.4 * np.std(raw[:, 0])
    assert np.allclose(out["x_raw"], raw[:, 0], atol=1e-6) and np.allclose(out["y_raw"], raw[:, 1], atol=1e-6)
    assert not out["coasted"].any()
    assert abs(out["x"].mean() - 10.0) < 0.05


def test_far_players_are_smoothed_harder_than_near_ones():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.3, (200, 2))
    near, _ = run(track(noise, foot_row=0.9 * 1080))
    far, _ = run(track(noise, foot_row=0.35 * 1080))
    assert np.std(far["x"]) < 0.8 * np.std(near["x"])


def test_a_short_occlusion_is_coasted_and_flagged():
    t = np.arange(60) / FPS
    xy = np.c_[3.0 * t, np.zeros(60)]                       # 3 m/s along x
    gone = list(range(20, 30))                               # 10 frames = 0.33 s
    df = track(xy)
    df = df[~df.frame_idx.isin(gone)]
    other = track(np.zeros((60, 2)) + 20.0, tid=2)           # someone is seen on every frame
    out, rep = run(pd.concat([df, other], ignore_index=True))
    out = out[out.track_id == 1]
    c = out[out["coasted"]]
    assert sorted(c["frame_idx"]) == gone and rep["coasted_rows"] == 10
    assert np.allclose(c.sort_values("frame_idx")["x"], 3.0 * np.array(gone) / FPS, atol=0.05)
    assert c["x_raw"].isna().all() and (c["det_conf"] == 0).all()
    assert (c["track_id"] == 1).all() and (c["role"] == "player").all()
    assert len(out) == 60


def test_a_frame_nobody_is_seen_in_is_the_cameras_gap_not_an_occlusion():
    df = track(np.c_[np.arange(60) * 0.05, np.zeros(60)])
    df = df[~df.frame_idx.isin(range(20, 26))]               # nothing at all on these frames
    out, rep = run(df)
    assert not out["coasted"].any() and len(out) == 54


def test_a_long_gap_is_not_bridged():
    df = track(np.c_[np.arange(80) * 0.05, np.zeros(80)])
    df = df[~df.frame_idx.isin(range(30, 50))]               # 20 frames > 0.5 s
    out, rep = run(df)
    assert not out["coasted"].any() and rep["segments"] == 2


def test_no_coasting_across_a_shot_change_or_a_replay():
    df = track(np.c_[np.arange(60) * 0.05, np.zeros(60)], shot=0)
    df.loc[df.frame_idx >= 30, "shot"] = 1
    df = df[~df.frame_idx.isin(range(28, 33))]
    out, _ = run(df)
    assert not out["coasted"].any()
    # same shot id either side, but a replay frame in between (other people's rows say so)
    mine = track(np.c_[np.arange(40) * 0.05, np.zeros(40)])
    mine = mine[~mine.frame_idx.isin([18, 19, 20, 21])]
    other = track(np.zeros((40, 2)), tid=2, live=True)
    other.loc[other.frame_idx == 20, "live_wide"] = False
    out2, _ = run(pd.concat([mine, other], ignore_index=True))
    assert not out2[(out2.track_id == 1)]["coasted"].any()


def test_coast_s_zero_turns_coasting_off():
    df = track(np.c_[np.arange(60) * 0.05, np.zeros(60)])
    df = df[~df.frame_idx.isin(range(20, 25))]
    out, rep = run(df, coast_s=0.0)
    assert not out["coasted"].any() and len(out) == len(df)


def test_a_single_wild_measurement_does_not_move_the_state():
    t = np.arange(120) / FPS
    xy = np.c_[2.0 * t, np.zeros(120)]
    z = xy.copy()
    z[60] += [6.0, -6.0]
    out, rep = run(track(z))
    assert abs(out.loc[60, "x"] - xy[60, 0]) < 0.4 and abs(out.loc[60, "y"]) < 0.4
    assert rep["restarts"] == 0
    assert out.loc[60, "x_raw"] == z[60, 0].astype(np.float32)        # the audit column keeps it


def test_a_real_jump_restarts_the_filter_and_is_not_averaged_across():
    t = np.arange(120) / FPS
    z = np.c_[np.zeros(120), np.zeros(120)]
    z[60:] += [8.0, 0.0]                                      # an identity swap: 8 m in one frame
    out, rep = run(track(z))
    assert abs(out.loc[40, "x"]) < 0.1 and abs(out.loc[90, "x"] - 8.0) < 0.1
    # nothing in between is a blend of the two places
    assert np.all((out["x"] < 0.3) | (out["x"] > 7.7))


def test_a_repeated_video_frame_counts_once_and_stays_a_row():
    rng = np.random.default_rng(2)
    z = np.array([4.0, 4.0]) + rng.normal(0, 0.2, (100, 2))
    df = track(z)
    dup = df.copy()
    dup = dup.iloc[1::2].copy()                               # every odd frame repeats the even before
    prev = df.iloc[0::2].iloc[: len(dup)]
    for c in ("bbox_x", "bbox_y", "x", "y"):
        dup[c] = prev[c].to_numpy()
    df.iloc[1::2] = dup.reindex(df.iloc[1::2].index)
    out, rep = run(df)
    assert rep["repeated_frames"] == 50 and len(out) == 100 and not out["coasted"].any()
    assert not np.array_equal(out["x"].to_numpy()[1:3], out["x"].to_numpy()[0:2])   # still a path


def test_ball_and_untracked_rows_pass_through_untouched():
    ball = track(np.c_[np.arange(20) * 1.0, np.zeros(20)], tid=9, role="ball")
    stray = track(np.zeros((20, 2)) + 3.0, tid=-1)
    person = track(np.zeros((20, 2)))
    out, _ = run(pd.concat([ball, stray, person], ignore_index=True))
    b = out[out.role == "ball"].sort_values("frame_idx")
    assert np.array_equal(b["x"].to_numpy(), np.arange(20.0)) and np.array_equal(b["x_raw"], b["x"])
    assert (out[out.track_id == -1]["x"] == 3.0).all() and not out["coasted"][out.role == "ball"].any()


def test_idempotent_columns_and_the_is_smoothed_test():
    rng = np.random.default_rng(3)
    df = track(rng.normal(0, 0.3, (100, 2)))
    assert not is_smoothed(df.assign(x_raw=df.x, y_raw=df.y, coasted=False))
    out, _ = run(df)
    assert is_smoothed(out)


def test_schema_carries_the_new_columns_and_defaults_old_states():
    df = track(np.zeros((10, 2)))
    st = MatchState(MatchMeta("m", FPS), df)
    assert {"x_raw", "y_raw", "coasted"} <= set(st.tracks.columns)
    assert np.allclose(st.tracks["x_raw"], st.tracks["x"]) and not st.tracks["coasted"].any()
    out, _ = run(df.assign(x=np.linspace(0, 1, 10)))
    st2 = MatchState(MatchMeta("m", FPS), out)
    assert st2.tracks["coasted"].dtype == bool


def test_smoothing_leaves_a_long_run_a_physical_path():
    rng = np.random.default_rng(4)
    t = np.arange(600) / FPS
    z = np.c_[5 * np.sin(0.4 * t) * 4, 3 * np.cos(0.3 * t)] + rng.normal(0, 0.3, (600, 2))
    out, _ = run(track(z))
    assert step_stats(out, FPS)["gt12"] < 0.001 and step_stats(pd.DataFrame(track(z)), FPS)["gt12"] > 0.05


def test_manoeuvres_are_followed_not_rounded_off():
    """Sprint start (7 m/s^2) and a 90 degree cut at 6 m/s: the smoothed path stays within 0.3 m
    of the truth and keeps the peak speed, at the image rows where the noise is modest."""
    for row in (0.6, 0.9):
        r = synthetic_response(FPS, row, n_noise=8)
        for name in ("sprint", "cut"):
            m = r[name]
            assert m["noise_free_peak_err_m"] < 0.3, (row, name, m)
            assert 0.85 < m["noise_free_peak_speed_kept"] < 1.2, (row, name, m)
            assert m["noisy_rms_err_m"] < 0.6 * m["raw_rms_err_m"], (row, name, m)


def crowd(n=150, n_players=12, common=None, accel=None, seed=0):
    """`n_players` people at fixed spots, each measured with a little noise; `common` (n, 2) is a
    camera error every one of them shares; `accel` (m/s^2) makes the whole crowd speed up for a
    second from frame 40 (the worst case for a trend filter: real common motion)."""
    rng = np.random.default_rng(seed)
    base = np.c_[np.linspace(-30, 30, n_players), np.linspace(-20, 20, n_players)]
    t = np.arange(n) / FPS
    s = np.zeros(n)
    if accel:
        tt = np.clip(t - 40 / FPS, 0, 1.0)
        v_end = accel * 1.0
        s = np.where(t - 40 / FPS < 1.0, 0.5 * accel * tt ** 2, 0.5 * accel + v_end * (t - 40 / FPS - 1.0))
        s = np.where(t < 40 / FPS, 0.0, s)
    frames, truths = [], []
    for k in range(n_players):
        truth = base[k] + np.c_[s * (1 if k % 2 else -1), np.zeros(n)]      # two teams, opposite ways
        z = truth + rng.normal(0, 0.03, truth.shape) + (0 if common is None else common)
        df = track(z, tid=k + 1, foot_row=800.0 - 20 * k)
        df["bbox_x"] = df["bbox_x"] + 10 * k                  # distinct boxes
        frames.append(df)
        truths.append(truth)
    return pd.concat(frames, ignore_index=True), np.stack(truths)


def test_noise_every_player_in_the_frame_shares_is_taken_off():
    rng = np.random.default_rng(5)
    walk = np.cumsum(rng.normal(0, 0.05, (150, 2)), axis=0)
    walk = walk - np.array([np.convolve(walk[:, a], np.ones(31) / 31, "same") for a in (0, 1)]).T
    common = rng.normal(0, 0.25, (150, 2)) + walk                    # a jittering camera
    df, truth = crowd(common=common)
    on, _ = run(df)
    off, _ = run(df, common_mode=False)

    def err(o):
        o = o.sort_values(["track_id", "frame_idx"])
        got = o[["x", "y"]].to_numpy().reshape(12, 150, 2)
        return float(np.sqrt(np.mean(np.sum((got - truth) ** 2, axis=2))))

    assert err(on) < 0.7 * err(off)
    assert err(on) < 0.1


def test_real_common_motion_is_kept_within_the_trend_filters_bias():
    """Both teams speed up at 1.5 m/s^2 for a second. The filter may take at most about
    a/2 * sigma^2 of it (0.19 m for sigma 0.5 s); and half the crowd running each way
    leaves the median step at zero, so the bias is smaller still."""
    df, truth = crowd(accel=1.5)
    out, _ = run(df)
    got = out.sort_values(["track_id", "frame_idx"])[["x", "y"]].to_numpy().reshape(12, 150, 2)
    assert np.abs(got - truth).max() < 0.25
    # worst case for the filter: everyone the same way
    df2, truth2 = crowd(accel=1.5)
    df2["x"] = df2["x"] + 0.0
    same = df2.copy()
    for k in range(12):
        sel = same.track_id == k + 1
        same.loc[sel, "x"] = truth2[k, :, 0] * 0 + (truth2[0, :, 0] - truth2[0, 0, 0]) + (-30 + 5 * k)
    o2, _ = run(same)
    got2 = o2.sort_values(["track_id", "frame_idx"])["x"].to_numpy().reshape(12, 150)
    want2 = np.stack([(truth2[0, :, 0] - truth2[0, 0, 0]) + (-30 + 5 * k) for k in range(12)])
    assert np.abs(got2 - want2).max() < 0.3


def test_a_whole_frame_jump_is_not_mistaken_for_noise():
    jump = np.zeros((150, 2))
    jump[80:] += [5.0, 0.0]                                   # a cut / camera jump
    df, truth = crowd(common=jump)
    out, _ = run(df)
    x = out.sort_values(["track_id", "frame_idx"])["x"].to_numpy().reshape(12, 150)
    assert np.allclose(x[:, 110] - x[:, 50], 5.0, atol=0.3)

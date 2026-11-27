"""The smoothed player state, end to end through `run_observations`, and what draws it."""
import numpy as np
import pandas as pd

from footlytics.perception.state_smooth import StateSmoothConfig
from footlytics.pipeline.radar import Observation, PipelineConfig, run_observations
from footlytics.state.schema import MatchMeta, MatchState, Role
from footlytics.viz.radar import live_only, plot_frame

PLAYER = Role.PLAYER.value
FPS = 25.0


def scene(n=120, noise=0.25, hide=None, seed=0, extra=None):
    """Six players walking along x at different depths, boxes consistent with the pitch."""
    rng = np.random.default_rng(seed)
    obs, truth = [], []
    for t in range(n):
        base = np.c_[np.arange(6) * 6.0 - 18 + 1.2 * t / FPS, np.arange(6) * 2.0 - 5]
        keep = np.arange(6) if hide is None or t not in hide[1] else np.array(
            [i for i in range(6) if i != hide[0]])
        xy = (base + rng.normal(0, noise, base.shape))[keep]
        truth.append(base)
        dets = np.c_[900 + 60 * keep, np.full(len(keep), 500.0), np.full(len(keep), 30.0),
                     np.full(len(keep), 90.0), np.full(len(keep), 0.9), np.zeros(len(keep))]
        dets[:, 0] += 0.3 * t                                  # boxes never repeat exactly
        obs.append(Observation(dets, [PLAYER] * len(dets), xy))
    return obs, truth


def run(obs, **kw):
    cfg = PipelineConfig(verbose=False, montage=False, stitch=False, touchline_people=False,
                         goalkeepers=False, use_appearance=False, ball_max_gap=0, **kw)
    return run_observations(obs, FPS, MatchMeta(match_id="t", fps=FPS), cfg)


def test_the_emitted_position_is_the_smoothed_state_and_the_measurement_is_kept():
    obs, truth = scene()
    state, report = run(obs)
    t = state.tracks
    assert {"x_raw", "y_raw", "coasted"} <= set(t.columns) and report["state_smoothing"]["segments"] >= 6
    sm = np.hypot(t["x"] - t["x_raw"], t["y"] - t["y_raw"])
    assert sm.median() > 0.02                                  # it moved the state
    # the raw columns are exactly what the tracker was given
    first = obs[10]
    row = t[(t.frame_idx == 10)].sort_values("x_raw")
    assert np.allclose(np.sort(row["x_raw"].to_numpy()), np.sort(first.xy[:, 0]), atol=1e-4)
    assert (t["x"] - t["x_raw"]).abs().max() < 1.5             # smoothed, not displaced


def test_the_smoothed_state_is_steadier_than_the_measurement():
    obs, _ = scene(noise=0.3)
    state, _ = run(obs)
    p = state.tracks[state.tracks["track_id"] >= 0].sort_values(["track_id", "frame_idx"])
    d_raw = p.groupby("track_id")["x_raw"].diff().abs().mean()
    d_sm = p.groupby("track_id")["x"].diff().abs().mean()
    assert d_sm < 0.4 * d_raw


def test_smooth_state_off_is_the_old_behaviour():
    obs, _ = scene()
    state, report = run(obs, smooth_state=False)
    t = state.tracks
    assert "state_smoothing" not in report
    assert np.array_equal(t["x"].to_numpy(), t["x_raw"].to_numpy()) and not t["coasted"].any()


def test_a_short_occlusion_is_carried_and_marked():
    obs, _ = scene(hide=(2, range(50, 58)))
    state, report = run(obs)
    t = state.tracks
    c = t[t["coasted"]]
    assert len(c) == 8 and report["state_smoothing"]["coasted_rows"] == 8
    assert c["track_id"].nunique() == 1 and set(c["frame_idx"]) == set(range(50, 58))
    assert (c["det_conf"] == 0).all() and c["x_raw"].isna().all()
    # six people on every frame again
    assert (t.groupby("frame_idx").size() == 6).all()
    # excludable: the detections alone are what the tracker saw
    assert len(t[~t["coasted"]]) == 6 * 120 - 8


def test_coasting_can_be_turned_off_for_a_detection_benchmark():
    obs, _ = scene(hide=(2, range(50, 58)))
    state, _ = run(obs, state_smooth=StateSmoothConfig(coast_s=0.0))
    assert not state.tracks["coasted"].any() and len(state.tracks) == 6 * 120 - 8


def test_a_person_is_shown_only_once_confirmed_and_loses_nothing_for_it():
    """A 2-frame false alarm is never emitted; a 3-frame track appears with all 3 rows."""
    obs, _ = scene(n=60, noise=0.0)
    for t, n_frames, ax in ((10, 2, 40.0), (30, 3, -40.0)):
        for k in range(n_frames):
            o = obs[t + k]
            xy = np.vstack([o.xy, [[ax + 0.02 * k, 22.0]]])
            dets = np.vstack([o.dets, [1700 + 5 * t, 300.0, 30.0, 90.0, 0.4, 0.0]])
            obs[t + k] = Observation(dets, [PLAYER] * len(dets), xy)
    state, _ = run(obs, min_track_frames=2, smooth_state=False)
    far = state.tracks[state.tracks["y"] > 15]
    assert far["frame_idx"].nunique() == 3 and set(far["frame_idx"]) == {30, 31, 32}
    # min_track_frames=1 keeps everything, as before
    state1, _ = run(obs, min_track_frames=1, smooth_state=False)
    assert state1.tracks[state1.tracks["y"] > 15]["frame_idx"].nunique() == 5


def test_radar_does_not_draw_replays_and_close_ups():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame({
        "frame_idx": [0, 0, 1, 1], "period": 1, "timestamp": 0.0, "track_id": [1, 2, 1, 2],
        "role": PLAYER, "team": "home", "x": [0.0, 5.0, 0.0, 5.0], "y": 0.0,
        "live_wide": [True, True, False, False]})
    state = MatchState(MatchMeta(match_id="t", fps=FPS), df)
    assert len(live_only(state.tracks)) == 2
    for frame, drawn in ((0, 2), (1, 0)):
        _, ax = plt.subplots()
        plot_frame(state, frame, ax=ax, trail_frames=0, show_velocity=False)
        offsets = sum(len(c.get_offsets()) for c in ax.collections)
        assert offsets == drawn, (frame, offsets)
        plt.close("all")
    # a state without the column (fixed rig) is all live
    assert len(live_only(df.drop(columns=["live_wide"]))) == 4

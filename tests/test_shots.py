"""shots.py: keep replays, close-ups and graphics out of the tactical layer."""
import numpy as np
import pandas as pd

from footlytics.analytics.sequences import detect_sequences
from footlytics.perception.shots import ShotConfig, frame_signals, live_wide_segments
from footlytics.state.schema import MatchMeta, MatchState, Role, Team, TeamInfo

H, W = 540, 960
GRASS = (60, 140, 50)          # BGR, inside the pitch mask's hue band


def _wide(seed=0):
    """High side camera: stands on top, grass below, a few 20 px players."""
    f = np.full((H, W, 3), GRASS, np.uint8)
    f[:int(.2 * H)] = (90, 50, 40)
    rng = np.random.default_rng(seed)
    for _ in range(18):
        x, y = rng.integers(50, W - 50), rng.integers(int(.3 * H), H - 40)
        f[y:y + 22, x:x + 8] = (30, 30, 200)
    return f


def _close_on_grass():
    """Keeper-style close-up: grass behind, a person filling most of the frame."""
    f = np.full((H, W, 3), GRASS, np.uint8)
    f[int(.1 * H):, int(.3 * W):int(.7 * W)] = (40, 200, 230)
    return f


def _crowd():
    f = np.full((H, W, 3), (70, 60, 90), np.uint8)
    f[::7] = (120, 110, 160)
    return f


def _stinger():
    """Trophy diamond: dark red ring, red band, gold centre."""
    f = np.full((H, W, 3), GRASS, np.uint8)
    yy, xx = np.mgrid[:H, :W]
    d = np.abs(xx - W // 2) + np.abs(yy - H // 2)
    f[d < 230] = (20, 20, 120)
    f[d < 190] = (20, 20, 220)
    f[d < 120] = (60, 150, 190)          # gold: hue ~20, S ~170, V ~190
    return f


def test_wide_frame_is_wide_and_close_ups_are_not():
    cfg = ShotConfig()
    assert frame_signals(_wide()).is_wide(cfg)
    assert not frame_signals(_close_on_grass()).is_wide(cfg)
    assert not frame_signals(_crowd()).is_wide(cfg)


def test_stinger_is_found_and_a_wide_frame_is_not_one():
    assert frame_signals(_stinger()).stinger
    assert not frame_signals(_wide()).stinger


def test_static_overlays_do_not_make_a_close_up_wide_or_a_stinger():
    f = _crowd()
    f[10:60, 10:200] = (40, 40, 200)                    # score bug, top-left
    f[10:50, W - 150:W - 20] = (40, 140, 240)           # orange logo, top-right
    s = frame_signals(f)
    assert not s.is_wide(ShotConfig()) and not s.stinger


def test_a_big_person_box_overrules_the_picture():
    s = frame_signals(_wide(), person_h=0.45)
    assert not s.is_wide(ShotConfig())


FPS = 25.0


def _signals(plan):
    """plan: list of (kind, seconds) -> per-frame FrameSignals from synthetic frames."""
    mk = {"wide": _wide(), "close": _close_on_grass(), "stinger": _stinger()}
    sig = {k: frame_signals(v) for k, v in mk.items()}
    out = []
    for kind, s in plan:
        out += [sig[kind]] * int(s * FPS)
    return out


def _seg(plan, cfg=None):
    return live_wide_segments(_signals(plan), FPS, cfg or ShotConfig())


def test_wide_run_is_live_and_a_close_up_is_not():
    live, shot = _seg([("wide", 10), ("close", 5), ("wide", 10)])
    assert live[:250].all() and not live[250:375].any() and live[375:].all()
    assert len(set(shot[:250])) == 1 and shot[0] != shot[-1]


def test_replay_between_two_stingers_is_not_live_even_when_it_looks_wide():
    live, _ = _seg([("wide", 10), ("stinger", 1.5), ("wide", 12), ("stinger", 1.5), ("wide", 10)])
    assert live[:250].all()
    assert not live[250:250 + 37 + 300 + 37].any()
    assert live[-250:].all()


def test_lone_stinger_does_not_swallow_the_match():
    live, _ = _seg([("wide", 10), ("stinger", 1.5), ("wide", 100)])
    assert live[-2000:].all()
    assert not live[250:287].any()


def test_a_non_wide_blip_without_a_cut_is_bridged_but_one_with_a_cut_is_not():
    # a player walking past the camera: same colours, one tall foreground object
    near = _wide()
    near[int(.2 * H):int(.8 * H), 400:470] = (30, 30, 200)
    sig = frame_signals(near)
    assert not sig.is_wide(ShotConfig())
    plan = [frame_signals(_wide())] * 250 + [sig] * 8 + [frame_signals(_wide())] * 250
    live, shot = live_wide_segments(plan, FPS, ShotConfig())
    assert live.all() and len(set(shot)) == 1
    live, shot = _seg([("wide", 10), ("close", 0.3), ("wide", 10)])
    assert not live[250:257].any() and len(set(shot)) == 3


def test_a_short_wide_blip_between_close_ups_is_dropped():
    live, _ = _seg([("close", 10), ("wide", 1.0), ("close", 10)])
    assert not live.any()


def test_a_hard_cut_between_two_wide_frames_starts_a_new_shot():
    a, b = _wide(0), _wide(1)[::-1].copy()
    b[b[..., 1] == GRASS[1]] = (80, 170, 40)           # other camera, other white balance
    sig = [frame_signals(a)] * 100 + [frame_signals(b)] * 100
    live, shot = live_wide_segments(sig, FPS, ShotConfig())
    assert live.all() and shot[99] != shot[100]


def _state_with_low_block(live_wide, shot, seconds=30):
    rows = []
    for f in range(int(seconds * FPS)):
        t = f / FPS
        people = [(1, -50.0, 0.0, Team.HOME)] + [
            (2 + i, -40.0 + (i % 3) * 5, -10.0 + (i // 3) * 7, Team.HOME) for i in range(10)]
        people += [(20, 50.0, 0.0, Team.AWAY)] + [
            (21 + i, -15.0 + (i % 3) * 10, -20.0 + (i // 3) * 13, Team.AWAY) for i in range(10)]
        for tid, x, y, team in people:
            rows.append(dict(frame_idx=f, period=1, timestamp=t, track_id=tid, role=Role.PLAYER.value,
                             team=team.value, x=x, y=y, live_wide=live_wide[f], shot=shot[f]))
        rows.append(dict(frame_idx=f, period=1, timestamp=t, track_id=-1, role=Role.BALL.value,
                         team=Team.UNKNOWN.value, x=-15.0, y=0.0, live_wide=live_wide[f], shot=shot[f]))
    meta = MatchMeta(match_id="S", fps=FPS, home=TeamInfo("H", "H"), away=TeamInfo("A", "A"))
    poss = pd.DataFrame([dict(frame_idx=f, period=1, timestamp=f / FPS, ballStatus="AWAY")
                         for f in range(int(seconds * FPS))])
    return MatchState(meta, pd.DataFrame(rows)), poss


def test_default_state_is_all_live_so_old_callers_are_unchanged():
    st, poss = _state_with_low_block([True] * 750, [0] * 750)
    st = MatchState(st.meta, st.tracks.drop(columns=["live_wide", "shot"]))
    assert st.tracks["live_wide"].all() and (st.tracks["shot"] == 0).all()
    assert len(detect_sequences(st, poss)) == 1


def test_a_sequence_does_not_span_a_cut_and_frames_outside_the_mask_are_ignored():
    f = np.arange(750)
    live = ~((f >= 300) & (f < 400))                    # a 4 s replay in the middle
    shot = np.where(f < 300, 0, np.where(f < 400, 1, 2))
    st, poss = _state_with_low_block(live, shot)
    seq = detect_sequences(st, poss)
    assert len(seq) == 2
    assert seq["end_frame"].iloc[0] < 300 and seq["start_frame"].iloc[1] >= 400


def test_a_cut_alone_splits_a_sequence_even_when_both_sides_are_live():
    f = np.arange(750)
    st, poss = _state_with_low_block(np.ones(750, bool), np.where(f < 375, 0, 1))
    assert len(detect_sequences(st, poss)) == 2

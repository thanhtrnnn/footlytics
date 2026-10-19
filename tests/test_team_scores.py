import numpy as np

from footlytics.perception.teams import N_HUE_BINS, N_SAT_BINS, N_VAL_BINS, team_scores_by_track

SIZE = N_HUE_BINS * N_SAT_BINS * N_VAL_BINS


def kit(bin_, n, rng, spill=0.3):
    """n noisy histograms peaked at `bin_`, with some mass on its neighbour."""
    d = np.zeros((n, SIZE), np.float32)
    d[:, bin_] = 1.0 - spill
    d[:, (bin_ + 1) % SIZE] = spill
    d += rng.random((n, SIZE)).astype(np.float32) * 0.05
    return d / d.sum(1, keepdims=True)


def clip(rng):
    """10 home and 10 away players seen 30 frames each, two keepers in odd kits seen for the
    whole clip (300 frames each) and a referee seen 200 frames: the odd kits carry as many
    observations as a team, which is what pulls a 2-means on detections off the teams."""
    tracks, descs = [], []
    for tid in range(10):
        tracks += [tid] * 30; descs.append(kit(10, 30, rng))
    for tid in range(10, 20):
        tracks += [tid] * 30; descs.append(kit(30, 30, rng))
    for tid, b, n in ((90, 60, 300), (91, 80, 300), (92, 100, 200)):
        tracks += [tid] * n; descs.append(kit(b, n, rng))
    return np.array(tracks), np.concatenate(descs)


def test_players_split_by_team_whatever_the_odd_kits_weigh():
    T, D = clip(np.random.default_rng(0))
    s = team_scores_by_track(T, D)
    home, away = [s[t] for t in range(10)], [s[t] for t in range(10, 20)]
    assert (all(v < 0 for v in home) and all(v > 0 for v in away)) or \
           (all(v > 0 for v in home) and all(v < 0 for v in away))
    assert set(s) == set(range(20)) | {90, 91, 92}
    assert all(-1.0 <= v <= 1.0 for v in s.values())


def test_a_track_with_no_usable_descriptor_scores_zero():
    T, D = clip(np.random.default_rng(1))
    T = np.append(T, [55, 55]); D = np.vstack([D, np.zeros((2, SIZE), np.float32)])
    assert team_scores_by_track(T, D)[55] == 0.0


def test_too_few_tracks_for_four_groups_still_splits_two_kits():
    rng = np.random.default_rng(2)
    T = np.array([0] * 20 + [1] * 20 + [2] * 20)
    D = np.concatenate([kit(10, 20, rng), kit(10, 20, rng), kit(30, 20, rng)])
    s = team_scores_by_track(T, D)
    assert np.sign(s[0]) == np.sign(s[1]) != np.sign(s[2])

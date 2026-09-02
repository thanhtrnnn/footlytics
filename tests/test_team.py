import numpy as np

from footlytics.team import TeamClassifier, jersey_feature


def _crop(bgr, h=40, w=20):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    return img


def test_jersey_feature_is_fixed_length_and_normalised():
    f = jersey_feature(_crop((0, 0, 255)))
    assert f.ndim == 1
    assert abs(f.sum() - 1.0) < 1e-6


def test_two_colours_form_two_teams():
    red = [_crop((0, 0, 255)) for _ in range(6)]
    blue = [_crop((255, 0, 0)) for _ in range(6)]
    clf = TeamClassifier().fit(red + blue)
    labels = clf.predict(red + blue)
    assert set(labels) == {0, 1}
    assert len(set(labels[:6])) == 1
    assert len(set(labels[6:])) == 1
    assert labels[0] != labels[6]


def test_predict_is_deterministic_for_same_track():
    red = [_crop((0, 0, 255)) for _ in range(5)]
    blue = [_crop((255, 0, 0)) for _ in range(5)]
    clf = TeamClassifier().fit(red + blue)
    a = clf.predict([red[0]])
    b = clf.predict([red[0]])
    assert a == b


def _player_on_grass(kit_bgr, h=36, w=14):
    """Tiny 720p-scale crop: grass background with a kit-coloured torso patch."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (60, 150, 60)  # grass green (BGR)
    img[int(0.2 * h) : int(0.5 * h), int(0.3 * w) : int(0.7 * w)] = kit_bgr
    return img


def test_white_vs_dark_blue_kits_on_grass_form_two_teams():
    white = [_player_on_grass((235, 235, 235)) for _ in range(8)]
    navy = [_player_on_grass((90, 30, 10)) for _ in range(8)]
    clf = TeamClassifier().fit(white + navy)
    labels = clf.predict(white + navy)
    assert len(set(labels[:8])) == 1
    assert len(set(labels[8:])) == 1
    assert labels[0] != labels[8]


def test_grass_pixels_do_not_dominate_feature():
    a = jersey_feature(_player_on_grass((235, 235, 235)))
    b = jersey_feature(_player_on_grass((90, 30, 10)))
    # features of different kits must differ clearly even though 80% of pixels are grass
    assert np.abs(a - b).sum() > 0.5


def test_referee_colour_becomes_ref_not_a_team():
    white = [_player_on_grass((235, 235, 235)) for _ in range(8)]
    navy = [_player_on_grass((90, 30, 10)) for _ in range(8)]
    red = [_player_on_grass((20, 20, 220)) for _ in range(2)]
    clf = TeamClassifier().fit(white + navy + red)
    labels = clf.predict(white + navy + red)
    assert set(labels[:8]) != set(labels[8:16])
    assert set(labels[:8]) | set(labels[8:16]) == {0, 1}
    assert set(labels[16:]) == {"ref"}

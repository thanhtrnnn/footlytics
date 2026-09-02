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

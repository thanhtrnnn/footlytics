"""GSR adapter: ground truth -> Match State -> predictions is lossless for what GS-HOTA scores."""
import numpy as np
import pytest

from footlytics.bench.adapters.gsr import (gt_to_state, image_id, state_to_predictions,
                                           team_sides)
from footlytics.state.schema import Role, Team
from tests.bench_fixtures import PEOPLE, make_labels


def test_image_id_is_split_clip_frame_one_based():
    assert image_id("valid", "SNGS-033", 0) == "2033000001"
    assert image_id("test", "SNGS-117", 749) == "3117000750"


def test_gt_to_state_maps_roles_teams_frames():
    s = gt_to_state(make_labels(n_frames=2))
    t = s.tracks
    assert s.meta.match_id == "SNGS-990" and s.meta.source_type == "broadcast"
    assert sorted(t["frame_idx"].unique()) == [0, 1]
    assert len(t) == 2 * len(PEOPLE)                       # the null-pitch box is dropped
    gk = t[t["track_id"] == 1].iloc[0]
    assert gk["role"] == Role.GOALKEEPER.value and gk["team"] == Team.HOME.value
    assert t[t["track_id"] == 4].iloc[0]["team"] == Team.AWAY.value
    assert t[t["track_id"] == 7].iloc[0]["team"] == Team.OFFICIAL.value
    assert t[t["track_id"] == 9].iloc[0]["role"] == Role.OTHER.value
    assert t[t["track_id"] == 2].iloc[0]["jersey"] == 4.0
    assert np.isnan(t[t["track_id"] == 6].iloc[0]["jersey"])


def test_old_label_version_is_refused():
    with pytest.raises(ValueError, match="1.3"):
        gt_to_state(make_labels(version="1.2"))


def test_team_sides_from_goalkeepers_then_mean_x():
    t = gt_to_state(make_labels()).tracks
    assert team_sides(t) == {"home": "left", "away": "right"}
    swapped = t.copy()
    swapped["x"] = -swapped["x"]                           # camera from the other side
    assert team_sides(swapped) == {"home": "right", "away": "left"}
    no_gk = t[t["role"] != Role.GOALKEEPER.value]
    assert team_sides(no_gk) == {"home": "left", "away": "right"}
    one_team = t[t["team"] == Team.AWAY.value]
    assert team_sides(one_team) == {"away": "right"}


def test_roundtrip_predictions_match_ground_truth():
    labels = make_labels(n_frames=2)
    pred = state_to_predictions(gt_to_state(labels), "valid", use_jersey=True)["predictions"]
    gt = {(a["image_id"], a["track_id"]): a for a in labels["annotations"]
          if a["supercategory"] == "object" and a["bbox_pitch"] is not None}
    assert len(pred) == len(gt)
    for p in pred:
        g = gt[(p["image_id"], p["track_id"])]
        assert p["supercategory"] == "object" and p["category_id"] == g["category_id"]
        assert p["attributes"]["role"] == g["attributes"]["role"]
        assert p["attributes"]["team"] == g["attributes"]["team"]
        assert p["attributes"]["jersey"] == (g["attributes"]["jersey"]
                                             if g["attributes"]["role"] == "player" else None)
        for k in ("x_bottom_middle", "y_bottom_middle"):
            assert abs(p["bbox_pitch"][k] - g["bbox_pitch"][k]) < 1e-4
        assert set(p["bbox_pitch"]) == set(g["bbox_pitch"])


def test_jersey_off_writes_null_jerseys():
    pred = state_to_predictions(gt_to_state(make_labels()), "valid")["predictions"]
    assert all(p["attributes"]["jersey"] is None for p in pred)

import numpy as np
import pandas as pd
import pytest

from footlytics.perception.track import PitchTracker, Track, TrackerConfig, _KF, resolve_roles
from footlytics.state.schema import Role


def test_track_confirmed_uses_tracker_min_hits():
    trk = PitchTracker(TrackerConfig(fps=25.0, min_hits=5))
    for _ in range(4):
        trk.update(np.array([[0.0, 0.0]]), [Role.PLAYER.value])
    assert trk.tracks and not trk.tracks[0].confirmed
    trk.update(np.array([[0.0, 0.0]]), [Role.PLAYER.value])
    assert trk.tracks[0].confirmed


def test_resolve_roles_votes_long_tracks_and_keeps_short_ones():
    rows = [{"track_id": 1, "role": "player"}] * 12 + [{"track_id": 1, "role": "goalkeeper"}] * 3
    rows += [{"track_id": 2, "role": "player"}, {"track_id": 2, "role": "referee"}]
    out = resolve_roles(pd.DataFrame(rows), min_frames=10)
    assert set(out.loc[out.track_id == 1, "role"]) == {"player"}
    assert set(out.loc[out.track_id == 2, "role"]) == {"player", "referee"}


def test_ppda_refuses_a_press_zone_it_cannot_apply():
    from footlytics.analytics.tactics import ppda

    with pytest.raises(NotImplementedError):
        ppda(None, pd.DataFrame({"player_id": [], "period": [], "label": []}), {}, press_zone=0.4)

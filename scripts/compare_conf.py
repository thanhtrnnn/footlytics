"""Does a low confidence gate actually help tracking, or just look noisy?

The gate was set to 0.10 on the reasoning that recall matters more than
precision for tracking -- a missed player breaks a track, a false positive gets
filtered by min_hits. That was an argument, not a measurement. This measures it
end to end: same frames, same everything else, three gates.
"""
import sys, time; sys.path.insert(0, ".")
import pandas as pd
from footlytics.perception.detect import Detector, DetectorConfig
from footlytics.pipeline.radar import run, PipelineConfig
from footlytics.perception.track import TrackerConfig
from footlytics.state.schema import MatchMeta, TeamInfo
from footlytics.data.soccertrack import (SoccerTrackCalibration, score_against_mot,
                                         PITCH_L, PITCH_W)

D = "data/raw/soccertrack/117092"
cal = SoccerTrackCalibration.load(D, 117092)
rows = []
for conf, margin in [(0.10, 6.0), (0.25, 2.0), (0.35, 2.0)]:
    det = Detector("weights/yolo_v8x6_finetuned.pt",
                   DetectorConfig(tile=1280, overlap=0.25, conf=conf, ball_conf=conf,
                                  device="mps", half=False))
    meta = MatchMeta(match_id=f"conf{conf}", fps=25.0, pitch_length=PITCH_L,
                     pitch_width=PITCH_W, source_type="fixed_panoramic",
                     home=TeamInfo("A", "A"), away=TeamInfo("B", "B"))
    t0 = time.time()
    state, rep = run("data/raw/soccertrack/clips/117092.mp4", cal, det, meta,
                     cfg=PipelineConfig(stride=1, max_frames=250, use_appearance=True,
                                        stitch=True, roster_size=11,
                                        pitch_margin_m=margin, verbose=False),
                     tracker_cfg=TrackerConfig(fps=25.0))
    s = score_against_mot(state, f"{D}/117092.txt", cal)
    rows.append({"conf": conf, "margin_m": margin,
                 "tracklets": rep["tracklets_after_stitch"],
                 "recall": s["recall"], "precision": s["precision"],
                 "err_m": s["err_median_m"], "id_switches": s["id_switches"],
                 "players_per_frame": rep["median_players_per_frame"],
                 "sec": round(time.time() - t0)})
    state.save(f"data/out/conf_{conf}")
    print(f"  conf {conf} margin {margin} -> {rows[-1]}", flush=True)

r = pd.DataFrame(rows)
print("\n" + "=" * 78)
print(r.to_string(index=False))
print("\nFewer tracklets and fewer switches = better identity.")
r.to_csv("data/out/conf_comparison.csv", index=False)

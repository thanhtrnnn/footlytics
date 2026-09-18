"""End-to-end: our full pipeline on real video, scored against ground truth."""
import sys, time; sys.path.insert(0, ".")
from footlytics.perception.detect import Detector, DetectorConfig
from footlytics.pipeline.radar import run, PipelineConfig
from footlytics.perception.track import TrackerConfig
from footlytics.state.schema import MatchMeta, TeamInfo
from footlytics.data.soccertrack import (SoccerTrackCalibration, score_against_mot,
                                         PITCH_L, PITCH_W)

D = "data/raw/soccertrack/117092"
cal = SoccerTrackCalibration.load(D, 117092)
det = Detector("weights/yolo_v8x6_finetuned.pt",
               DetectorConfig(tile=1280, overlap=0.25, conf=0.10, ball_conf=0.10,
                              device="mps", half=False))
meta = MatchMeta(match_id="ST2-117092-run", fps=25.0,
                 pitch_length=PITCH_L, pitch_width=PITCH_W,
                 source_type="fixed_panoramic",
                 home=TeamInfo("Tsukuba B", "B"), away=TeamInfo("Tsukuba C1", "C1"))

t0 = time.time()
state, report = run("data/raw/soccertrack/clips/117092.mp4", cal, det, meta,
                    cfg=PipelineConfig(stride=1, max_frames=250, use_appearance=True,
                                       stitch=True, roster_size=11),
                    tracker_cfg=TrackerConfig(fps=25.0))
print(f"\ntotal {time.time()-t0:.0f}s")
state.save("data/out/ST2-117092-run")

print("\n" + "=" * 60)
print("SCORED AGAINST GROUND TRUTH")
s = score_against_mot(state, f"{D}/117092.txt", cal)
for k, v in s.items():
    print(f"  {k:18s} {v}")

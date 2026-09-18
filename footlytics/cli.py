"""Command line: `footlytics run`, `footlytics calib-assist`, `footlytics drift-check`."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer

app = typer.Typer(help="Footlytics: video -> Match State (pitch metres) -> analytics")


@app.callback()
def _root() -> None:
    """Footlytics CLI."""


def _read_frame(path: Path, idx: int):
    import cv2

    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, f = cap.read(); cap.release()
    if not ok:
        raise typer.BadParameter(f"cannot read frame {idx} of {path}")
    return f


@app.command()
def run(
    video: Path = typer.Argument(..., exists=True),
    calib: Path = typer.Option(..., help="Calibration JSON (Calibration.save) clicked on the anchor frame"),
    out: Path = typer.Option(Path("data/out/run"), help="output directory"),
    anchor_frame: int = typer.Option(0, help="frame the calibration was clicked on"),
    camera: bool = typer.Option(True, help="track camera motion relative to the anchor (off for a fixed rig)"),
    weights: Path = typer.Option(Path("weights/yolo_v8x6_finetuned.pt")),
    ball_model: Optional[Path] = typer.Option(None, help="dedicated ball detector weights (run at 1920 px)"),
    tile: int = typer.Option(0, help="tile size for panoramas (0 = whole frame)"),
    imgsz: int = typer.Option(1280),
    conf: float = typer.Option(0.25),
    device: str = typer.Option("mps", help="mps | cuda | cpu"),
    stride: int = typer.Option(1),
    max_frames: Optional[int] = typer.Option(None),
    source_type: str = typer.Option("tactical_cam", help="broadcast | fixed_panoramic | tactical_cam"),
    home: str = typer.Option("HOME"),
    away: str = typer.Option("AWAY"),
    radar: bool = typer.Option(True, help="render radar.mp4"),
) -> None:
    """Run perception on a clip and write tracks.parquet, meta.json, report.json, quality.json, radar.mp4."""
    from .geometry.camera_motion import AnchoredCamera
    from .geometry.homography import Calibration
    from .geometry.pitch import DEFAULT_PITCH
    from .perception.detect import Detector, DetectorConfig
    from .pipeline.quality import compute_quality, write_quality
    from .pipeline.radar import PipelineConfig, run as run_pipeline
    from .state.schema import MatchMeta, TeamInfo

    lm = DEFAULT_PITCH.landmarks()
    cal = Calibration.load(calib, lm)
    ok, verdict = cal.verdict(lm)
    typer.echo(f"[calibration] {verdict}")
    cam = AnchoredCamera(_read_frame(video, anchor_frame), start_at_anchor=(anchor_frame == 0)) if camera else None
    det = Detector(weights, DetectorConfig(tile=tile, imgsz=imgsz, conf=conf, device=device, half=(device == "cuda"),
                                           ball_weights=str(ball_model) if ball_model else None))
    meta = MatchMeta(match_id=video.stem, fps=25.0, source_type=source_type,
                     home=TeamInfo(home, home[:3].upper()), away=TeamInfo(away, away[:3].upper()))
    t0 = time.time()
    state, report = run_pipeline(video, cal, det, meta, PipelineConfig(stride=stride, max_frames=max_frames), camera=cam)
    out.mkdir(parents=True, exist_ok=True)
    state.save(out)
    clean = {k: v for k, v in report.items() if not k.startswith("_")}
    clean["wall_seconds"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(clean, indent=2, default=str))
    q = compute_quality(state, {**clean, "seconds": clean["wall_seconds"]}, stride=stride)
    write_quality(q, out)
    if radar:
        from .viz.radar import render_video

        render_video(state, str(out / "radar.mp4"))
    typer.echo(json.dumps({k: q[k] for k in ("frames", "mean_players_per_frame", "ball_coverage",
                                              "id_switch_rate_per_player_per_minute", "seconds_per_match_minute",
                                              "calib_success_rate") if k in q}, indent=2))


@app.command("calib-assist")
def calib_assist(
    video: Path = typer.Argument(..., exists=True),
    frame: int = typer.Option(0, help="frame to annotate (becomes the anchor frame)"),
    out: Path = typer.Option(Path("data/out/assist"), help="output stem: writes .jpg and .json"),
    model: str = typer.Option("weights/yolo_v8x6_finetuned.pt",
                              help="detector used to mask people out of the line search; \"\" to skip"),
    device: str = typer.Option("mps", help="mps | cuda | cpu"),
) -> None:
    """Detect pitch line intersections on one frame and write numbered candidates plus a calibration template."""
    from .geometry.calib_assist import write_assist

    img_bgr = _read_frame(video, frame)
    boxes = None
    if model:
        # Players stand on the lines and their edges read as line segments, so a
        # frame full of people yields candidate "intersections" that are really
        # shirt corners. Mask them out first.
        from .perception.detect import Detector, DetectorConfig
        from .state.schema import Role

        det = Detector(Path(model), DetectorConfig(tile=0, device=device, half=(device == "cuda")))
        dets = det.detect(img_bgr[:, :, ::-1])
        people = [d for d in dets if det.role_of(d[5]) != Role.BALL.value]
        if people:
            p = np.asarray(people, dtype=float)
            boxes = np.column_stack([p[:, 0], p[:, 1], p[:, 0] + p[:, 2], p[:, 1] + p[:, 3]])

    img, js = write_assist(img_bgr, out, person_boxes=boxes, anchor_frame=frame)
    masked = "no mask" if boxes is None else f"{len(boxes)} people masked"
    typer.echo(f"wrote {img} and {js} ({masked}). "
               "Label 4+ candidates with landmark names, then `footlytics calibrate`.")


@app.command()
def calibrate(
    template: Path = typer.Argument(..., exists=True, help="assist JSON with template.image_points filled in"),
    out: Path = typer.Option(..., help="calibration JSON to write"),
    tps: bool = typer.Option(False, help="thin-plate spline refinement for stitched panoramas"),
) -> None:
    """Build a Calibration from a filled assist template and print its verdict."""
    from .geometry.homography import Calibration
    from .geometry.pitch import DEFAULT_PITCH

    d = json.loads(template.read_text())["template"]
    pts = {k: tuple(v) for k, v in d["image_points"].items()}
    lm = DEFAULT_PITCH.landmarks()
    cal = Calibration.from_correspondences(pts, lm, tuple(d["image_size"]), use_tps=tps,
                                           notes=f"anchor_frame={d.get('anchor_frame', 0)}")
    cal.save(out)
    typer.echo(f"[calibration] {cal.verdict(lm)[1]} -> {out}")


@app.command("drift-check")
def drift_check(
    video: Path = typer.Argument(..., exists=True),
    calib: Path = typer.Option(...),
    out: Path = typer.Option(Path("data/out/drift")),
    anchor_frame: int = typer.Option(0),
    frames: list[int] = typer.Option([0], help="frames to overlay the pitch model on"),
) -> None:
    """Overlay the projected pitch model on chosen frames and report invalid frame runs."""
    import cv2
    import numpy as np

    from .geometry.camera_motion import AnchoredCamera, MovingCalibration
    from .geometry.homography import Calibration
    from .geometry.pitch import DEFAULT_PITCH, CENTRE_CIRCLE_R

    lm = DEFAULT_PITCH.landmarks()
    cal = Calibration.load(calib, lm)
    mc = MovingCalibration(cal)
    cam = AnchoredCamera(_read_frame(video, anchor_frame), start_at_anchor=(anchor_frame == 0))
    L, W = DEFAULT_PITCH.half_l, DEFAULT_PITCH.half_w
    segs = [((-L, -W), (L, -W)), ((L, -W), (L, W)), ((L, W), (-L, W)), ((-L, W), (-L, -W)), ((0, -W), (0, W))]
    for s, sx in (("L", -1), ("R", 1)):
        for key in ("pen_area", "goal_area"):
            a, b = lm[f"{key}_{s}T_goalline"], lm[f"{key}_{s}T_front"]
            c, d = lm[f"{key}_{s}B_front"], lm[f"{key}_{s}B_goalline"]
            segs += [(a, b), (b, c), (c, d)]
    ang = np.linspace(0, 2 * np.pi, 90)
    circle = np.c_[CENTRE_CIRCLE_R * np.cos(ang), CENTRE_CIRCLE_R * np.sin(ang)]
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(frames)
    invalid: list[int] = []
    cap = cv2.VideoCapture(str(video)); idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        H = cam.update(fr)
        if H is None:
            invalid.append(idx)
        elif idx in wanted:
            vis = fr.copy()
            for a, b in segs:
                p = mc.pitch_to_image(np.array([a, b], float), H)
                if np.all(np.abs(p) < 6000):
                    cv2.line(vis, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (255, 0, 255), 2)
            cv2.polylines(vis, [mc.pitch_to_image(circle, H).astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), 2)
            cv2.imwrite(str(out / f"drift_{idx}.jpg"), vis)
        idx += 1
    cap.release()
    runs, s0, p0 = [], None, None
    for i in invalid:
        if s0 is None or i != p0 + 1:
            if s0 is not None:
                runs.append((s0, p0))
            s0 = i
        p0 = i
    if s0 is not None:
        runs.append((s0, p0))
    typer.echo(f"frames {idx}, invalid {len(invalid)}, runs {runs}, reanchors {cam.n_reanchors}")


if __name__ == "__main__":
    app()

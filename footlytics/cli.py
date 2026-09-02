"""Command line interface: `footlytics run clip.mp4 --out data/out/run1`."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from footlytics.pipeline import run_pipeline
from footlytics.track import DEFAULT_TRACKER

app = typer.Typer(help="FOOTLYTICS Game-State Engine prototype")


@app.callback()
def _root() -> None:
    """FOOTLYTICS CLI."""


@app.command()
def run(
    clip: Path = typer.Argument(..., exists=True, help="input video"),
    out: Path = typer.Option(Path("data/out/run"), help="output directory"),
    max_frames: int | None = typer.Option(None, help="limit frames for quick runs"),
    model: str = typer.Option("yolo11n.pt", help="ultralytics model name or path"),
    device: str | None = typer.Option(None, help="mps | cuda | cpu (auto)"),
    tracker: str = typer.Option(DEFAULT_TRACKER, help="tracker yaml (ultralytics bytetrack.yaml, botsort.yaml, or custom path)"),
    calib: Path | None = typer.Option(None, help="landmark JSON for static calibration (see footlytics/calibration.py)"),
    ball_model: Path | None = typer.Option(None, help="dedicated ball detector weights (run at 1920 px)"),
    imgsz: int = typer.Option(1280, help="detector input size for people tracking"),
) -> None:
    res = run_pipeline(clip, out, max_frames=max_frames, model_name=model, device=device, tracker=tracker,
                       calib=calib, ball_weights=ball_model, imgsz=imgsz)
    typer.echo(json.dumps(res["quality"], indent=2))


if __name__ == "__main__":
    app()


@app.command("calib-assist")
def calib_assist(
    clip: Path = typer.Argument(..., exists=True, help="input video"),
    frame: int = typer.Option(0, help="frame index to annotate"),
    out: Path = typer.Option(Path("data/out/calib_assist"), help="output prefix (.jpg and .json are written)"),
    model: str = typer.Option("yolo11n.pt", help="detector used to mask people out of the line search"),
) -> None:
    """Detect pitch line intersections on one frame to help build a calibration JSON."""
    from ultralytics import YOLO

    from footlytics.calib_assist import write_assist
    from footlytics.pipeline import _read_frame
    from footlytics.track import default_device

    img = _read_frame(clip, frame)
    r = YOLO(model).predict(img, imgsz=1280, conf=0.2, classes=[0], device=default_device(), verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else None
    img_path, json_path = write_assist(img, out, person_boxes=boxes, frame_index=frame)
    typer.echo(f"wrote {img_path} and {json_path}")

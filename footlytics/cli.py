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

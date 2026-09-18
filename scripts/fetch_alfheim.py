#!/usr/bin/env python3
"""Fetch a slice of the Alfheim (Tromsø IL) dataset for real-footage testing.

Why this dataset: two *stationary* camera arrays covering the whole pitch,
available as individual cameras or a stitched panorama -- the same optical
situation as a Bepro rig -- plus ZXY body-sensor positions at 20 Hz as ground
truth. That combination lets us measure our error in metres on real football
instead of guessing from synthetic tests.

    https://datasets.simula.no/alfheim/   (Pettersen et al., ACM MMSys 2014)

Two limitations to keep in mind
------------------------------
* Only Tromsø IL players carried sensors, so ground truth covers ~11 players,
  not 22. Fine for validating calibration and tracking; useless for validating
  team assignment.
* It is 2013 footage. Resolution and compression are well below a modern Bepro
  camera, so treat detection numbers here as a pessimistic floor.

Verified specs (read from the H.264 bitstream, not from the paper)
------------------------------------------------------------------
* single camera: 1280x960, 30 fps, baseline profile, ~1.3 MB per 3 s chunk
* three single-camera views (0, 1, 2) plus a stitched `panorama` view
* 943 chunks per half = ~47 min; panorama chunks are ~12 MB per 3 s (~10x)
* no IDR keyframes in the stream, so seeking is unreliable -- decode forward
  from the start of the concatenated file rather than seeking into it

The alignment trick: every video chunk is named with the wall-clock time it
starts, and every ZXY row carries a wall-clock timestamp. So frame k of a chunk
maps to a real instant, and we can look up exactly where each player was.

Usage
-----
    python scripts/fetch_alfheim.py --minutes 2 --view panorama
    python scripts/fetch_alfheim.py --minutes 2 --view 0    # single camera, ~10x smaller
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

BASE = "https://datasets.simula.no/downloads/alfheim"

MATCHES = {
    "2013-11-03": ("tromso_stromsgodset", ["First Half", "Second Half"]),
    "2013-11-07": ("tromso_anji", ["First Half", "Second Half"]),
    "2013-11-28": ("tromso_tottenham", ["First Half", "Second Half"]),
}

# Alfheim pitch, per the dataset paper. ZXY uses a corner origin.
PITCH_L, PITCH_W = 105.0, 68.0

CHUNK_RE = re.compile(r"(\d+)_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\.h264")


def _get(url: str, timeout: int = 120, retries: int = 4) -> bytes:
    """Fetch a URL from the Simula host, which needs two accommodations.

    Its certificate chain is incomplete, so verification is off -- acceptable
    for a public research dataset at a URL hard-coded here, and the alternative
    is that nothing downloads at all.

    And it must go through `requests`, not urllib: urllib reliably gets a
    BrokenPipeError on this host regardless of User-Agent or read strategy,
    while requests fetches the identical bytes without complaint.
    """
    import time

    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, verify=False, timeout=timeout)
            r.raise_for_status()
            if r.content:
                return r.content
            last = RuntimeError("empty response")
        except Exception as e:                       # noqa: BLE001 - retry anything
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} attempts: {url}\n  last error: {last}")


def list_chunks(date: str, half: str, view: str) -> list[tuple[int, datetime, str]]:
    url = f"{BASE}/{date}/{urllib.parse.quote(half)}/{view}/"
    html = _get(url, timeout=60).decode("utf-8", "replace")
    out = []
    for href in re.findall(r'href="\./([^"]+\.h264)"', html):
        name = urllib.parse.unquote(href)
        m = CHUNK_RE.match(name)
        if m:
            ts = datetime.strptime(m.group(2)[:26], "%Y-%m-%d %H:%M:%S.%f")
            out.append((int(m.group(1)), ts, url + href))
    return sorted(out, key=lambda c: c[0])


def download_video(date: str, half: str, view: str, minutes: float, out_dir: Path) -> dict:
    chunks = list_chunks(date, half, view)
    if not chunks:
        raise RuntimeError(f"no chunks listed for {date} / {half} / {view}")
    # Chunks are ~3 s each; take enough to cover the requested duration.
    n = max(int(minutes * 60 / 3), 1)
    chunks = chunks[:n]
    raw = out_dir / f"alfheim_{date}_{view}.h264"
    print(f"[fetch] {len(chunks)} chunks from {view} "
          f"(starts {chunks[0][1].time()}, ~{len(chunks) * 3 / 60:.1f} min)")

    with open(raw, "wb") as f:
        for i, (_, _, url) in enumerate(chunks):
            f.write(_get(url))
            print(f"  {i + 1}/{len(chunks)}  {raw.stat().st_size / 1e6:7.1f} MB", end="\r")
    print()
    return {"raw": raw, "start": chunks[0][1], "n_chunks": len(chunks)}


def remux(raw: Path, fps: float = 30.0) -> Path:
    """Wrap the raw Annex-B stream in an mp4 so OpenCV can seek it."""
    mp4 = raw.with_suffix(".mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-r", str(fps),
           "-f", "h264", "-i", str(raw), "-c", "copy", str(mp4)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[fetch] ffmpeg not found -- install it, or point OpenCV at the .h264 directly")
        return raw
    print(f"[fetch] wrote {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)")
    return mp4


def load_zxy(date: str, half: str, out_dir: Path, start=None, seconds=None):
    """Download and normalise the sensor ground truth.

    Converts ZXY's corner origin to our centre-spot convention so it can be
    compared with a MatchState directly.
    """
    import pandas as pd

    slug = MATCHES[date][0]
    suffix = "first" if "First" in half else "second"
    name = (f"{date}_{slug}_{suffix}.csv" if date != "2013-11-28"
            else f"{date}_{slug}.csv")
    url = f"{BASE}/{date}/zxy/{name}"
    dst = out_dir / name
    if not dst.exists():
        print(f"[fetch] {name} (~87 MB)")
        dst.write_bytes(_get(url, timeout=600))

    cols = ["ts", "tag_id", "x", "y", "heading", "direction",
            "energy", "speed", "total_distance"]
    df = pd.read_csv(dst, names=cols, on_bad_lines="skip")
    df["ts"] = pd.to_datetime(df["ts"], format="mixed")
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x", "y", "tag_id"])

    if start is not None and seconds is not None:
        df = df[(df["ts"] >= start) & (df["ts"] <= start + timedelta(seconds=seconds))]

    # corner origin -> centre origin, matching footlytics' convention
    df["x_pitch"] = df["x"] - PITCH_L / 2
    df["y_pitch"] = df["y"] - PITCH_W / 2
    return df.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2013-11-03", choices=sorted(MATCHES))
    ap.add_argument("--half", default="First Half")
    ap.add_argument("--view", default="panorama",
                    help="'panorama' (stitched, ~12 MB per 3 s) or '0'/'1'/'2' (single camera)")
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=30.0,
                help="30 fps, confirmed from the H.264 SPS VUI and by frame count")
    ap.add_argument("--out", default="data/raw/alfheim")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--skip-zxy", action="store_true")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    info = {}
    if not a.skip_video:
        info = download_video(a.date, a.half, a.view, a.minutes, out)
        remux(info["raw"], a.fps)

    if not a.skip_zxy:
        gt = load_zxy(a.date, a.half, out,
                      start=info.get("start"),
                      seconds=a.minutes * 60 if info else None)
        p = out / f"zxy_{a.date}_{a.view}.parquet"
        gt.to_parquet(p, index=False)
        print(f"[fetch] ground truth -> {p}  ({len(gt):,} rows, "
              f"{gt.tag_id.nunique()} players)")
        print(f"        x {gt.x_pitch.min():.1f}..{gt.x_pitch.max():.1f} m, "
              f"y {gt.y_pitch.min():.1f}..{gt.y_pitch.max():.1f} m "
              f"(centre-origin, ready to compare with MatchState)")
        if info:
            print(f"        video starts {info['start']} -- use this to align frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())

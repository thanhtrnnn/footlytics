"""V.League ball detector adaptation: candidates, pseudo-labels, test-set evaluation.

Subcommands (`detect` needs the GPU host, the rest are CPU):

  detect   run a ball detector over the clip and the *dev* V.League windows, save the candidates
  pseudo   candidates -> ball tracks -> pseudo-label file (+ review sheets)
  eval     score a candidates file against the hand-labelled test set

Frozen-split windows are never read: `dev_windows` raises on anything the manifest does not
mark "dev", and there is no flag to turn that off.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "bench/vleague/manifest.json"
WINDOW = 12
HIT_PX = 20.0


def dev_windows(vl_root: Path, matches: list[str] | None = None) -> dict[str, list[Path]]:
    """'match/idx' -> the 25 jpgs of every *dev* window (offset order). Frozen is refused."""
    man = json.loads(MANIFEST.read_text())["matches"]
    out = {}
    for m, ent in sorted(man.items()):
        if matches is not None and m not in matches:
            continue
        for idx, meta in sorted(ent["frames"].items()):
            if meta["split"] != "dev":
                continue
            d = vl_root / m / idx
            out[f"{m}/{idx}"] = [d / f"{off:+03d}.jpg" for off in range(-WINDOW, WINDOW + 1)]
    for key in out:
        assert man[key.split("/")[0]]["frames"][key.split("/")[1]]["split"] == "dev", key
    return out


def pitch_mask_small(bgr: np.ndarray) -> np.ndarray:
    """Largest grass region at 1/4 resolution, dilated ~40 px (a ball just off the grass counts)."""
    small = cv2.resize(bgr, (bgr.shape[1] // 4, bgr.shape[0] // 4), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(grass)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return cv2.dilate(mask, np.ones((11, 11), np.uint8))


def candidates(model, bgr: np.ndarray, dev, thr: float, max_n: int) -> np.ndarray:
    from footlytics.perception.ball_detr import infer

    c = np.array(infer(model, bgr[..., ::-1], dev, thr, max_n), np.float32).reshape(-1, 5)
    m = pitch_mask_small(bgr)
    on = np.array([float(m[min(int((y + h / 2) / 4), m.shape[0] - 1), min(int((x + w / 2) / 4), m.shape[1] - 1)] > 0)
                   for x, y, w, h, _ in c], np.float32)
    return np.concatenate([c, on[:, None]], 1)


def cmd_detect(a):
    import torch
    from transformers import RTDetrV2ForObjectDetection

    dev = torch.device("cuda")
    model = RTDetrV2ForObjectDetection.from_pretrained(str(Path(a.weights).expanduser())).to(dev).eval()
    out: dict = {"weights": str(a.weights)}
    if a.video:
        cap = cv2.VideoCapture(str(Path(a.video).expanduser()))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lo, hi = int(a.start * 25), min(n, int(a.end * 25)) if a.end else n
        dets = {}
        if a.frames:                       # only these clip frames (e.g. the test set), by seeking
            for i in json.loads(Path(a.frames).read_text()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, bgr = cap.read()
                if ok:
                    dets[i] = candidates(model, bgr, dev, a.thr, a.max_n)
            lo = hi = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        for i in range(lo, hi):
            ok, bgr = cap.read()
            if not ok:
                break
            if (i - lo) % a.every == 0:
                dets[i] = candidates(model, bgr, dev, a.thr, a.max_n)
            if (i - lo) % 1000 == 0:
                print(f"clip frame {i}/{hi}", flush=True)
        out["clip"] = dets
    if a.vl_root:
        wins = dev_windows(Path(a.vl_root).expanduser(), a.matches)
        out["windows"] = {}
        for key, paths in wins.items():
            out["windows"][key] = [candidates(model, cv2.imread(str(p)), dev, a.thr, a.max_n) for p in paths]
            print("window", key, flush=True)
    Path(a.out).write_bytes(pickle.dumps(out))
    print("wrote", a.out)


def cmd_pseudo(a):
    from footlytics.perception.ball_pseudo import build_tracks, label_rows

    d = pickle.loads(Path(a.cands).read_bytes())
    labels = {}
    summary = {}
    if "clip" in d:
        frames = sorted(d["clip"])
        # consecutive-frame lists (every=1 required for tracking)
        assert frames == list(range(frames[0], frames[-1] + 1)), "clip candidates must be stride 1"
        seq = [d["clip"][f] for f in frames]
        tracks = build_tracks(seq)
        lab = label_rows(tracks, seq)
        for k, v in lab.items():
            labels[f"clip/{frames[0] + k:06d}"] = {**v, "src": "clip"}
        summary["clip"] = {"frames": len(frames), "tracks": len(tracks), "labelled": len(lab)}
    for key, seq in d.get("windows", {}).items():
        tracks = build_tracks(seq)
        lab = label_rows(tracks, seq)
        for k, v in lab.items():
            labels[f"win/{key}/{k - WINDOW:+03d}"] = {**v, "src": "win"}
        summary[key] = {"tracks": len(tracks), "labelled": len(lab)}
    Path(a.out).write_text(json.dumps({"labels": labels, "summary": summary}, indent=0))
    print(json.dumps(summary)[:2000])
    print("labels", len(labels), "non-interp", sum(not v["interp"] for v in labels.values()))


class Frames:
    """Random access to the clip (cv2 seek) and the window jpgs by pseudo-label key."""

    def __init__(self, video, vl_root):
        self.cap = cv2.VideoCapture(str(Path(video).expanduser())) if video else None
        self.vl_root = Path(vl_root).expanduser() if vl_root else None

    def get(self, key: str):
        if key.startswith("clip/"):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(key.split("/")[1]))
            ok, bgr = self.cap.read()
            return bgr if ok else None
        _, match, idx, off = key.split("/")
        return cv2.imread(str(self.vl_root / match / idx / f"{int(off):+03d}.jpg"))


def neighbour(key: str, d: int) -> str:
    parts = key.split("/")
    if parts[0] == "clip":
        return f"clip/{int(parts[1]) + d:06d}"
    return "/".join(parts[:3] + [f"{int(parts[3]) + d:+03d}"])


def strip(frames: Frames, key: str, box, size=96, scale=3, step=4, label=""):
    """Three crops (t-step, t, t+step) around the box centre at t, corner ticks around the box on t."""
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    cells = []
    for d in (-step, 0, step):
        img = frames.get(neighbour(key, d))
        if img is None:
            img = np.zeros((1080, 1920, 3), np.uint8)
        x0 = int(np.clip(cx - size / 2, 0, img.shape[1] - size))
        y0 = int(np.clip(cy - size / 2, 0, img.shape[0] - size))
        cell = cv2.resize(img[y0:y0 + size, x0:x0 + size], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if d == 0:
            pad = 5
            xa, ya = (box[0] - pad - x0) * scale, (box[1] - pad - y0) * scale
            xb, yb = (box[0] + box[2] + pad - x0) * scale, (box[1] + box[3] + pad - y0) * scale
            L = 10
            for (px, py, sx, sy) in ((xa, ya, 1, 1), (xb, ya, -1, 1), (xa, yb, 1, -1), (xb, yb, -1, -1)):
                cv2.line(cell, (int(px), int(py)), (int(px + sx * L), int(py)), (0, 0, 255), 1)
                cv2.line(cell, (int(px), int(py)), (int(px), int(py + sy * L)), (0, 0, 255), 1)
        cv2.rectangle(cell, (0, 0), (cell.shape[1] - 1, cell.shape[0] - 1), (60, 60, 60), 1)
        cells.append(cell)
    row = np.concatenate(cells, 1)
    cv2.putText(row, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return row


def cmd_sheet(a):
    import random

    lab = json.loads(Path(a.labels).read_text())["labels"]
    keys = sorted(k for k, v in lab.items() if v["interp"] == a.interp and (a.src is None or v["src"] == a.src)
                  and (a.lo is None or (k.startswith("clip/") and a.lo <= int(k.split("/")[1]) < a.hi)))
    random.Random(a.seed).shuffle(keys)
    keys = keys[:a.n]
    frames = Frames(a.video, a.vl_root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for s in range(0, len(keys), a.per_sheet):
        rows = []
        for i, k in enumerate(keys[s:s + a.per_sheet]):
            rows.append(strip(frames, k, lab[k]["box"], label=str(s + i)))
            index[s + i] = k
        cv2.imwrite(str(out / f"sheet-{s // a.per_sheet:02d}.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    (out / "index.json").write_text(json.dumps(index))
    print("sheets", (len(keys) + a.per_sheet - 1) // a.per_sheet, "samples", len(keys))


def test_keys() -> list[str]:
    """The test frames: 118 clip frames from 8:00 to 10:00 (the pseudo-labelled training part of the
    clip ends at 6:30), and 4 frames of each ninh-binh dev window (a match never trained on)."""
    keys = [f"clip/{int(round(f)):06d}" for f in np.linspace(12000, 14950, 118)]
    man = json.loads(MANIFEST.read_text())["matches"]["ninh-binh-pvf-cand"]["frames"]
    for idx, meta in sorted(man.items()):
        if meta["split"] == "dev":
            keys += [f"win/ninh-binh-pvf-cand/{idx}/{off:+03d}" for off in (-12, -4, 4, 12)]
    return keys


def union_candidates(cand_files: list[str], key: str, topn=8, merge_px=10.0) -> np.ndarray:
    """Candidates of several detectors for one frame, deduplicated, best first (origin dropped)."""
    rows = []
    for f in cand_files:
        d = pickle.loads(Path(f).read_bytes()) if isinstance(f, str) else f
        if key.startswith("clip/"):
            c = d["clip"][int(key.split("/")[1])]
        else:
            _, m, idx, off = key.split("/")
            c = d["windows"][f"{m}/{idx}"][int(off) + WINDOW]
        rows += [r for r in c[:topn]]
    rows.sort(key=lambda r: -r[4])
    out = []
    for r in rows:
        cr = np.array([r[0] + r[2] / 2, r[1] + r[3] / 2])
        if all(np.hypot(*(cr - np.array([o[0] + o[2] / 2, o[1] + o[3] / 2]))) > merge_px for o in out):
            out.append(r)
    return np.array(out).reshape(-1, 6)[:10]


def cmd_testsheet(a):
    keys = json.loads(Path(a.keys).read_text()) if a.keys else test_keys()
    frames = Frames(a.video, a.vl_root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = [pickle.loads(Path(f).read_bytes()) for f in a.cands]
    hints = {}
    for n, key in enumerate(keys):
        if a.only is not None and n not in a.only:
            continue
        img = frames.get(key)
        c = union_candidates(cache, key)
        hints[n] = {"key": key, "cands": [[float(v) for v in r[:5]] for r in c]}
        sc = 0.8
        big = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        for x in range(0, 1920, 100):
            cv2.line(big, (int(x * sc), 0), (int(x * sc), 8), (0, 255, 255), 1)
            if x % 200 == 0:
                cv2.putText(big, str(x), (int(x * sc) + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for y in range(0, 1080, 100):
            cv2.line(big, (0, int(y * sc)), (8, int(y * sc)), (0, 255, 255), 1)
            if y % 200 == 0:
                cv2.putText(big, str(y), (10, int(y * sc) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for i, r in enumerate(c):
            cx, cy = (r[0] + r[2] / 2) * sc, (r[1] + r[3] / 2) * sc
            cv2.circle(big, (int(cx), int(cy)), 13, (0, 0, 255), 1)
            cv2.putText(big, str(i), (int(cx) + 14, int(cy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(big, f"#{n}", (big.shape[1] - 80, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"{n:03d}-full.jpg"), big, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if len(c):
            cells = []
            for i, r in enumerate(c):
                cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
                x0 = int(np.clip(cx - 40, 0, 1920 - 80))
                y0 = int(np.clip(cy - 40, 0, 1080 - 80))
                cell = cv2.resize(img[y0:y0 + 80, x0:x0 + 80], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                cv2.putText(cell, str(i), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(cell, (0, 0), (239, 239), (60, 60, 60), 1)
                cells.append(cell)
            while len(cells) % 5:
                cells.append(np.zeros_like(cells[0]))
            rows = [np.concatenate(cells[i:i + 5], 1) for i in range(0, len(cells), 5)]
            cv2.imwrite(str(out / f"{n:03d}-cands.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    path = out / "hints.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update({str(k): v for k, v in hints.items()})
    path.write_text(json.dumps(old))
    print("frames", len(hints))


def cmd_zoom(a):
    frames = Frames(a.video, a.vl_root)
    img = frames.get(a.key)
    half, sc = 50, 6
    x0 = int(np.clip(a.x - half, 0, 1920 - 2 * half))
    y0 = int(np.clip(a.y - half, 0, 1080 - 2 * half))
    cell = cv2.resize(img[y0:y0 + 2 * half, x0:x0 + 2 * half], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    for g in range(0, 2 * half, 10):
        col = (0, 255, 255)
        cv2.line(cell, (g * sc, 0), (g * sc, 10), col, 1)
        cv2.line(cell, (0, g * sc), (10, g * sc), col, 1)
        cv2.putText(cell, str(x0 + g), (g * sc + 2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        cv2.putText(cell, str(y0 + g), (12, g * sc + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    cv2.imwrite(a.out, cell, [cv2.IMWRITE_JPEG_QUALITY, 92])


def cmd_extra(a):
    """Pseudo-labels -> training frames on disk + the `--extra` json for train_ball_detr.py.

    Clip frames come from the pseudo-labelled head of the clip only (< --clip-end seconds, so the
    test part of the clip is never seen); windows from the *dev* windows of --train-matches.
    """
    lab = json.loads(Path(a.labels).read_text())["labels"]
    frames = Frames(a.video, a.vl_root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads(MANIFEST.read_text())["matches"]
    extra, skipped = [], 0
    for key, v in sorted(lab.items()):
        if key.startswith("clip/"):
            f = int(key.split("/")[1])
            if f >= a.clip_end * 25 or f % a.stride:
                continue
            dst = out / f"clip_{f:06d}.jpg"
        else:
            _, m, idx, off = key.split("/")
            if m not in a.train_matches or man[m]["frames"][idx]["split"] != "dev" or idx in a.skip_windows \
                    or int(off) % a.win_stride:
                continue
            dst = out / f"win_{m}_{idx}_{int(off):+03d}.jpg"
        if not dst.exists():
            img = frames.get(key)
            if img is None:
                skipped += 1
                continue
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        extra.append([str(dst), v["box"]])
    Path(a.extra_json).write_text(json.dumps(extra))
    print("extra frames", len(extra), "skipped", skipped)


def cmd_eval(a):
    """Score candidates against test labels: {key: {"box": [x, y, w, h] | None, ...}}."""
    d = pickle.loads(Path(a.cands).read_bytes())
    gt = json.loads(Path(a.gt).read_text())["frames"]
    lookup = {}
    for f, c in d.get("clip", {}).items():
        lookup[f"clip/{f:06d}"] = c
    for key, seq in d.get("windows", {}).items():
        for k, c in enumerate(seq):
            lookup[f"win/{key}/{k - WINDOW:+03d}"] = c
    res = {}
    groups: dict[str, list] = {}
    for key, g in gt.items():
        if g.get("status") == "unsure":
            continue
        groups.setdefault("all", []).append(key)
        groups.setdefault("clip" if key.startswith("clip") else "windows", []).append(key)
    for name, keys in groups.items():
        for thr in a.thrs:
            vis = hit_top = hit_any = fp = nvis = nfr = 0
            fp_novis = nnovis = 0
            for key in keys:
                g = gt[key]["box"]
                c = lookup[key]
                c = c[c[:, 4] >= thr] if len(c) else c
                nfr += 1
                if g is None:
                    nnovis += 1
                    fp_novis += len(c)
                    fp += len(c)
                    continue
                nvis += 1
                gc = np.array([g[0] + g[2] / 2, g[1] + g[3] / 2])
                dist = [math.hypot(r[0] + r[2] / 2 - gc[0], r[1] + r[3] / 2 - gc[1]) for r in c]
                near = [x <= HIT_PX for x in dist]
                hit_any += any(near)
                hit_top += bool(dist) and near[0]
                fp += sum(1 for x in near if not x)
            res[f"{name}@{thr}"] = {
                "frames": nfr, "ball_visible": nvis, "not_visible": nnovis,
                "recall_any": round(hit_any / max(nvis, 1), 4),
                "recall_top1": round(hit_top / max(nvis, 1), 4),
                "fp_per_frame": round(fp / max(nfr, 1), 4),
                "fp_per_frame_novis": round(fp_novis / max(nnovis, 1), 4) if nnovis else None}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("detect")
    p.add_argument("--weights", required=True)
    p.add_argument("--video")
    p.add_argument("--start", type=float, default=0.0, help="seconds")
    p.add_argument("--end", type=float, default=0.0, help="seconds (0 = to the end)")
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--vl-root", help="data/vleague root: runs over the dev windows only")
    p.add_argument("--matches", nargs="*")
    p.add_argument("--frames", help="json list of clip frame indices to run instead of a stride-1 sweep")
    p.add_argument("--thr", type=float, default=0.05)
    p.add_argument("--max-n", type=int, default=15)
    p.add_argument("--out", required=True)
    p = sub.add_parser("pseudo")
    p.add_argument("--cands", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("sheet")
    p.add_argument("--labels", required=True)
    p.add_argument("--video")
    p.add_argument("--vl-root")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--per-sheet", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interp", action="store_true")
    p.add_argument("--src", choices=["clip", "win"])
    p.add_argument("--lo", type=int, help="clip frame range of the sample [lo, hi)")
    p.add_argument("--hi", type=int)
    p = sub.add_parser("testsheet")
    p.add_argument("--cands", nargs="+", required=True)
    p.add_argument("--video")
    p.add_argument("--vl-root")
    p.add_argument("--keys")
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--out", required=True)
    p = sub.add_parser("zoom")
    p.add_argument("--video")
    p.add_argument("--vl-root")
    p.add_argument("--key", required=True)
    p.add_argument("--x", type=float, required=True)
    p.add_argument("--y", type=float, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("extra")
    p.add_argument("--labels", required=True)
    p.add_argument("--video")
    p.add_argument("--vl-root")
    p.add_argument("--out", required=True)
    p.add_argument("--extra-json", required=True)
    p.add_argument("--clip-end", type=float, default=390.0, help="seconds of the clip usable for training")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--win-stride", type=int, default=2)
    p.add_argument("--train-matches", nargs="+", default=["cong-an-tp-ho-chi-minh-ha-noi", "shb-da-nang-hai-phong"])
    p.add_argument("--skip-windows", nargs="*", default=["0171012"], help="dev windows that overlap the test part of the clip")
    p = sub.add_parser("eval")
    p.add_argument("--cands", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--thrs", type=float, nargs="+", default=[0.05, 0.2, 0.3, 0.5])
    p.add_argument("--out")
    a = ap.parse_args()
    {"detect": cmd_detect, "pseudo": cmd_pseudo, "sheet": cmd_sheet, "extra": cmd_extra, "testsheet": cmd_testsheet, "zoom": cmd_zoom, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

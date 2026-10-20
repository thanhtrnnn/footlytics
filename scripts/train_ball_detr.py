"""Fine-tune RT-DETRv2 (Apache-2.0) into a single-class football detector on SoccerNet GSR *train*.

Run on the benchmark host (shared lab box, see docs):

    CUDA_MPS_PIPE_DIRECTORY=/tmp/iec-nomps ~/footlytics-bench/venv/bin/python \
        scripts/train_ball_detr.py --out ~/footlytics-bench/weights/ball-rtdetrv2 --hours 4

Data: `<root>/data/gsr/train.zip` (images are read straight out of the zip) and the labels
`<root>/data/gsr/labels/train/<SEQ>/Labels-GameState.json`. valid/test are the benchmark and are
refused. A few train clips are held out (`--holdout`) for the training-time check.

Model: `PekingU/rtdetr_v2_r50vd` (Apache-2.0, COCO). The COCO head is kept: the ball is COCO class
32 ("sports ball"), so the pretrained prior is the starting point and only that class is trained.
The ball is ~18 px on a 1080p frame, so training uses full-resolution 960 px crops (no downscale)
and inference runs on the full 1920x1088 frame.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

BASE = "PekingU/rtdetr_v2_r50vd"
BALL_CLS = 32            # COCO "sports ball" in the 80-class head
CROP = 960
RADIUS_PX = 20.0         # same gate as the bench's ball_det_recall


def load_split(root: Path, split: str) -> dict[str, list]:
    """seq -> [(file_name, (x, y, w, h) or None)] from the GSR labels."""
    if split != "train":
        raise SystemExit("refusing to read %r: valid/test are the benchmark" % split)
    out = {}
    for p in sorted((root / "data/gsr/labels" / split).glob("SNGS-*/Labels-GameState.json")):
        lab = json.loads(p.read_text())
        ball = {}
        for a in lab["annotations"]:
            if a.get("supercategory") == "object" and a["attributes"].get("role") == "ball" \
                    and a.get("bbox_image"):
                b = a["bbox_image"]
                ball[a["image_id"]] = (b["x"], b["y"], b["w"], b["h"])
        out[p.parent.name] = [(i["file_name"], ball.get(i["image_id"])) for i in lab["images"]]
    return out


class CropSet(torch.utils.data.Dataset):
    def __init__(self, zip_path: Path, frames: list[tuple], size: int, n: int, pos_share=0.75):
        self.zip_path, self.frames, self.size, self.n, self.pos = zip_path, frames, size, n, pos_share
        self._z = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._z is None:
            self._z = zipfile.ZipFile(self.zip_path)
        rng = random.Random(i * 7919 + int(time.time() * 1000) % 100003)
        seq, name, box = rng.choice(self.frames)
        img = cv2.imdecode(np.frombuffer(self._z.read(f"{seq}/img1/{name}"), np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        s = rng.uniform(0.8, 1.25)                       # source px per output px
        c = min(int(self.size * s), H, W)
        if box and rng.random() < self.pos:
            bx, by = box[0] + box[2] / 2, box[1] + box[3] / 2
            x0 = int(np.clip(bx - rng.uniform(0.1, 0.9) * c, 0, W - c))
            y0 = int(np.clip(by - rng.uniform(0.1, 0.9) * c, 0, H - c))
        else:
            x0, y0 = rng.randint(0, W - c), rng.randint(0, H - c)
        crop = img[y0:y0 + c, x0:x0 + c]
        boxes = []
        if box:
            x, y, w, h = box
            ix0, iy0, ix1, iy1 = max(x, x0), max(y, y0), min(x + w, x0 + c), min(y + h, y0 + c)
            if ix1 > ix0 and iy1 > iy0 and (ix1 - ix0) * (iy1 - iy0) >= 0.6 * w * h:
                boxes = [((ix0 + ix1) / 2 - x0, (iy0 + iy1) / 2 - y0, ix1 - ix0, iy1 - iy0)]
        k = self.size / c
        crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR if k > 1 else cv2.INTER_AREA)
        b = np.array(boxes, np.float32).reshape(-1, 4) * k / self.size
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
            b[:, 0] = 1 - b[:, 0]
        # mild photometric jitter (BGR uint8)
        crop = np.clip(crop.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255)
        rgb = np.ascontiguousarray(crop[..., ::-1]) / 255.0
        return (torch.from_numpy(rgb).permute(2, 0, 1).float(),
                {"class_labels": torch.full((len(b),), BALL_CLS, dtype=torch.long),
                 "boxes": torch.from_numpy(b).float()})


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


@torch.no_grad()
def evaluate(model, zip_path: Path, frames: list[tuple], device, thr=0.05, every=1) -> dict:
    """Candidate recall within RADIUS_PX of the GT ball on full frames (held-out train clips)."""
    model.eval()
    z = zipfile.ZipFile(zip_path)
    n = hit_any = hit_top = ncand = 0
    for seq, name, box in frames[::every]:
        if box is None:
            continue
        img = cv2.imdecode(np.frombuffer(z.read(f"{seq}/img1/{name}"), np.uint8), cv2.IMREAD_COLOR)
        cands = predict_frame(model, img[..., ::-1], device, thr)
        n += 1
        ncand += len(cands)
        gx, gy = box[0] + box[2] / 2, box[1] + box[3] / 2
        d = [math.hypot(c[0] + c[2] / 2 - gx, c[1] + c[3] / 2 - gy) for c in cands]
        hit_any += bool(d and min(d) <= RADIUS_PX)
        hit_top += bool(d and d[0] <= RADIUS_PX)
    model.train()
    return {"frames": n, "det_recall": hit_any / max(n, 1), "top1_recall": hit_top / max(n, 1),
            "cands_per_frame": ncand / max(n, 1)}


def predict_frame(model, rgb: np.ndarray, device, thr: float, max_n: int = 15):
    """Full-frame inference -> [(x, y, w, h, score)] most confident first (shared with the pipeline)."""
    from footlytics.perception.ball_detr import infer
    return infer(model, rgb, device, thr, max_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/footlytics-bench")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--hours", type=float, default=4.0, help="wall-clock cap for training")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--holdout", type=int, default=6, help="train clips held out for the check")
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    root, out = Path(a.root).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    from transformers import RTDetrV2ForObjectDetection

    data = load_split(root, "train")
    seqs = sorted(data)
    hold = seqs[-a.holdout:] if a.holdout else []
    tr = [(s, n, b) for s in seqs if s not in hold for n, b in data[s]]
    ho = [(s, n, b) for s in hold for n, b in data[s]]
    zp = root / "data/gsr/train.zip"
    print(f"train clips {len(seqs) - len(hold)} frames {len(tr)} (ball in {sum(b is not None for *_, b in tr)}); "
          f"held out {hold}", flush=True)

    dev = torch.device("cuda")
    model = RTDetrV2ForObjectDetection.from_pretrained(a.base).to(dev).train()
    bb = [p for n, p in model.named_parameters() if "backbone" in n]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr * 0.1}, {"params": rest, "lr": a.lr}], weight_decay=1e-4)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    dl = torch.utils.data.DataLoader(CropSet(zp, tr, CROP, a.steps * a.bs), batch_size=a.bs, num_workers=a.workers,
                                     collate_fn=collate, persistent_workers=True, prefetch_factor=4)
    t0, best, log = time.time(), -1.0, []
    for step, (x, labels) in enumerate(dl, 1):
        x = x.to(dev, non_blocking=True)
        labels = [{k: v.to(dev) for k, v in l.items()} for l in labels]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(pixel_values=x, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step}/{a.steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        done = step >= a.steps or time.time() - t0 > a.hours * 3600
        if step % a.eval_every == 0 or done:
            ev = evaluate(model, zp, ho, dev, every=5)
            print(f"EVAL step {step} {json.dumps(ev)}", flush=True)
            log.append({"step": step, "seconds": round(time.time() - t0), **ev})
            model.save_pretrained(out / "last")
            if ev["det_recall"] > best:
                best = ev["det_recall"]
                model.save_pretrained(out / "best")
            (out / "train-log.json").write_text(json.dumps(log, indent=1))
        if done:
            break
    print(f"done in {(time.time() - t0) / 3600:.2f} h; best held-out det_recall {best:.4f}", flush=True)


if __name__ == "__main__":
    main()

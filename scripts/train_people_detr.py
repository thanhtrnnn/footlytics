"""Fine-tune RT-DETRv2 (Apache-2.0) into a 4-class football people detector (`player`, `goalkeeper`,
`referee`, `other`) on SoccerNet GSR *train*, eliminating the AGPL Ultralytics people dependency.

Run on the benchmark host:

    CUDA_MPS_PIPE_DIRECTORY=/home/iec/bela/footlytics-bench/cache/no-mps \
    FOOTLYTICS_BENCH_ROOT=/home/iec/bela/footlytics-bench \
    /home/iec/bela/footlytics-bench/venv/bin/python scripts/train_people_detr.py \
        --root /home/iec/bela/footlytics-bench \
        --out /home/iec/bela/footlytics-bench/weights/people-rtdetrv2 \
        --steps 4000 --bs 6

Data: `<root>/data/gsr/train.zip` and `<root>/data/gsr/labels/train/<SEQ>/Labels-GameState.json`.
`valid` and `test` are strictly refused. A held-out subset of train clips (`--holdout`) selects the
`best` checkpoint by person IoU>=0.5 F1 + goalkeeper/referee role accuracy.
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
ROLE_TO_CLS = {
    "player": 0,
    "goalkeeper": 1,
    "referee": 2,
    "other": 3,
}


def load_split(root: Path, split: str) -> dict[str, list[tuple[str, list[tuple[float, float, float, float, int]]]]]:
    """seq -> [(file_name, [(x, y, w, h, cls_id), ...])] from GSR train labels only."""
    if split != "train":
        raise SystemExit(f"refusing to read {split!r}: valid/test are the benchmark")
    out: dict[str, list[tuple[str, list[tuple[float, float, float, float, int]]]]] = {}
    for p in sorted((root / "data/gsr/labels" / split).glob("SNGS-*/Labels-GameState.json")):
        lab = json.loads(p.read_text())
        by_img: dict[str, list[tuple[float, float, float, float, int]]] = {}
        for a in lab["annotations"]:
            if a.get("supercategory") != "object":
                continue
            role = str((a.get("attributes") or {}).get("role") or "other").lower()
            if role == "ball":
                continue
            bi = a.get("bbox_image")
            if not bi:
                continue
            w, h = float(bi["w"]), float(bi["h"])
            if w <= 2.0 or h <= 2.0:
                continue
            cls_id = ROLE_TO_CLS.get(role, 3)
            by_img.setdefault(str(a["image_id"]), []).append(
                (float(bi["x"]), float(bi["y"]), w, h, cls_id)
            )
        out[p.parent.name] = [
            (str(img["file_name"]), by_img.get(str(img["image_id"]), []))
            for img in lab["images"]
        ]
    return out


def seed_person_heads_from_coco(model) -> int:
    """Copy COCO class 0 ('person') classification rows into classes 1..3 so goalkeeper/referee/other
    start from a strong person detector prior rather than unrelated COCO classes (bicycle/car/motorcycle)."""
    copied = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim == 2 and param.shape[0] == 80:
                param[1:4].copy_(param[0:1].expand(3, -1))
                copied += 1
            elif param.ndim == 1 and param.shape[0] == 80:
                param[1:4].copy_(param[0:1].expand(3) - 1.2)
                copied += 1
    return copied


class FullFrameSet(torch.utils.data.Dataset):
    """Full-frame multi-scale football samples resized to `(target_h, target_w)` multiples of 32."""

    def __init__(self, zip_path: Path, frames: list[tuple[str, str, list]], imgsz: int, n: int):
        self.zip_path = zip_path
        self.frames = frames
        self.imgsz = imgsz
        self.n = n
        self._z: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        if self._z is None:
            self._z = zipfile.ZipFile(self.zip_path)
        rng = random.Random(i * 104729 + int(time.time() * 1000) % 100003)
        seq, name, boxes = rng.choice(self.frames)
        raw = self._z.read(f"{seq}/img1/{name}")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        oh, ow = img.shape[:2]

        # Random mild horizontal/vertical crop (85%..100%) to vary scale while preserving pitch context
        if rng.random() < 0.65:
            frac = rng.uniform(0.84, 1.0)
            cw, ch = round(ow * frac), round(oh * frac)
            x0 = rng.randint(0, max(0, ow - cw))
            y0 = rng.randint(0, max(0, oh - ch))
            img = img[y0 : y0 + ch, x0 : x0 + cw]
        else:
            x0, y0, cw, ch = 0, 0, ow, oh

        out_w = (self.imgsz // 32) * 32
        out_h = -(-round(ch * (out_w / max(cw, 1))) // 32) * 32
        resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

        kept_boxes: list[list[float]] = []
        kept_cls: list[int] = []
        for bx, by, bw, bh, cid in boxes:
            ix0 = max(bx, float(x0))
            iy0 = max(by, float(y0))
            ix1 = min(bx + bw, float(x0 + cw))
            iy1 = min(by + bh, float(y0 + ch))
            iw, ih = ix1 - ix0, iy1 - iy0
            if iw <= 2.0 or ih <= 2.0 or (iw * ih) < 0.45 * (bw * bh):
                continue
            cx_n = ((ix0 + ix1) * 0.5 - x0) / cw
            cy_n = ((iy0 + iy1) * 0.5 - y0) / ch
            w_n = iw / cw
            h_n = ih / ch
            kept_boxes.append([cx_n, cy_n, w_n, h_n])
            kept_cls.append(int(cid))

        b_arr = np.array(kept_boxes, dtype=np.float32).reshape(-1, 4)
        c_arr = np.array(kept_cls, dtype=np.int64)
        if rng.random() < 0.5:
            resized = resized[:, ::-1]
            if len(b_arr):
                b_arr[:, 0] = 1.0 - b_arr[:, 0]

        resized = np.clip(
            resized.astype(np.float32) * rng.uniform(0.82, 1.18) + rng.uniform(-18.0, 18.0),
            0.0,
            255.0,
        )
        rgb = np.ascontiguousarray(resized[..., ::-1]) / 255.0
        return (
            torch.from_numpy(rgb).permute(2, 0, 1).float(),
            {
                "class_labels": torch.from_numpy(c_arr).long(),
                "boxes": torch.from_numpy(b_arr).float(),
            },
        )


def collate(batch):
    max_h = max(x.shape[1] for x, _ in batch)
    max_w = max(x.shape[2] for x, _ in batch)
    padded = torch.zeros((len(batch), 3, max_h, max_w), dtype=torch.float32)
    labels = []
    for i, (x, lab) in enumerate(batch):
        _, h, w = x.shape
        padded[i, :, :h, :w] = x
        if len(lab["boxes"]) and (h != max_h or w != max_w):
            b = lab["boxes"].clone()
            b[:, 0] *= w / max_w
            b[:, 2] *= w / max_w
            b[:, 1] *= h / max_h
            b[:, 3] *= h / max_h
            labels.append({"class_labels": lab["class_labels"], "boxes": b})
        else:
            labels.append(lab)
    return padded, labels


def _box_iou_matrix(a_xywh: np.ndarray, b_xywh: np.ndarray) -> np.ndarray:
    if len(a_xywh) == 0 or len(b_xywh) == 0:
        return np.zeros((len(a_xywh), len(b_xywh)), dtype=np.float32)
    ax1, ay1 = a_xywh[:, 0:1], a_xywh[:, 1:2]
    ax2, ay2 = ax1 + a_xywh[:, 2:3], ay1 + a_xywh[:, 3:4]
    bx1, by1 = b_xywh[:, 0], b_xywh[:, 1]
    bx2, by2 = bx1 + b_xywh[:, 2], by1 + b_xywh[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
    area_a = a_xywh[:, 2:3] * a_xywh[:, 3:4]
    area_b = b_xywh[:, 2] * b_xywh[:, 3]
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


@torch.no_grad()
def evaluate(
    model,
    zip_path: Path,
    frames: list[tuple[str, str, list]],
    device,
    imgsz: int = 1280,
    thr: float = 0.25,
    every: int = 10,
) -> dict:
    from footlytics.perception.detect import nms
    from footlytics.perception.people_detr import infer_people

    model.eval()
    z = zipfile.ZipFile(zip_path)
    tp = fp = fn = role_ok = 0
    gk_gt = gk_hit = ref_gt = ref_hit = 0
    sampled = frames[::every]
    for seq, name, gt_list in sampled:
        raw = z.read(f"{seq}/img1/{name}")
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        preds = infer_people(model, bgr[..., ::-1], device, thr=thr, max_n=300, imgsz=imgsz)
        if len(preds):
            keep: list[int] = []
            for c in np.unique(preds[:, 5]):
                idx = np.flatnonzero(preds[:, 5] == c)
                keep.extend(idx[nms(preds[idx, :4], preds[idx, 4], 0.55)])
            preds = preds[np.array(sorted(keep), dtype=int)]
        gt_arr = np.array([g[:4] for g in gt_list], dtype=np.float32).reshape(-1, 4)
        gt_cls = [int(g[4]) for g in gt_list]
        for c in gt_cls:
            if c == 1:
                gk_gt += 1
            elif c == 2:
                ref_gt += 1
        if len(gt_arr) == 0:
            fp += len(preds)
            continue
        if len(preds) == 0:
            fn += len(gt_arr)
            continue
        ious = _box_iou_matrix(gt_arr, preds[:, :4])
        used_p: set[int] = set()
        matched_g = 0
        for gi in range(len(gt_arr)):
            best_pi = -1
            best_iou = 0.5
            for pi in range(len(preds)):
                if pi not in used_p and ious[gi, pi] >= best_iou:
                    best_iou = float(ious[gi, pi])
                    best_pi = pi
            if best_pi >= 0:
                used_p.add(best_pi)
                matched_g += 1
                p_cid = int(preds[best_pi, 5])
                if p_cid == gt_cls[gi]:
                    role_ok += 1
                    if gt_cls[gi] == 1:
                        gk_hit += 1
                    elif gt_cls[gi] == 2:
                        ref_hit += 1
        tp += matched_g
        fn += len(gt_arr) - matched_g
        fp += len(preds) - len(used_p)
    model.train()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    role_acc = role_ok / max(tp, 1)
    return {
        "frames": len(sampled),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "role_acc": round(role_acc, 4),
        "gk_recall": round(gk_hit / max(gk_gt, 1), 4),
        "ref_recall": round(ref_hit / max(ref_gt, 1), 4),
        "score": round(f1 * 0.75 + role_acc * 0.25, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/footlytics-bench")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=3600)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--holdout", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=900)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=250)
    a = ap.parse_args()

    root, out = Path(a.root).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    from transformers import RTDetrV2ForObjectDetection

    data = load_split(root, "train")
    seqs = sorted(data)
    hold = seqs[-a.holdout :] if a.holdout else []
    tr = [(s, n, b) for s in seqs if s not in hold for n, b in data[s] if b]
    ho = [(s, n, b) for s in hold for n, b in data[s]]
    zp = root / "data/gsr/train.zip"
    n_boxes = sum(len(b) for *_, b in tr)
    print(
        f"train clips {len(seqs) - len(hold)} frames {len(tr)} boxes {n_boxes}; held out {hold}",
        flush=True,
    )

    dev = torch.device("cuda")
    model = RTDetrV2ForObjectDetection.from_pretrained(a.base).to(dev)
    seeded = seed_person_heads_from_coco(model)
    model.train()
    print(f"seeded {seeded} classification head tensors from COCO person prior", flush=True)

    bb = [p for n, p in model.named_parameters() if "backbone" in n]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW(
        [{"params": bb, "lr": a.lr * 0.1}, {"params": rest, "lr": a.lr}], weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(a.warmup, 1))
        * 0.5
        * (1.0 + math.cos(math.pi * min(s, a.steps) / max(a.steps, 1))),
    )
    dl = torch.utils.data.DataLoader(
        FullFrameSet(zp, tr, a.imgsz, a.steps * a.bs),
        batch_size=a.bs,
        num_workers=a.workers,
        collate_fn=collate,
        persistent_workers=(a.workers > 0),
        prefetch_factor=3 if a.workers > 0 else None,
    )

    t0, best, log = time.time(), -1.0, []
    for step, (x, labels) in enumerate(dl, 1):
        x = x.to(dev, non_blocking=True)
        labels = [{k: v.to(dev) for k, v in lab.items()} for lab in labels]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(pixel_values=x, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step}/{a.steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        done = step >= a.steps or (time.time() - t0) > a.hours * 3600
        if step % a.eval_every == 0 or done:
            ev = evaluate(model, zp, ho, dev, imgsz=a.imgsz, every=10)
            print(f"EVAL step {step} {json.dumps(ev)}", flush=True)
            log.append({"step": step, "seconds": round(time.time() - t0), **ev})
            model.save_pretrained(out / "last")
            if ev["score"] > best:
                best = ev["score"]
                model.save_pretrained(out / "best")
            (out / "train-log.json").write_text(json.dumps(log, indent=2) + "\n")
        if done:
            break
    print(f"done in {(time.time() - t0) / 3600:.2f} h; best held-out score {best:.4f}", flush=True)


if __name__ == "__main__":
    main()

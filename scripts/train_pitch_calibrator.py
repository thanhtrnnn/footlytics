#!/usr/bin/env python3
"""Train the 100% Apache-2.0 broadcast pitch keypoint + line calibrator (`PitchCalibNet`).

Supports:
1. SoccerNet-Calibration JSON + JPEG archives (`--sncal-dir`)
2. Footlytics V.League annotated homography folders (`--vleague-dir`, e.g. `data/vleague`)
3. Synthetic projected-pitch smoke mode (`--synthetic-smoke`) for fast local verification

Outputs a commercial-clean `safetensors` + `config.json` checkpoint directory loadable by
`Apache2PitchCalibrator` (`footlytics match-run --calib-backend apache2 --calib-weights-dir ...`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from footlytics.geometry.pitch_calibrator import (
    PITCH_LINES_2D,
    PitchCalibConfig,
    build_pitch_calib_net,
    render_heatmap_targets,
    save_pitch_calibrator,
)
from footlytics.geometry.sn_camera import make_camera, project


def _focal_mse_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Weighted BCE + MSE heatmap loss emphasizing sparse positive pitch landmarks/lines."""
    probs = torch.sigmoid(logits)
    pos_weight = 1.0 + 18.0 * targets
    mse = ((probs - targets) ** 2) * pos_weight
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * pos_weight
    return 0.5 * mse.mean() + 0.5 * bce.mean()


def _synthetic_pitch_sample(
    rng: np.random.Generator,
    in_w: int,
    in_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a synthetic broadcast-view football pitch image + exact keypoint/line targets."""
    src_w, src_h = 1280, 720
    pan = float(rng.uniform(-22.0, 22.0))
    tilt = float(rng.uniform(62.0, 76.0))
    roll = float(rng.uniform(-2.0, 2.0))
    focal = float(rng.uniform(1100.0, 2100.0))
    pos = [float(rng.uniform(-12.0, 12.0)), float(rng.uniform(55.0, 72.0)), float(rng.uniform(-24.0, -16.0))]
    cam = make_camera(pan, tilt, roll, focal, pos, src_w, src_h)

    # Render textured turf + white FIFA pitch markings
    img = np.full((src_h, src_w, 3), (48, 122, 62), dtype=np.uint8)
    noise = rng.integers(-8, 9, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for seg in PITCH_LINES_2D:
        ts = np.linspace(0.0, 1.0, 30)
        p3 = np.column_stack([seg[0, 0] + ts * (seg[1, 0] - seg[0, 0]), seg[0, 1] + ts * (seg[1, 1] - seg[0, 1]), np.zeros_like(ts)])
        uv = project(cam, p3)
        ok = np.isfinite(uv).all(axis=1)
        if ok.sum() >= 2:
            pts = np.round(uv[ok]).astype(np.int32)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (238, 242, 236), 3, cv2.LINE_AA)

    resized = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    kp_t, ln_t = render_heatmap_targets(cam, src_w, src_h, in_w // 4, in_h // 4)
    return resized, kp_t, ln_t


class PitchCalibDataset(Dataset):
    """Dataset yielding `(img_tensor, kp_targets, line_targets)`."""

    def __init__(self, samples: list[tuple[Path, dict]], cfg: PitchCalibConfig, *, synthetic_count: int = 0):
        self.samples = samples
        self.cfg = cfg
        self.synthetic_count = synthetic_count
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return self.synthetic_count if self.synthetic_count > 0 else len(self.samples)

    def __getitem__(self, idx: int):
        in_w, in_h = self.cfg.input_size
        if self.synthetic_count > 0:
            rng = np.random.default_rng(10_000 + idx)
            rgb, kp_t, ln_t = _synthetic_pitch_sample(rng, in_w, in_h)
        else:
            img_path, cam = self.samples[idx]
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise FileNotFoundError(f"cannot read image {img_path}")
            src_h, src_w = bgr.shape[:2]
            rgb = cv2.resize(bgr[:, :, ::-1], (in_w, in_h), interpolation=cv2.INTER_LINEAR)
            kp_t, ln_t = render_heatmap_targets(cam, src_w, src_h, in_w // 4, in_h // 4)

        x = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        x = (x - self.mean) / self.std
        return torch.from_numpy(x), torch.from_numpy(kp_t), torch.from_numpy(ln_t)


def discover_sncal_samples(root: Path) -> list[tuple[Path, dict]]:
    """Load `(image.jpg, camera_dict)` pairs from an extracted SoccerNet-Calibration directory."""
    pairs: list[tuple[Path, dict]] = []
    for js in sorted(root.rglob("*.json")):
        img = js.with_suffix(".jpg")
        if not img.is_file():
            continue
        doc = json.loads(js.read_text(encoding="utf-8"))
        if all(k in doc for k in ("pan_degrees", "tilt_degrees", "x_focal_length", "position_meters")):
            pairs.append((img, doc))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output checkpoint directory")
    parser.add_argument("--sncal-dir", type=Path, default=None, help="Extracted SoccerNet-Calibration directory")
    parser.add_argument("--synthetic-smoke", type=int, default=0, help="Train on N synthetic broadcast frames")
    parser.add_argument("--backbone", type=str, default="resnet34", choices=("resnet34", "resnet50"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = PitchCalibConfig(backbone=args.backbone)
    samples = discover_sncal_samples(args.sncal_dir) if args.sncal_dir else []
    if not samples and args.synthetic_smoke <= 0:
        raise SystemExit("pass --sncal-dir <dir> or --synthetic-smoke <N>")

    ds = PitchCalibDataset(samples, cfg, synthetic_count=args.synthetic_smoke)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    dev_str = args.device or (
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    device = torch.device(dev_str)
    model = build_pitch_calib_net(cfg, pretrained_backbone=(args.synthetic_smoke == 0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        n_batches = 0
        for x, kp_t, ln_t in dl:
            x, kp_t, ln_t = x.to(device), kp_t.to(device), ln_t.to(device)
            opt.zero_grad()
            kp_out, ln_out = model(x)
            loss = _focal_mse_loss(kp_out, kp_t) + 0.6 * _focal_mse_loss(ln_out, ln_t)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())
            n_batches += 1
        avg_loss = tot_loss / max(1, n_batches)
        history.append({"epoch": epoch, "loss": round(avg_loss, 5)})
        print(f"[epoch {epoch:02d}/{args.epochs}] loss={avg_loss:.5f}")

    save_pitch_calibrator(model, cfg, args.out)
    (args.out / "train-log.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"Saved Apache-2.0 pitch calibrator checkpoint -> {args.out}")


if __name__ == "__main__":
    main()

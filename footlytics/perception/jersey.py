"""Track-level jersey number recognition from sharp torso crops."""
from __future__ import annotations

import re
from collections.abc import Sequence

import cv2
import numpy as np
import pandas as pd

TORSO_TOP = 0.12
TORSO_BOTTOM = 0.58
TORSO_LEFT = 0.15
TORSO_RIGHT = 0.85

_STRIP_CHARS = " \t\n\r.,:;!?'\"-_/#()[]"
_DIGITS_RE = re.compile(r"[0-9]{1,2}")


def parse_jersey_number(text: str) -> int | None:
    """Parse an OCR string into a valid shirt number (1..99), or None."""
    cleaned = str(text).strip(_STRIP_CHARS)
    if not _DIGITS_RE.fullmatch(cleaned):
        return None
    num = int(cleaned)
    return num if 1 <= num <= 99 else None


def crop_sharpness(crop_rgb: np.ndarray) -> float:
    """Laplacian variance of a crop as a focus/sharpness proxy."""
    if crop_rgb.size == 0 or min(crop_rgb.shape[:2]) < 3:
        return 0.0
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_jersey_crop(
    frame_rgb: np.ndarray,
    bbox: Sequence[float],
    min_box_h: int = 36,
    min_crop_px: int = 12,
) -> np.ndarray | None:
    """Extract the torso crop for a player box (x, y, w, h) in pixels, or None."""
    x, y, w, h = map(float, bbox)
    if h < min_box_h or w <= 0:
        return None
    H, W = frame_rgb.shape[:2]
    x0 = max(round(x + w * TORSO_LEFT), 0)
    x1 = min(round(x + w * TORSO_RIGHT), W)
    y0 = max(round(y + h * TORSO_TOP), 0)
    y1 = min(round(y + h * TORSO_BOTTOM), H)
    if (x1 - x0) < min_crop_px or (y1 - y0) < min_crop_px:
        return None
    return np.ascontiguousarray(frame_rgb[y0:y1, x0:x1])


def select_track_samples(
    samples: Sequence[tuple[int, int, np.ndarray, float]],
    max_per_track: int = 12,
    min_frame_gap: int = 3,
) -> list[tuple[int, np.ndarray]]:
    """Pick up to `max_per_track` spaced, highest-quality crops per track.

    Each input entry is `(track_id, frame_idx, crop_rgb, quality)`.
    Returns `(track_id, crop_rgb)` pairs.
    """
    by_track: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for tid, frame_idx, crop_rgb, quality in samples:
        by_track.setdefault(int(tid), []).append((int(frame_idx), crop_rgb, float(quality)))

    out: list[tuple[int, np.ndarray]] = []
    for tid, items in by_track.items():
        items.sort(key=lambda item: (-item[2], item[0]))
        picked_frames: list[int] = []
        for frame_idx, crop_rgb, _ in items:
            if len(picked_frames) >= max_per_track:
                break
            if any(abs(frame_idx - pf) < min_frame_gap for pf in picked_frames):
                continue
            picked_frames.append(frame_idx)
            out.append((tid, crop_rgb))
    return out


class TrOCRJerseyReader:
    """Reads shirt numbers from RGB torso crops using a pretrained TrOCR model."""

    def __init__(
        self,
        model_name: str = "microsoft/trocr-small-str",
        device: str = "cuda",
        batch_size: int = 32,
        min_conf: float = 0.4,
    ):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.min_conf = min_conf
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device).eval()

    def read_batch(self, crops: Sequence[np.ndarray]) -> list[tuple[int | None, float]]:
        """Run batched TrOCR inference on RGB crops, returning `(number, confidence)` pairs."""
        if not crops:
            return []
        import torch
        from PIL import Image

        special_ids = set(getattr(self.processor.tokenizer, "all_special_ids", []) or [])
        results: list[tuple[int | None, float]] = []
        step = max(int(self.batch_size), 1)
        use_autocast = self.device.startswith("cuda")

        for i in range(0, len(crops), step):
            chunk = crops[i:i + step]
            imgs = [Image.fromarray(np.ascontiguousarray(c)) for c in chunk]
            pixel_values = self.processor(
                images=imgs, return_tensors="pt"
            ).pixel_values.to(self.device)
            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast),
            ):
                outputs = self.model.generate(
                    pixel_values,
                    max_new_tokens=4,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            scores = outputs.scores or ()
            n_steps = len(scores)
            step_probs = [torch.softmax(s.float(), dim=-1).max(dim=-1).values for s in scores]
            gen_ids = outputs.sequences[:, -n_steps:] if n_steps else None
            texts = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)

            for bi, text in enumerate(texts):
                if n_steps:
                    all_p = [float(step_probs[t][bi].item()) for t in range(n_steps)]
                    non_special = [
                        all_p[t] for t in range(n_steps)
                        if int(gen_ids[bi, t].item()) not in special_ids
                    ]
                    used = non_special if non_special else all_p
                    conf = float(np.clip(np.mean(used), 0.0, 1.0))
                else:
                    conf = 0.0
                num = parse_jersey_number(text)
                if num is not None and conf >= self.min_conf:
                    results.append((num, conf))
                else:
                    results.append((None, 0.0))

        return results


def read_track_jerseys(
    samples: Sequence[tuple[int, int, np.ndarray, float]],
    reader,
    max_per_track: int = 12,
    min_frame_gap: int = 3,
) -> pd.DataFrame:
    """Select top crops per track, run `reader.read_batch`, and return a reads DataFrame."""
    selected = select_track_samples(
        samples,
        max_per_track=max_per_track,
        min_frame_gap=min_frame_gap,
    )
    if not selected:
        return pd.DataFrame(columns=["track_id", "number", "confidence"])

    preds = reader.read_batch([crop for _, crop in selected])
    rows = [
        {"track_id": tid, "number": num, "confidence": conf}
        for (tid, _), (num, conf) in zip(selected, preds)
        if num is not None
    ]
    return pd.DataFrame(rows, columns=["track_id", "number", "confidence"])

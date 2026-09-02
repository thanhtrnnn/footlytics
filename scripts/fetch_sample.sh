#!/usr/bin/env bash
# Download one public broadcast-style football clip and trim to 30 s and 3 min.
# Usage: bash scripts/fetch_sample.sh [URL]
# Default source: roboflow sports example video (Google Drive). Requires gdown or yt-dlp for other URLs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/clips
SRC="${1:-https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4tSqaVZ}"
RAW=data/clips/sample_raw.mp4
if [ ! -f "$RAW" ]; then
  case "$SRC" in
    *drive.google.com*) uv run --with gdown gdown "$SRC" -O "$RAW" ;;
    *youtube.com*|*youtu.be*) uv run --with yt-dlp yt-dlp -f "bv*[height<=720]+ba/b[height<=720]" -o "$RAW" "$SRC" ;;
    *) curl -L "$SRC" -o "$RAW" ;;
  esac
fi
ffmpeg -y -loglevel error -i "$RAW" -t 30  -vf "scale=-2:720" -r 25 -an data/clips/sample_30s.mp4
ffmpeg -y -loglevel error -i "$RAW" -t 180 -vf "scale=-2:720" -r 25 -an data/clips/sample_3min.mp4
ls -lh data/clips

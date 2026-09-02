#!/usr/bin/env bash
# Download one public broadcast-style football clip and trim to 30 s and 3 min.
# Usage: bash scripts/fetch_sample.sh [URL]
# Default source: public tactical-cam clip on YouTube (research smoke test only). Google Drive and direct URLs also accepted.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/clips
SRC="${1:-https://www.youtube.com/watch?v=YwujR_aUmQ8}"  # Brazil vs France tactical cam; section 20:00-23:00
RAW=data/clips/sample_raw.mp4
if [ ! -f "$RAW" ]; then
  case "$SRC" in
    *drive.google.com*) uv run --with gdown gdown "$SRC" -O "$RAW" ;;
    *youtube.com*|*youtu.be*) uv run --with yt-dlp yt-dlp -f "bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]" --download-sections "*20:00-23:00" --force-keyframes-at-cuts -o "$RAW" "$SRC" ;;
    *) curl -L "$SRC" -o "$RAW" ;;
  esac
fi
ffmpeg -y -loglevel error -i "$RAW" -ss 30 -t 30 -vf "scale=-2:720" -r 25 -an data/clips/sample_30s.mp4
ffmpeg -y -loglevel error -i "$RAW" -t 180 -vf "scale=-2:720" -r 25 -an data/clips/sample_3min.mp4
ls -lh data/clips

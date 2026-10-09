# Real-video source audit — 08/10/2026

User requested the real demo from scalliontor/Footlytics.

Upstream main at audit: `98998e42ab195e17684ba4a7e6123d73a70939be`.
The recursive tree has no MP4/WebM/MOV/GIF/MKV assets. Only main branch; releases and issues empty. `.gitignore` excludes `/data/`, `*.mp4`, `*.mkv`, `*.parquet`. The Colab notebook `scoccer.ipynb` has no saved execution outputs; its render cell creates `/content/radar.mp4`, while MatchState saves under Drive. This is an output path, not an uploaded demo URL.

## Local source located
`data/raw/alfheim/alfheim_2013-11-03_panorama.mp4`, already downloaded before this task, matches the dataset/date/view in upstream `scripts/fetch_alfheim.py`. Local fetch log identifies Tromsø–Strømsgodset, first half, panorama starting `2013-11-03 18:01:12.794293`. The file is H.264, 4450×2000, about100seconds. The log's planned2minutes is not the measured playable duration.

A30-second excerpt is prepared for private review at the task's outputs directory, scaled to1600px with H.264/faststart. It shows actual source footage. It is NOT a radar/tracking output, not paired with a newly produced MatchState, and not the Brazil–France F2 benchmark.

## Publication boundary
The authoritative provider currently says Alfheim may only be used for non-commercial research and restricts re-identification/performance profiles. Source: https://datasets.simula.no/alfheim/ . Upstream's description of open access does not remove this condition. The clip is kept for local review and NOT added to the public startup landing.

Requested missing information: path/link for a real exported video/radar from the repo owner's Colab/Drive, with a source appropriate for public presentation. No replacement of the live synthetic illustration or Cloudflare deployment occurred in this task.

## Cập nhật commit mới
Đã kiểm tra upstream main: `c787ff6d8bbd317f2d14761507dc8a2650556144`, commit “Publish annotated GT showcase and pipeline video demos”, lúc23:28:47 ngày08/10/2026 (UTC+7). Đây là commit nối tiếp snapshot98998e42 đã kiểm ở lượt trước. Hiện docs/demos có4MP4,4poster, README attribution và manifest. Hai clipGT dùng annotation của SoccerTrack v2; hai clip10giây là dự đoán detector/tracker. Chưa đổi landing hoặc deploy trong lượt kiểm tra commit này.

## Public showcase release

The next user request authorizes publication and redesign. All four original exports/posters are now copied unchanged into web/assets/demos. A ten-second hero reel crops the physical-footage region (1600×506) from the GT export, re-encodes H.264 CRF24/faststart without audio, and keeps an explicit GT label. No private Alfheim footage is shipped.

Footage and annotations: SoccerTrack v2 / AtomScott and contributors, match 117092, CC BY 4.0. Primary dataset card checked again: https://huggingface.co/datasets/atomscott/soccertrack-v2 . Source release: https://github.com/scalliontor/Footlytics/blob/c787ff6d8bbd317f2d14761507dc8a2650556144/docs/demos/README.md . Attribution, changes and license appear on the page. File sidecars preserve URLs, commit, hashes, sizes and origin.

GT exports do not measure automatic recognition. Predictions remain unscored, with misses, false positives and fragmented IDs. Both confidence and pitch margin vary (0.25/2m vs 0.10/6m), so they are not a controlled one-variable experiment. Legacy radar's 105×76m assumption differs from the dataset card's 105×68m; no calibrated metric, speed, formation or identity accuracy claims arise from the videos. Historical Brazil–France F2 figures retain separate source/date/configuration.

# FOOTLYTICS — Kế hoạch kỹ thuật V0 (Game-State Engine)

Phiên bản 0.1, 2026-09-03.

## 1. Mục tiêu

Pipeline end-to-end chạy local: video trận đấu -> bảng game state (22 cầu thủ + bóng, tọa độ mét trên sân 105x68) -> overlay video, sân 2D (radar), báo cáo chất lượng `quality.json`. Số đo từ pipeline trả lời blocker B8-B12 trong `docs/feasibility.md`.

## 2. Môi trường và ràng buộc

- Apple M2, 24 GB RAM, torch MPS (không CUDA). Clip 30 s đến 3 phút, 720p, 25 fps. Disk trống ~20 GB.
- uv, Python 3.11 (3.14 chưa có wheel cho ultralytics/opencv). ffmpeg 9.
- Cloud GPU (Colab/RunPod T4/A10) cho fine-tune, train pitch-keypoint, full match, so sánh SoccerMaster và sn-gamestate.

## 3. Kiến trúc pipeline

| Module | Nhiệm vụ | Công nghệ bản đầu |
|---|---|---|
| `ingest.py` | đọc video, resize 720p, 25 fps | ffmpeg / cv2 |
| `detect.py` | phát hiện người và bóng | YOLO11 (COCO: person, sports ball), sau đó fine-tune player/gk/ref/ball |
| `track.py` | gán id xuyên frame | ultralytics ByteTrack / BoT-SORT |
| `team.py` | phân loại đội | HSV histogram của crop áo -> KMeans k=2; gk/ref là outlier hoặc class riêng |
| `pitch.py` | keypoint sân | YOLO-pose trên football-field-detection (32 keypoints) hoặc PnLCalib |
| `homography.py` | ánh xạ ảnh -> sân | cv2.findHomography RANSAC, EMA theo thời gian, loại frame lỗi lớn |
| `ball.py` | quỹ đạo bóng | detections -> tọa độ sân -> nội suy tuyến tính khoảng trống ngắn |
| `gamestate.py` | ghép bảng | pandas -> parquet + csv |
| `render.py` | overlay + radar | cv2, matplotlib |
| `quality.py` | báo cáo chất lượng | json + markdown |
| `cli.py` | `footlytics run clip.mp4 --out dir` | typer |

## 4. Output contract

Cột: `timestamp_s, frame, player_id, team (0/1/gk/ref), role, x_m, y_m, ball_x_m, ball_y_m, det_conf, calib_ok, interpolated`. Gốc tọa độ góc dưới trái, sân 105x68 m.

## 5. Lộ trình

| Stage | Nội dung | Kết quả |
|---|---|---|
| 0 | COCO YOLO11 + ByteTrack + HSV team, tọa độ ảnh | pipeline chạy end-to-end trên clip 30 s |
| 1 | pitch keypoints + homography | tọa độ mét, radar 2D |
| 2 | detector fine-tune (player/gk/ref/ball), nội suy bóng | ball coverage tăng, role đúng |
| 3 | cloud: sn-gamestate, SoccerMaster trên cùng clip | quyết định base stack |

Cổng sau Stage 0: nếu trung bình < 15/22 cầu thủ được track, làm Stage 2 trước Stage 1.

## 6. Báo cáo chất lượng (`quality.json`)

frames processed, mean detections/frame so với 22, % frame có >= 20 cầu thủ, ID switch proxy (track sinh mới sau frame 1), out-of-frame %, calibration success rate, median reprojection error, ball coverage %, giây xử lý / phút trận (B12).

## 7. Kiểm thử

Test trước, code sau: `test_homography.py` (góc sân -> 0,0 và 105,68), `test_team.py` (hai màu áo -> hai cụm), `test_gamestate.py` (schema, không NaN cột bắt buộc, timestamp tăng), `test_ball.py` (nội suy khoảng trống <= N frame, để NaN khoảng dài). Smoke: `footlytics run data/clips/sample_30s.mp4`.

## 8. Rủi ro

Disk 20 GB; Python 3.14; chưa có weights pitch keypoint (fallback: train Colab ~1 h); ultralytics AGPL phải license hoặc thay trước khi bán; góc camera clip công khai khác V.League nên kết luận B8 tạm thời.

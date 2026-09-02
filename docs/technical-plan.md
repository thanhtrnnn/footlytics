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
| 1 | calibration: landmark JSON (frame 0) + camera motion KLT/RANSAC; pitch keypoint model chỉ là tuỳ chọn | tọa độ mét, radar 2D |
| 2 | detector fine-tune (player/gk/ref/ball), nội suy bóng | ball coverage tăng, role đúng |
| 3 | cloud: sn-gamestate, SoccerMaster trên cùng clip | quyết định base stack |

Cổng sau Stage 0: nếu trung bình < 15/22 cầu thủ được track, làm Stage 2 trước Stage 1.

## 6. Báo cáo chất lượng (`quality.json`)

frames processed, mean detections/frame so với 22, % frame có >= 20 cầu thủ, ID switch proxy (track sinh mới sau frame 1), out-of-frame %, calibration success rate, median reprojection error, ball coverage %, giây xử lý / phút trận (B12).

## 7. Kiểm thử

Test trước, code sau: `test_homography.py` (góc sân -> 0,0 và 105,68), `test_team.py` (hai màu áo -> hai cụm), `test_gamestate.py` (schema, không NaN cột bắt buộc, timestamp tăng), `test_ball.py` (nội suy khoảng trống <= N frame, để NaN khoảng dài). Smoke: `footlytics run data/clips/sample_30s.mp4`.

## 8. Rủi ro

Disk 20 GB; Python 3.14; chưa có weights pitch keypoint (fallback: train Colab ~1 h); ultralytics AGPL phải license hoặc thay trước khi bán; góc camera clip công khai khác V.League nên kết luận B8 tạm thời.

## 9. Cập nhật Stage 1 (2026-09-03)

- **Model pitch keypoint của Roboflow (YOLOv8x-pose, 32 keypoints) thất bại trên tactical cam góc rộng:** 8-10 keypoint "tự tin" nhưng sai vị trí (vẽ vòng cấm vào giữa sân). Model huấn luyện trên broadcast, out-of-distribution với góc rộng 720p. Giữ `pitch.py` làm tuỳ chọn cho broadcast; sẽ kiểm chứng lại khi có clip broadcast.
- **Giải pháp Stage 1 đã chạy:** `calibration.py` đọc JSON landmark (id vertex 1-32 hoặc tên) cho frame 0, khớp homography RANSAC; `camera_motion.py` theo dõi chuyển động camera (KLT trên nền tĩnh, mask cầu thủ, RANSAC homography từng frame) và lan truyền H_t = H_0 · inv(H_0→t). Camera tactical cam **không** tĩnh: lia và zoom rõ trong 30 s, median frame bị nhoè.
- **Kết quả clip 30 s:** 6 landmark, sai số trung vị 0.064 m; 100% hàng game state nằm trong sân; overlay đường kẻ sân khớp ở frame 0, 375, 749 (lệch vài px ở vòng cấm xa cuối clip). Radar 2D hợp lý. Chi phí thêm khoảng 20 s/phút trận (tổng 67 s/phút).
- **Cách lấy landmark:** phát hiện đường kẻ trắng (mask cỏ, mask cầu thủ, HoughLinesP, gộp đường, giao điểm) rồi gán nhãn tay 4 điểm chắc chắn; tự kiểm chứng bằng giao điểm vòng tròn giữa sân với đường giữa sân (dự đoán trùng vạch trắng). Script trong lịch sử phiên, cần đóng gói thành `footlytics calib-assist`.
- **Nợ kỹ thuật:** (1) tự động hoá calibration frame 0 (PnLCalib, hoặc ghép line-snap vào keypoint model), (2) chống trôi KLT trên clip dài: neo lại định kỳ bằng đường kẻ sân, (3) lọc người ngoài sân bằng tọa độ mét.

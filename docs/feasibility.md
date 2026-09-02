# FOOTLYTICS — Đánh giá khả thi kỹ thuật (Technical Feasibility)

Phiên bản 0.1, 2026-09-03. Trả lời blocker kỹ thuật B8-B12 bằng thí nghiệm đo được. Cột "Kết quả" cập nhật sau mỗi milestone.

## 1. Các stack ứng viên

| Stack | Bao phủ | License | Phần cứng | Độ chín | Phù hợp broadcast VN | Chi phí tích hợp |
|---|---|---|---|---|---|---|
| **SoccerMaster** (CVPR 2026 Oral) | detection, tracking, calibration, jersey OCR, role, team, game state reconstruction (GSR) | chưa rõ, kiểm tra repo | CUDA 12.1, SigLIP2-large, dataset Soccer Factory ~130 GB | mới, nghiên cứu | chưa kiểm chứng | cao (cloud GPU, pipeline nặng) |
| **sn-gamestate / TrackLab** (SoccerNet GSR baseline) | YOLO11 + StrongSORT + PRTReid + TVCalib/PnLCalib + MMOCR | GPL-3.0 | GPU, Python 3.9 | ổn định, có metric GS-HOTA | broadcast single-cam, đúng bài toán | trung bình, khó thương mại hóa vì GPL |
| **roboflow/sports + ultralytics** | player/ball/pitch-keypoint datasets, radar 2D, team clustering | MIT (sports), AGPL-3.0 (ultralytics) | chạy được MPS trên M2 | cộng đồng lớn | tốt cho prototype | thấp |
| **PnLCalib / TVCalib** (calibration riêng) | camera calibration từ đường kẻ sân | xem repo | GPU khuyến nghị | nghiên cứu | broadcast | trung bình |

## 2. Rủi ro license

- `ultralytics` AGPL-3.0: dùng cho nghiên cứu được; sản phẩm SaaS thương mại cần Ultralytics Enterprise License hoặc thay bằng detector Apache-2.0 (RT-DETR qua HF transformers, YOLOX, RF-DETR) trước khi bán.
- `sn-gamestate` GPL-3.0: chỉ dùng làm benchmark/so sánh, không nhúng vào sản phẩm.
- Dữ liệu SoccerNet cần NDA; Soccer Factory chỉ dùng trên cloud.

## 3. Bảng blocker kỹ thuật B8-B12

| Mã | Câu hỏi | Thí nghiệm | Chỉ số | Ngưỡng đạt | Kết quả | Kết luận |
|---|---|---|---|---|---|---|
| B8 | Một camera broadcast có đủ tái dựng game state liên tục? | Chạy Stage 0-2 trên clip broadcast 3 phút | % frame open-play có >= 18/22 cầu thủ được track; out-of-frame % | >= 70% frame; out-of-frame <= 25% | Stage 0, tactical cam 3 phút: 66.7% frame >= 18 người (gồm trọng tài), trung bình 16.6/frame; clip 30 s: 100% frame >= 18, trung bình 20.7 | Sát ngưỡng. Tactical cam đủ; broadcast chưa đo. Cần lọc người ngoài sân (Stage 1) và detector chuyên football (Stage 2) |
| B9 | Sai số nào làm analyst mất tin? | So vị trí với điểm mốc sân đã biết; đếm ID switch | sai số vị trí (m); ID switch / cầu thủ / phút | <= 2 m; <= 1 switch/cầu thủ/phút | Sai số vị trí: chưa đo (cần Stage 1). ID switch proxy: 30 s = 3.5, 3 phút = 4.3 switch/cầu thủ/phút (ByteTrack tuỳ chỉnh; mặc định ultralytics cho 9.1) | Chưa đạt. Cần ReID hoặc appearance tracker (BoT-SORT + ReID) ở Stage 2 |
| B10 | Use case nào bắt buộc cần ball tracking? | Đo ball coverage %; liệt kê metric tính được không cần bóng | ball coverage % | shape/line height/compactness không cần bóng; pressing/transition cần bóng >= 60% coverage | Ball coverage COCO yolo11n: 30 s = 0.1%, 3 phút = 30.5% (kèm nội suy <= 5 frame) | Chưa đạt cho pressing/transition. Cần detector bóng riêng (Stage 2) |
| B11 | MVP cần danh tính cầu thủ hay chỉ team + id tạm? | Tính toàn bộ metric L2 chỉ với team + track id | số metric tính được / tổng | 100% metric L2 không cần tên | Pipeline Stage 0 gán team 0/1/ref chỉ bằng màu áo; không cần tên | Giữ giả thuyết: đủ cho L2, jersey OCR hoãn |
| B12 | Độ trễ chấp nhận: 1h / 3h / overnight? | Đo giây xử lý / phút trận trên M2 và trên T4/A10 | s per match-minute; ngoại suy 90 phút | overnight chắc chắn đạt; mục tiêu <= 3h/trận trên 1 GPU | M2 (MPS, yolo11n, imgsz 1280): 45-48 s / phút trận, tức khoảng 72 phút cho 90 phút. yolo11s: 92 s/phút | Đạt overnight và mục tiêu 3h ngay trên laptop; GPU cloud sẽ nhanh hơn nhiều |

### 3.1. Số đo Stage 0 (2026-09-03, clip Brazil vs France tactical cam, 720p, 25 fps, yolo11n COCO, ByteTrack tuỳ chỉnh)

| Clip | frames | người/frame | % frame >= 18 | % frame >= 20 | track ids | births | switch/cầu thủ/phút | ball coverage | s / phút trận |
|---|---|---|---|---|---|---|---|---|---|
| 30 s | 750 | 20.7 | 100% | 93% | 47 | 39 | 3.5 | 0.1% | 45 |
| 3 phút | 4466 | 16.6 | 67% | 52% | 297 | 285 | 4.3 | 30.5% | 48 |

Ghi chú: "người" gồm cả trọng tài và người ngoài sân được YOLO phát hiện (chưa có calibration để lọc). Đội được gán bằng cụm màu áo (2 đội + trọng tài); kiểm tra bằng mắt trên overlay: đúng phần lớn, vài lỗi ở cầu thủ nhỏ hoặc bị che.

### 3.2. Số đo Stage 1 (2026-09-03, clip 30 s, calibration tay 6 landmark + camera motion)

| Chỉ số | Giá trị |
|---|---|
| Sai số trung vị landmark | 0.064 m |
| Hàng game state trong sân | 100% |
| Trôi calibration sau 30 s (mắt) | vài px ở vòng cấm xa, không đo được số |
| Feature KLT theo dõi mỗi frame | trung bình 599 |
| Chi phí | 67 s / phút trận (thêm ~20 s so với Stage 0) |
| Model pitch keypoint Roboflow trên tactical cam | thất bại (keypoint sai vị trí), không dùng |

Detector chuyên football (Roboflow, HF mirror martinjolif) lấy mẫu 30 frame clip 3 phút: model player/gk/ref phát hiện trung bình 13.7 player so với 19.9 person của COCO yolo11n (kém trên góc rộng); model bóng riêng ở imgsz 1920 thấy bóng 27/30 frame (1.23 box/frame, có dương tính giả), ở 1280 chỉ 13/30. Kết luận Stage 2: giữ COCO cho người, thêm model bóng riêng ở 1920 cho bóng.

Cập nhật B9: sai số vị trí tại landmark 0.06 m; sai số thực tế cầu thủ phụ thuộc điểm chân bbox và trôi camera, chưa đo bằng ground truth.

## 4. Kế hoạch compute

- Local Apple M2 (MPS, 24 GB RAM, ~20 GB disk trống): phát triển, test, clip 30 s đến 3 phút.
- Colab / RunPod T4 hoặc A10: fine-tune detector, train pitch-keypoint model, chạy full match, chạy sn-gamestate và SoccerMaster để so sánh.
- Ước tính cost per processed match: đo B12 rồi nhân giá GPU/giờ (T4 ~0.2-0.5 USD/h, A10 ~0.7-1 USD/h). Điền sau M3.

## 5. Dữ liệu

- Roboflow Universe: football-players-detection, football-ball-detection, football-field-detection (keypoints). Dùng để fine-tune.
- SoccerNet GSR: 200 clip, cần đăng ký NDA. Dùng làm benchmark GS-HOTA.
- Soccer Factory (SoccerMaster): 7000 video, 130 GB, cloud only.
- Footage VN: phụ thuộc B5-B7 (quyền upload/lưu/xử lý, quyền dùng làm training data, nguồn broadcast hay tactical cam). Câu hỏi cho phỏng vấn Hà Nội FC, nằm ngoài phạm vi kỹ thuật.

## 6. Khuyến nghị

1. Stage 0-2 chạy local ngay (ultralytics + roboflow sports) để có pipeline end-to-end và số đo B8-B12.
2. Stage 3 trên cloud: so sánh cùng clip với sn-gamestate và SoccerMaster bằng GS-HOTA-like metric.
3. Cổng quyết định (decision gate) sau Stage 3: chọn base stack theo độ chính xác, chi phí/trận, license.
4. Sau M1: nếu trung bình < 15/22 cầu thủ được track trên clip broadcast, ưu tiên fine-tune detector (Stage 2) trước calibration (Stage 1).

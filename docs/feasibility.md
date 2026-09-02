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
| B8 | Một camera broadcast có đủ tái dựng game state liên tục? | Chạy Stage 0-2 trên clip broadcast 3 phút | % frame open-play có >= 18/22 cầu thủ được track; out-of-frame % | >= 70% frame; out-of-frame <= 25% | _chưa đo_ | |
| B9 | Sai số nào làm analyst mất tin? | So vị trí với điểm mốc sân đã biết; đếm ID switch | sai số vị trí (m); ID switch / cầu thủ / phút | <= 2 m; <= 1 switch/cầu thủ/phút | _chưa đo_ | |
| B10 | Use case nào bắt buộc cần ball tracking? | Đo ball coverage %; liệt kê metric tính được không cần bóng | ball coverage % | shape/line height/compactness không cần bóng; pressing/transition cần bóng >= 60% coverage | _chưa đo_ | |
| B11 | MVP cần danh tính cầu thủ hay chỉ team + id tạm? | Tính toàn bộ metric L2 chỉ với team + track id | số metric tính được / tổng | 100% metric L2 không cần tên | _chưa đo_ | Giả thuyết: đủ, jersey OCR hoãn |
| B12 | Độ trễ chấp nhận: 1h / 3h / overnight? | Đo giây xử lý / phút trận trên M2 và trên T4/A10 | s per match-minute; ngoại suy 90 phút | overnight chắc chắn đạt; mục tiêu <= 3h/trận trên 1 GPU | _chưa đo_ | |

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

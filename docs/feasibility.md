# FOOTLYTICS — Đánh giá khả thi kỹ thuật (hợp nhất)

Phiên bản 1.1, 2026-09-23 (1.0: 2026-09-10). 1.1 thêm mục 3.4: ba lỗi pipeline sau hợp nhất và số đo sau khi sửa. Gộp số đo của hai repo. Hai bối cảnh đo **không so sánh trực tiếp được**: upstream đo camera panorama cố định 3840x1504 (SoccerTrack v2) trên ground truth từng frame; V0 đo tactical cam 720p lia/zoom, không có ground truth.

## 1. Stack đã chọn

| Thành phần | Chọn | License | Ghi chú |
|---|---|---|---|
| Detector | SoccerMaster `yolo_v8x6_finetuned.pt` (HF `xleprime/SoccerMaster`, 195 MB, không gated), chạy tile 1280 / overlap 0.25 trên panorama, `tile=0` trên 720p | ultralytics AGPL-3.0 | phải mua Enterprise License hoặc đổi detector Apache-2.0 trước khi bán |
| Bóng | + model bóng riêng Roboflow ở 1920 px (mirror HF martinjolif) | MIT dataset, weights roboflow | coverage 30.5% lên 94.9% trên tactical cam (pipeline V0); pipeline Match State: 98.6% frame có calibration sau khi sửa lề ngoài sân (3.4) |
| Tracking | `PitchTracker` upstream: Kalman `[x,y,vx,vy]` mét, gate 12 m/s, appearance veto 0.5 | mã riêng | không dùng ByteTrack |
| Identity | `identity.py` upstream: nối tracklet, phủ quyết số áo trước khi nối, quota 11 người theo thời gian | mã riêng | OCR số áo chưa có model |
| Calibration | DLT + TPS + `verdict()` upstream; camera chuyển động bằng `AnchoredCamera` V0 | mã riêng | model keypoint/line học máy thất bại trên cả hai loại footage ngoài broadcast |
| Dữ liệu | SoccerTrack v2 (gated, CC-BY), Alfheim ZXY (mở), Soccer Factory (fine-tune) | | V0 chỉ có một clip công khai |
| So sánh sau | sn-gamestate (GPL-3.0, chỉ benchmark), SoccerMaster full pipeline (cloud) | | Stage 3 cũ |

## 2. Bảng blocker B8-B12

| Mã | Câu hỏi | Ngưỡng | Số đo upstream (panorama cố định, GT) | Số đo V0 (tactical cam 720p, không GT) | Kết luận |
|---|---|---|---|---|---|
| B8 | Một camera đủ tái dựng game state liên tục? | >= 70% frame có >= 18/22 | recall 74% ở conf 0.10 (21 người/frame), 60% ở 0.25 (14/frame), precision 77% / 94% | 3 phút: 60% frame >= 18 sau lọc ngoài sân, trung bình 15.5; 30 s: 100% >= 18 | Đạt sát ngưỡng ở cả hai; detector là nút thắt, không phải tracker |
| B9 | Sai số nào làm analyst mất tin? | <= 2 m; <= 1 switch/cầu thủ/phút | vị trí trung vị 0.18-0.21 m so GT; ID switch 347/250 frame ở conf 0.10, 179 ở 0.25; nối tracklet: 129 mảnh về 22 người, 0 hàn nhầm | landmark 0.064 m (mục tiêu méo, xem 3.2); switch proxy 2.67/cầu thủ/phút | Vị trí đạt; switch chưa đạt nếu không có identity layer |
| B10 | Use case nào cần ball tracking? | pressing/transition cần coverage >= 60% | ball gate 0.10 + prior một bóng, chưa báo coverage | 94.9% với model riêng 1920 px (V0). Sau hợp nhất: 11.7% vì lọc ngoài sân 2 m xoá bóng; sửa bằng lề riêng 10 m: 99.3% (30 s), 98.6% frame có calibration (3 phút), xem 3.4 | Đạt (sau khi sửa) |
| B11 | Cần danh tính hay chỉ team + id tạm? | 100% metric L2 không cần tên | tactics.py chạy trên track_id + team; số áo chỉ để nối mảnh | team 0/1/ref chỉ bằng màu áo | Giữ giả thuyết; số áo là Phase 1 |
| B12 | Độ trễ 1h / 3h / overnight? | <= 3h/trận trên 1 GPU | ~37 giờ/hiệp cho pass perception trên M2 Pro (panorama 4K, tile x6); A100 chưa đo đủ trận | 203 s/phút trận (người 1280 + bóng 1920 + camera) trên M2, ~5 giờ/90 phút | Overnight đạt; 3h cần GPU và đo lại trên A100 |

### 2.1. Bài học chung của cả hai repo
- Model pitch geometry học máy huấn luyện trên broadcast (SoccerMaster LinesDetection, Roboflow YOLOv8x-pose 32 keypoint) **thất bại tự tin** trên footage ngoài phân phối (panorama fisheye ban đêm, tactical cam góc rộng 720p). Cả hai quay về click landmark tay; upstream chấm điểm bằng `verdict()`, V0 kiểm bằng giao điểm vòng tròn giữa sân.
- Điểm chân bbox, không phải tâm bbox: sai 6.7 m nếu dùng tâm (upstream, camera tổng hợp).
- Kích thước sân là số đo, không phải mặc định: SoccerTrack v2 rộng ~76 m, không phải 68 m; 2.14% vị trí ngoài biên cho đến khi sửa (upstream).
- Quãng đường phải tích phân từ vị trí đã làm mượt: cộng dịch chuyển thô phóng đại 1.9x (upstream, GT thật).

## 3. Số đo chi tiết

### 3.1. Upstream (README của scalliontor/Footlytics)
- Calibration tổng hợp: pinhole chính xác 1e-13 m; 3 px nhiễu click với 8 landmark 0.09 m; 4 landmark chụm 1.04 m so với 35 landmark trải 0.06 m; panorama ghép 1.25 m về 0.22 m với TPS.
- Tracker tổng hợp 22 cầu thủ 1 phút: gần hoàn hảo 103 switch chỉ chuyển động, 10 với appearance; 0.35 m nhiễu + 5% mất 243 về 77; 0.6 m + 25% mất 3561 về 3095 (detector là ràng buộc).
- Identity trên GT thật cắt vụn: 5 cắt/người 129 mảnh về 22 tracklet, 0 hàn nhầm, 22/22; 20 cắt + 5% số áo đọc được: 29 tracklet 2 hàn nhầm 13/22 về 23 tracklet 0 hàn nhầm 21/22. Phủ quyết số áo trước khi nối: hàn nhầm 19 về 1.
- Đội: đỏ-xanh 100%, đỏ-đỏ 68%, trắng-trắng 64% mỗi track; quota theo thời gian giữ 11v11 ở 100% frame so 0% khi chia toàn cục.
- Thật (SoccerTrack v2 117092, 6000 frame GT): tốc độ tối đa trung vị 7.13 m/s, chiếm dụng theo phần ba sân 27/41/33%.

### 3.2. V0 (tactical cam Brazil vs France, 720p 25 fps)
- Stage 0 (yolo11n COCO 1280 + ByteTrack tuỳ chỉnh): 30 s 20.7 người/frame, 100% frame >= 18; 3 phút 16.6, 67%; switch proxy 3.5-4.3/cầu thủ/phút (mặc định ultralytics 9.1; BoT-SORT + ReID 3.7, không cải thiện; imgsz 1920 bắt thêm khán giả).
- Stage 1 (landmark frame 0 + camera motion): 6 landmark, sai số trung vị 0.064 m **nhưng mục tiêu bị méo** vì rescale 120x70 về 105x68 (chấm phạt đền lệch 1.38 m, vòng cấm 1.13 m); phải đo lại bằng mô hình sân upstream. 100% hàng trong sân, overlay khớp đến frame cuối 30 s.
- Stage 2 (AnchoredCamera + bóng riêng + lọc ngoài sân, 3 phút): cắt cảnh frame 3657-4501 (34 s) bị loại đúng, 81.2% frame có calibration, 139 lần neo lại, ball coverage 94.9%, 15.5 người/frame, switch 2.67/cầu thủ/phút, 203 s/phút trận trên M2.
- Model player/gk/ref Roboflow: 13.7 player/frame so 19.9 person COCO trên góc rộng, không dùng. Model pitch keypoint Roboflow: sai vị trí, không dùng.

### 3.3. F1 (sau hợp nhất, 2026-09-10, pipeline upstream + cầu nối camera, YOLOv8x6 `tile=0` 1280 px trên M2)
| Chỉ số | Giá trị |
|---|---|
| `verdict()` landmark clip mẫu theo mô hình sân upstream | **good**: 6 landmark, trung vị 0.24 m, trung bình 0.27 m, max 0.48 m tại `corner_RT` (so 0.064 m của V0 trên mục tiêu méo) |
| Người/frame với YOLOv8x6 `tile=0` trên 720p (so 20.7 yolo11n) | trung vị **21.0** trên clip 30 s (750 frame), 541 box ngoài sân bị bỏ, `validate()` sạch, team split "sound" (consistency 0.97, balance 0.93) |
| `tracklets_after_stitch` | clip 30 s: 71 track thô về **55** tracklet cho ~25 người (ByteTrack tuỳ chỉnh V0: 47 track thô). Tracker mét + appearance HSV chưa thắng ByteTrack trên 720p; crop áo quá nhỏ để appearance có tác dụng. Clip 3 phút (stride 2, 2251 frame): 182 track thô về **154** tracklet (V0 ByteTrack cùng clip: 297 track thô, 155 sau lọc); switch proxy ~2.4/cầu thủ/phút, ngang V0 2.67. Nối tracklet mới giảm 15%, còn xa 22 người: identity là việc của Phase 1 |
| Alfheim 2 phút, camera cố định (panorama 4450x2000, 300 frame, không mask cầu thủ) | 0 frame mất. Chuỗi (t-1)->t trôi 7-25 px giữa các lần neo (trung vị 8.4 px ở góc khung); sau khi theo dõi từ keyframe: trung vị 0.75 px, p95 2.1 px ở góc, tối đa 1.35 px ở giữa khung. Cầu nối suy biến đúng về camera cố định |
| Giây/frame YOLOv8x6 trên M2 | 0.34 s/frame (3.0 fps) ở 1280 px: **506 s/phút trận** ở stride 1, 262 s ở stride 2; gấp 2.5 lần V0 (yolo11n + bóng 1920 + camera: 203 s). Clip 3 phút: 81.25% frame có calibration (cắt cảnh 34 s bị loại đúng như V0), 135 lần neo lại, trung vị 20 người/frame, `validate()` sạch, 100% vị trí trong sân |

### 3.4. F2 (2026-09-23, sửa ba lỗi pipeline sau hợp nhất; YOLOv8x6 `tile=0` 1280 px + bóng riêng 1920 px, stride 1, M2)
Cả ba lỗi có sẵn trong fork, không do việc hợp nhất gây ra; commit `27d938d`.

| Lỗi | Trước khi sửa | Sau khi sửa |
|---|---|---|
| Lọc ngoài sân dùng lề 2 m của cầu thủ cho cả bóng. Bóng cạnh biên dọc xa chiếu ra y = -37.5 trên nửa rộng 34 m (cầu thủ đứng đó cũng chiếu ra cùng y). V0 không lọc bóng | 30 s: bóng 11.7% frame. 12 frame mẫu quanh frame neo: model bóng thấy 12/12, 11 bị xoá | `ball_margin_m` = 10 m cho `Role.BALL`: 30 s **99.3%** frame (745/750); 3 phút **3605/3657 frame có calibration (98.6%)**, 80.1% tổng frame vì cắt cảnh 34 s bị loại; 0 bóng bị xoá. Dùng tâm box thay đáy box cho bóng: đã đo và bỏ, đẩy bóng ra xa hơn (y = -38.4) |
| `split_officials` tính ngưỡng theo số id track. 3 phút có 258 track thô cho ~25 người | 50 mảnh (19% id, dưới ngưỡng 25%) thành "trọng tài": 7.0 trọng tài/frame so 2.7 cầu thủ đội nhà, `validate()`: 3 v 7 | KMeans có trọng số theo số lần xuất hiện, thêm ngưỡng `max_obs_share` 0.2: 3 phút **8.2 v 8.2 cầu thủ/frame, `validate()` sạch**, không tách trọng tài (cụm thứ ba 94 mảnh = 36% id). 30 s: tách 15 mảnh = 9.9% số lần xuất hiện, 2.1 trọng tài/frame |
| `apply_to_state` chỉ ghi `team`, không ghi `role` | trọng tài vẫn là `player`, bị tính vào `MatchState.players` | `role` = `referee` trong parquet; 30 s trung vị 19 cầu thủ + 2 trọng tài/frame (trước: 21 gồm cả trọng tài) |

Các số khác của clip 3 phút giữ nguyên so với trước khi sửa: 81.23% frame có calibration, 139 lần neo lại, trung vị 20 người/frame, 7345 box ngoài sân bị bỏ; switch proxy 2.42/cầu thủ/phút; 261 track thô về 179 tracklet; ~595 s/phút trận có model bóng.

Còn mở: trên clip 3 phút trọng tài chưa được tách, nằm lẫn trong hai đội. Tắt ngưỡng id để thử: cụm 94 mảnh (16.7% số lần xuất hiện) cho 3.3 "trọng tài"/frame, trung vị cầu thủ tụt còn 17 và `validate()` báo thiếu người, tức là lại tách quá tay. Cần tách theo tracklet sau khi nối, hoặc thêm vị trí (trợ lý trọng tài đi dọc biên), thay vì chỉ màu áo.

## 4. Compute và chi phí
M2 cho test và clip ngắn; Colab A100 cho panorama và chấm điểm. Cost per processed match = (giây/phút trận trên A100) x giá GPU/giờ; điền sau F1.

## 5. Dữ liệu và quyền
SoccerTrack v2 cần xin quyền (gated). Alfheim mở, tải bằng `scripts/fetch_alfheim.py`. Footage VN: B5-B7 (quyền upload/lưu/xử lý, quyền dùng training, broadcast hay tactical cam) vẫn là câu hỏi cho phỏng vấn CLB.

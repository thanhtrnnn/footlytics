# FOOTLYTICS — Kế hoạch kỹ thuật V1 (hợp nhất hai repo)

Phiên bản 1.0, 2026-09-10. Thay thế bản V0 (repo `thanhtrnnn/footlytics`, nay là archive).

## 1. Bối cảnh

Hai repo cùng công ty được so sánh ngày 2026-09-10:
- `scalliontor/Footlytics` (Vu Hung Anh): Match State làm hợp đồng dữ liệu, mô hình sân theo Luật (35 landmark), detector YOLOv8x6 fine-tune SoccerNet chạy theo tile, tracking Kalman + Hungarian **trong toạ độ mét** có appearance, nhận dạng tracklet (nối mảnh, phủ quyết số áo, quota 11 người theo thời gian), kinematics và tactics, chấm điểm trên ground truth SoccerTrack v2 và Alfheim ZXY. Chưa có packaging, pytest, CLI, và **không xử lý camera chuyển động** (README ghi rõ là giới hạn).
- `thanhtrnnn/footlytics` (V0, 2026-09-03): pipeline cho tactical cam lia/zoom/cắt cảnh: `AnchoredCamera` (KLT + SIFT neo lại + phát hiện cắt cảnh, fail-closed), pass bóng riêng 1920 px (coverage 30.5% lên 94.9%), trọng tài là cụm màu thứ ba, homography RANSAC, calib-assist, quality.json, 38 pytest, CLI, tài liệu blocker.

Kết luận: bổ sung nhau, không cạnh tranh. Quyết định: fork repo của Vu Hung Anh, nhận kiến trúc Match State, chuyển phần độc đáo của V0 vào dưới dạng PR nhỏ. Mục tiêu hai loại camera: rig cố định toàn sân là sản phẩm, camera chuyển động là cầu nối cho footage CLB đang có.

## 2. Kiến trúc (giữ nguyên của upstream)

```
video -> perception -> MATCH STATE (mét, gốc tâm sân, x về khung thành đội khách) -> analytics -> báo cáo
```

Match State: bảng `TRACK_SCHEMA` 17 cột (`frame_idx, period, timestamp, track_id, role, team, jersey, bbox_*, det_conf, x, y, z, speed, accel`), lưu `tracks.parquet` + `meta.json`, tự kiểm tra bằng `validate()`. Analytics là hàm thuần trên DataFrame, không thấy pixel.

Bố cục sau hợp nhất:

| Gói | Nguồn | Ghi chú |
|---|---|---|
| `state/schema.py` | upstream | hợp đồng dữ liệu |
| `geometry/pitch.py` | upstream | bỏ `pitch.py` V0 (rescale 120x70 làm méo vòng cấm 1.1 m, chấm phạt đền 1.4 m) |
| `geometry/homography.py` | upstream | DLT + TPS + `verdict()`; cân nhắc thêm RANSAC (V0) |
| `geometry/camera_motion.py` | **V0** | `AnchoredCamera` + `MovingCalibration` |
| `geometry/calib_assist.py` | **V0** | giao điểm đường kẻ, đặt tên landmark theo upstream |
| `geometry/annotate.py` | upstream | mở rộng: vẽ sẵn ứng viên từ calib-assist |
| `perception/detect.py` | upstream + V0 | tile YOLOv8x6; tuỳ chọn model bóng riêng 1920 px |
| `perception/track.py`, `identity.py` | upstream | tracking mét, nối tracklet |
| `perception/teams.py` | upstream + V0 | fallback cụm thứ ba = trọng tài khi detector không có lớp referee |
| `analytics/*` | upstream | block, formation, PPDA, press distance, transitions, kinematics |
| `pipeline/radar.py` | upstream + V0 | thêm `camera=`, bỏ frame không calibration, report thêm số |
| `pipeline/quality.py` | **V0** | quality.json/.md |
| `cli.py` | **V0** | `footlytics run`, `calib-assist`, `drift-check` |
| `tests/` | **mới** | pytest: test V0 chuyển sang + bọc harness `scripts/test_*.py` của upstream bằng ngưỡng README |

## 3. Cầu nối camera chuyển động

- `MovingCalibration(anchor)`: `feet_to_pitch(bboxes, H_anchor_to_t)` warp điểm chân về pixel frame neo bằng `inv(H_anchor_to_t)` rồi gọi `anchor.image_to_pitch`, nên TPS vẫn áp dụng. Camera cố định: bước KLT xấp xỉ đơn vị, kết quả trùng đường cố định.
- `pipeline.run(..., camera=AnchoredCamera(frame_neo))`: sau `detector.detect`, mask cầu thủ (nới 6 px), `camera.update`; frame không khớp được neo thì bỏ detection (track coast, `max_age` 30 frame; sau cắt cảnh dài sẽ sinh tracklet mới, nối tracklet và quota đội xử lý). Report thêm `frames_calibrated`, `frames_uncalibrated`, `calib_success_rate`, `reanchors`.
- Ngưỡng giữ nguyên V0: KLT 12 inlier / tỉ lệ 0.5; SIFT nửa độ phân giải, 30 inlier / 0.5; neo lại mỗi 50 frame, thử lại mỗi 10 frame khi mất; đổi cảnh khi độ lệch xám trung bình > 20.
- Landmark clip mẫu (Brazil vs France, 720p) đổi sang tên upstream: `halfway_T`, `halfway_B`, `corner_RT`, `pen_area_RB_front`, `centre_circle_T`, `centre_circle_B` (T = touchline xa, y âm).

## 4. Lộ trình

| Mốc | Nội dung | Kết quả cần có |
|---|---|---|
| F0 | fork, pyproject/uv, pytest bọc harness upstream, weights SoccerMaster | `pytest` xanh |
| F1 | cầu nối camera chuyển động, chạy clip 3 phút qua tracker mét + identity của upstream; hồi quy Alfheim 2 phút có/không camera | `calib_success_rate` ~0.81 trên clip cắt cảnh; Alfheim không đổi |
| F2 | pass bóng riêng, `quality.py`, CLI, calib-assist gắn vào `annotate.py` | `footlytics run` ra `tracks.parquet`, `quality.json`, radar, overlay |
| F3 | docs, Notion, PR sửa lỗi nhỏ upstream | 4 PR mở |
| F4 | theo roadmap upstream: Phase 1 identity (OCR số áo), Phase 2 analytics (sequence detector L4 trên `tactics.py`), Phase 3 90 phút trên GPU thuê | ngoài phạm vi kế hoạch này |

## 5. Compute

M2 (MPS): test, clip 720p ngắn với `tile=0`. Colab A100 (`scoccer.ipynb`): tile x6 trên panorama, chấm SoccerTrack v2 (cần xin quyền dataset), fine-tune.

## 6. Rủi ro

`PitchTracker` cần mét mỗi frame, frame mất calibration làm track chết và identity khởi động lại (đo, không giả định). YOLOv8x6 trên M2 720p có thể chậm, giữ công tắc yolo11n cho local. Hướng T/B của landmark phải kiểm bằng `pitch_to_image` overlay trước khi tin số. Upstream có thể muốn đặt module chỗ khác: PR nhỏ, rebase thường xuyên.

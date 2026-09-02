# FOOTLYTICS — Tầm nhìn Tactical Query MVP

Phiên bản 0.1, 2026-09-03. Tài liệu tầm nhìn, chưa phải spec triển khai. Mục tiêu: từ game state (Game-State Engine) đến truy vấn tình huống bằng ngôn ngữ chiến thuật.

## 1. Mô hình dữ liệu phân lớp

| Lớp | Nội dung | Nguồn |
|---|---|---|
| L0 | Video gốc (broadcast hoặc tactical cam) | CLB |
| L1 | Bảng game state: `timestamp_s, frame, player_id, team, role, x_m, y_m, ball_x_m, ball_y_m, det_conf, calib_ok, interpolated` | Game-State Engine |
| L2 | Đặc trưng đội theo frame: centroid, width, depth, line height, compactness (convex hull), khoảng cách giữa các tuyến, chiều cao khối phòng ngự, đội kiểm soát bóng, vùng bóng (thirds x channels) | tính từ L1 |
| L3 | Phân đoạn pha (phase): in possession, out of possession, transition +, transition -, set piece, dead ball | quy tắc trên L1+L2 |
| L4 | Chuỗi chiến thuật (tactical sequence): `(start_s, end_s, team, label, score, evidence_metrics)` | detector trên L2+L3 |
| L5 | Chỉ mục và truy vấn: DuckDB trên parquet, ranking, export | sản phẩm |

## 2. Định nghĩa metric L2 (bản đầu)

- **Formation:** gom cụm outfield theo tọa độ dọc sân (y) thành 3-4 tuyến bằng KMeans/agglomerative với ràng buộc thứ tự; đếm cầu thủ mỗi tuyến, ví dụ 4-3-3.
- **Line height:** trung bình y của 4 cầu thủ outfield sâu nhất, đo từ khung thành nhà (m).
- **Width:** max(x) - min(x) của outfield (m). **Depth:** max(y) - min(y) (m).
- **Compactness:** diện tích convex hull của 10 outfield (m2), càng nhỏ càng compact.
- **Distance between lines:** khoảng cách trung bình y giữa các tuyến liền kề.
- **Defensive block height:** vị trí trung bình khối khi out of possession, phân loại high/mid/low theo ngưỡng thirds.
- **PPDA proxy:** đường chuyền đối thủ cho phép trên mỗi hành động phòng ngự trong 60% sân đối phương. Cần event data hoặc suy luận từ ball tracking; bản đầu dùng số giây đối thủ giữ bóng trên phần sân họ / số lần áp sát.
- **Press intensity:** số cầu thủ phòng ngự trong bán kính R = 5 m quanh người cầm bóng trong T = 3 s sau turnover.

## 3. Sequence detector L4 (rule-based trước, học máy sau)

| Label | Điều kiện kích hoạt (bản đầu) |
|---|---|
| High press | Out of possession, defensive block high (line height >= 60 m), >= 3 cầu thủ trong 10 m quanh bóng khi bóng ở third phòng ngự đối thủ, kéo dài >= 3 s |
| Low block | Out of possession, line height <= 30 m, compactness <= ngưỡng, kéo dài >= 8 s |
| Build-up vs N-man press | In possession trong third nhà, đối thủ có N cầu thủ trong third đó, chuỗi >= 4 s |
| Transition after midfield loss | Chuyển possession xảy ra ở third giữa, xét 6 s tiếp theo, đo tốc độ lùi của khối và khoảng cách đến bóng |
| Overload flank | >= 4 cầu thủ tấn công trong một channel biên (x ngoài 1/4 sân) vs <= 3 phòng ngự |

Mỗi detector trả `score` (0-1) dựa trên độ vượt ngưỡng, kèm `evidence_metrics` để analyst kiểm tra.

## 4. Bề mặt truy vấn

Luồng: câu truy vấn chiến thuật -> lọc L4 (label, đội, trận, thời gian, ngưỡng) -> xếp hạng theo score -> trả chuỗi tình huống -> hiển thị video clip đồng bộ + sân 2D + bảng metric -> xuất clip/báo cáo.

Ví dụ (từ Notion mục 7): `High Press — 18 chuỗi`, `Triển khai bóng trước đối thủ pressing 4 người — 11 chuỗi`, `Low block — 22 chuỗi`, `Chuyển trạng thái sau khi mất bóng ở giữa sân — 9 chuỗi`.

Lưu trữ: một parquet L1 và một parquet L4 mỗi trận; DuckDB làm index và truy vấn nhiều trận (opposition analysis 3-5 trận).

## 5. Vòng lặp xác thực với analyst

1. Chạy detector trên trận thật, đưa danh sách chuỗi cho analyst.
2. Analyst đánh giá đúng/sai từng chuỗi, ghi lý do.
3. Điều chỉnh ngưỡng theo game model của từng CLB (Notion 5.4: tactical KPIs cấu hình theo CLB).
4. Khi có đủ nhãn, huấn luyện classifier thay quy tắc.

## 6. Ngoài phạm vi V1

Phân tích live/halftime, nhận diện tên cầu thủ (jersey OCR), multi-camera fusion, event data đầy đủ (chuyền, sút).

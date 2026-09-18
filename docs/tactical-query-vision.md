# FOOTLYTICS — Tầm nhìn Tactical Query MVP (trên Match State)

Phiên bản 1.0, 2026-09-10. Quy ước toạ độ theo Match State upstream: mét, gốc tâm sân, **x dọc sân** (âm về khung thành nhà, dương về khung thành khách), **y ngang sân** (âm về touchline T, dương về touchline B). `analytics/tactics.normalise_direction` lật cả hiệp để đội nhà luôn tấn công +x.

## 1. Lớp dữ liệu

| Lớp | Nội dung | Đã có trong upstream |
|---|---|---|
| L0 | Video (broadcast, tactical cam, panorama cố định) | `MatchMeta.source_type` |
| L1 | Match State: `TRACK_SCHEMA` 17 cột, `validate()` | `state/schema.py` |
| L2 | Đặc trưng đội theo frame | `tactics.team_block`: `def_line_x` (trung bình 4 outfield sâu nhất), `fwd_line_x`, `depth_m`, `width_m`, `centroid`, `compactness_m` (khoảng cách trung bình tới centroid), lấy mẫu mỗi 25 frame |
| L2 | Kinematics | `kinematics.add_kinematics` (Hampel + làm mượt 0.8 s, kẹp 12 m/s), `distance_summary` (HSR > 5.5, sprint > 7.0) |
| L3 | Pha bóng | `possession` hiện đọc từ `ballStatus` của SoccerTrack XML; **chưa có** suy ra từ tracking |
| L4 | Chuỗi chiến thuật | `tactics.formation_over_time` (cửa sổ 300 s, 8 mẫu), `ppda` (tham số `press_zone` chưa áp dụng), `press_distance`, `transitions` (giữ >= 1.5 s, cửa sổ 5 s) |
| L5 | Chỉ mục và truy vấn | **chưa có** |

## 2. Còn thiếu so với tầm nhìn V0
- Possession suy ra từ Match State (cầu thủ gần bóng nhất trong bán kính, giữ >= N frame) để `press_distance`, `transitions`, `ppda` chạy không cần event feed.
- Compactness theo convex hull và khoảng cách giữa các tuyến (upstream dùng trung bình tới centroid; giữ cả hai, đo tương quan trên GT).
- Sequence detector rule-based trả `(start_s, end_s, team, label, score, evidence_metrics)`:

| Label | Điều kiện đầu | Dựa trên |
|---|---|---|
| High press | out of possession, `def_line_x` của đội phòng ngự (trong khung tấn công của họ) >= +7.5 m, >= 3 cầu thủ trong 10 m quanh bóng khi bóng ở third phòng ngự đối thủ, >= 3 s | `team_block`, `press_distance` |
| Low block | out of possession, `def_line_x` <= -22.5 m, `compactness_m` dưới ngưỡng, >= 8 s | `team_block` |
| Build-up vs N-man press | in possession ở third nhà, N cầu thủ đối phương trong third đó, >= 4 s | `team_block` + possession |
| Transition after midfield loss | đổi possession ở third giữa, xét 6 s: tốc độ lùi khối và khoảng cách tới bóng | `transitions` |
| Overload flank | >= 4 tấn công trong một kênh biên vs <= 3 phòng ngự | vị trí L1 |

- L5: parquet L1 + parquet L4 mỗi trận, DuckDB index nhiều trận (opposition 3-5 trận); truy vấn theo ngôn ngữ chiến thuật, trả chuỗi có clip video đồng bộ + radar + bảng metric, xuất clip/báo cáo. Ví dụ: `High press — 18 chuỗi`, `Low block — 22 chuỗi`.

## 3. Vòng lặp xác thực với analyst
Chạy detector trên trận thật, analyst chấm đúng/sai từng chuỗi, chỉnh ngưỡng theo game model từng CLB, đủ nhãn thì thay quy tắc bằng classifier. Số chỉ đưa HLV sau khi `validate()` sạch và montage đội đã xem bằng mắt (quy tắc upstream).

## 4. Ngoài phạm vi V1
Live/halftime, số áo (Phase 1 upstream), multi-camera fusion, event data đầy đủ.

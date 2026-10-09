# footlytics — Nguồn nội dung và asset
Ngày: 08/10/2026.
Nguồn canonical đã đọc: https://github.com/thanhtrnnn/footlytics/tree/2f584af8046b0643f4ea7d73093d28a38d90e5a5
Local HEAD 10/09/2026 khác mốc nguồn 23/09/2026; không dùng tài liệu local cũ để đảo quyết định repo chính.

| ID | Nội dung | Nguồn canonical | Điều kiện lên trang |
|---|---|---|---|
| F01 | Hướng phần mềm/thuật toán, Vietnam-first, analyst là người dùng | docs/notion-summary.md | Có cơ sở; không suy khách hàng/đối tác |
| F02 | Phân tích đối thủ là tác vụ khởi đầu | docs/notion-summary.md | Nêu định hướng; chưa khẳng định đã hoàn thiện trải nghiệm |
| F03 | Pipeline Match State, radar, quality report | docs/technical-plan.md 1.1; README.md | Có mã/thử nghiệm; cần chọn output thật để minh hoạ |
| F04 | Tactical Query qua nhiều trận | docs/tactical-query-vision.md | Đang phát triển; mockup cần nhãn minh hoạ, không app/query giả |
| F05 | Camera chuyển động có đoạn mất calibration | docs/feasibility.md mục 3.4 | Phải ghi giới hạn nếu demo footage loại này |
| F06 | Ball presence 745/750 (99,3%) trên clip 30s | docs/feasibility.md mục 3.4, 23/09/2026 | Không chọn headline. Nếu dùng: đúng clip/cấu hình/mẫu số, gọi là hiện diện detection, không accuracy |
| F07 | 98,6% trên frame đã calibration, ~80,1% trên toàn frame | Cùng nguồn, clip 3 phút | Hai mẫu số phải hiện cạnh nhau; không rút thành “bắt bóng 98,6% mọi trận” |
| F08 | Danh tính/trọng tài/compute còn hạn chế | docs/feasibility.md mục 3.4 | Không hứa trọn trận, liên tục 22 cầu thủ hoặc realtime |
| F09 | Hà Nội FC, PVF | docs/notion-summary.md | Là mục tiêu tiếp cận; không đặt logo/khẳng định đối tác |
| F10 | Repo chính thanhtrnnn/footlytics | docs/technical-plan.md 1.1, 23/09/2026 | Nguồn kỹ thuật; không cần đưa owner GitHub lên landing |
| F11 | Pilot, giá, SLA, số khách hàng | Chưa có hợp đồng/bằng chứng đã xác nhận | Chỉ nêu lời mời/hướng thử nghiệm; không hứa giá/thời hạn |
| F12 | Founder, pháp nhân, liên hệ | Chưa xác nhận | Cần thông tin thật; không suy từ tên tác giả Git |

## Link nguồn cố định
- [Ý tưởng](https://github.com/thanhtrnnn/footlytics/blob/2f584af8046b0643f4ea7d73093d28a38d90e5a5/docs/notion-summary.md)
- [Kỹ thuật](https://github.com/thanhtrnnn/footlytics/blob/2f584af8046b0643f4ea7d73093d28a38d90e5a5/docs/technical-plan.md)
- [Khả thi](https://github.com/thanhtrnnn/footlytics/blob/2f584af8046b0643f4ea7d73093d28a38d90e5a5/docs/feasibility.md)
- [Tactical Query](https://github.com/thanhtrnnn/footlytics/blob/2f584af8046b0643f4ea7d73093d28a38d90e5a5/docs/tactical-query-vision.md)

## Manifest asset cần hoàn thành
| Trường | Nội dung cần điền trước công bố |
|---|---|
| Asset | Tên video/poster/radar/quality report thật; không đặt tên file chưa tồn tại như asset đã có |
| Nguồn | Video gốc, tác giả/đơn vị, đoạn bắt đầu-kết thúc và quyền công bố |
| Quyền | Quyền xử lý, hiển thị trên web, chỉnh clip; quyền huấn luyện nếu có phải riêng |
| Lượt chạy | Commit, cấu hình/model, ngày, loại camera, thời lượng/frame |
| Ghép cặp | Timestamp video và radar tương ứng; phần mất dữ liệu được đánh dấu |
| Dữ liệu | quality report gắn với chính clip đó; giới hạn/detection không đổi thành accuracy |
| Nội dung minh hoạ | Thật hay mockup; nếu mockup phải có nhãn ở ngay visual |
| Owner | Người chọn và chịu trách nhiệm duyệt asset |

Trong docs local đã rà chưa tìm thấy asset landing đủ manifest. Không khẳng định dataset mở hoặc video public là được tự do công bố trên landing.
Asset minh hoạ phải cho thấy cơ chế và giới hạn; ảnh tạo bằng AI không thay thế bằng chứng tracking thật.

## Bối cảnh phương án thay thế
Trong hồ sơ startup đã đối chiếu trang chính thức BEPRO (https://bepro.ai/) và Hudl Wyscout (https://www.hudl.com/products/wyscout), ngày 08/10/2026.
Landing đầu không cần bảng “chúng tôi hơn đối thủ”. Muốn so sánh phải kiểm chức năng/giá hiện hành và cùng điều kiện; không lấy ý tưởng query làm bằng chứng độc quyền.

## Những thông tin còn cần
Kênh liên hệ/người nhận pilot; đầu ra được phép công bố; thương hiệu web; danh sách đội ngũ được phép công bố; nơi hosting; nhánh/codebase triển khai.
Những mục này không cản việc soạn copy/prototype có nhãn, nhưng phải rõ trước public release.

## Nguồn cho các góc kỹ thuật đã chọn — 08/10/2026
Chi tiết: [TECHNICAL.md](TECHNICAL.md). Path mã dưới đây tính từ package footlytics/ trong repo local; số đo/roadmap ưu tiên tài liệu canonical cố định ở các link bên trên.

| ID | Nguồn | Điều kiện/giới hạn |
|---|---|---|
| FT01 | state/schema.py; technical-plan §2 | 17 cột/Parquet/meta; validate không tự là kiểm GT |
| FT02 | geometry/pitch.py; geometry/homography.py | 35 landmark, DLT/TPS; cần calibration; residual fit không là accuracy tracking |
| FT03 | geometry/camera_motion.py; pipeline/radar.py | KLT/SIFT/MovingCalibration; mất calibration có phần bỏ detection; fixed-camera cần regression theo phiên bản |
| FT04 | perception/detect.py; feasibility §3.4 | F2 tile=0/1280 + bóng 1920; không gán tiled benchmark cho clip này |
| FT05 | perception/track.py | Kalman/Hungarian trong mét, appearance/gate; không đảm bảo identity xuyên trận |
| FT06 | perception/identity.py; feasibility §3.4 | Jersey reads interface/veto ≠ OCR hoàn thiện; track fragmentation/trọng tài còn hạn chế |
| FT07 | analytics/kinematics.py; analytics/tactics.py; tactical-query-vision §1 | Smoothing/metrics có mã; possession từ event/ballStatus; PPDA/events và press_zone cần đối chiếu bản chạy |
| FT08 | pipeline/quality.py; feasibility §3.4 | Presence/calibration cần mẫu số; F2 23/09 remote, chưa chạy lại local; compute offline |
| FT09 | tactical-query-vision §2–3 | Sequence detector, L5 index/query, DuckDB và analyst loop là lộ trình |
| FT10 | state/schema.py; technical-plan §2; tactical-query-vision | Data contract có mã; adapter/third-party integration là đề xuất nếu chưa có implementation |

**Phiên bản:** local HEAD f9bbcc1ddf4e92596d1adfc6bff5852b04e9cff1; canonical docs 2f584af8046b0643f4ea7d73093d28a38d90e5a5. Số F2 là thông tin từ tài liệu remote, không bằng chứng lượt chạy local.
**Số chọn:** 30 giây 745/750 bóng; 3 phút 3605/3657 frame calibrated có bóng (80.1% toàn clip), calibration 81.23%, 139 re-anchor, 261 raw track → 179 tracklet, ~595 s/phút trận trên M2. Không trộn clip/mẫu số hay đổi thành accuracy.
**Case chọn:** lỗi lề sân bóng F2, presence 11.7% → 99.3% trên clip 30 giây. Before/after chỉ thành demo thật khi có artifact/config tương ứng; tăng presence không chứng minh giảm sai số vị trí.

## Publication note — 08/10/2026
The deployed landing uses only the selected historical facts with the scope and caveats shown beside them. Mechanism diagrams are schematic/synthetic and labeled; no unverified footage, full trace, current deployment configuration or pilot receiver is represented as verified. The ledger below preserves evidence and research gaps; these are not claims of a new benchmark run. See CHECKPOINTS.md and DEPLOY.md for actual release status.


## Real-video showcase release c787ff6
- Ground truth 40s + radar 4min: supplied SoccerTrack v2 annotations, not automatic-model accuracy evidence.
- Prediction 10s exports: conf0.25/margin2m and conf0.10/margin6m; both parameters vary, no controlled confidence-only comparison. Misses, false positives and fragmented IDs remain.
- Legacy radar assumes105×76m, dataset card105×68m; not verified metric calibration/kinematics/identity.
- Hero: labelled10s upper-video crop from GT40s. Four gallery exports remain byte-identical to upstream.
- Source/annotations SoccerTrack v2 / AtomScott and contributors, CC BY4.0. Page discloses attribution, changes and limits.
- F2 Brazil–France historical metrics remain independent and were not re-run. Tactical Query still roadmap; no traction/contact/pricing invented.

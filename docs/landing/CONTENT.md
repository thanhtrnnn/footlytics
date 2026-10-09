# footlytics — Nội dung landing ưu tiên kỹ thuật
08/10/2026 · Bản 0.2 · Chưa xây UI.
Thay outline 0.1 theo yêu cầu người dùng; dùng cùng [TECHNICAL.md](TECHNICAL.md) và [CLAIMS.md](CLAIMS.md). Copy chỉ mô tả prototype/phần đã có nguồn; ghi chú dựng không đưa lên trang.

## 1. Hero — kỹ thuật làm nền cho tác vụ
**Dòng nhận diện:** footlytics · Football video intelligence · Prototype
**Headline:** Từ điểm ảnh đến dữ liệu sân.
**Copy:** Hiệu chuẩn hình học, bù chuyển động camera và tracking trong mét tạo Match State cho phân tích bóng đá. Hướng phát triển tiếp theo: tìm chuỗi chiến thuật trong video, với analyst kiểm chứng đầu ra.
**CTA đọc chính:** Khám phá pipeline
**CTA hợp tác khi có điểm nhận thật:** Trao đổi pilot kỹ thuật
Visual: clip/radar thật đồng bộ timestamp, camera/lượt chạy/ngày có nhãn. Chưa có asset đủ quyền thì preview dùng placeholder rõ, không dựng bằng chứng giả hoặc nút gửi giả.

## 2. Demo — thấy phép chuyển đổi
**Tiêu đề:** Cùng một thời điểm, hai cách đọc trận đấu.
**Copy:** Đối chiếu khung hình gốc với vị trí chiếu lên sân. Đoạn mất calibration được đánh dấu để người đọc thấy phần dữ liệu chưa sử dụng được.
Dựng: video + radar + timeline calibration; poster/caption trên mobile. Bảng dữ liệu xuất từ đúng lượt chạy, không tạo mẫu như dữ liệu thật. Xem biểu đồ chất lượng có thể hiểu được ngay cả khi không play media.

## 3. Match State — hợp đồng dữ liệu
**Tiêu đề:** Một lớp dữ liệu chung cho các bước phân tích.
**Copy:** Match State lưu vị trí theo mét cùng thời gian, track, vai trò, đội và tín hiệu detection. Tách perception khỏi analytics giúp thay đổi mô hình nhận diện mà giữ cùng hợp đồng đầu ra.
Dựng: pipeline video → perception/geometry → tracking → Match State → analytics/report.
Disclosure schema 17 cột, tracks.parquet/meta.json và validate(); dùng tên cột thật. validate sạch không đồng nghĩa vị trí đúng ground truth. Feed thay thế là khả năng kiến trúc, không tuyên bố đã tích hợp hệ thống thương mại.

## 4. Geometry và camera
**Tiêu đề:** Camera chuyển động, hệ toạ độ sân giữ một chuẩn.
**Copy:** DLT ánh xạ điểm ảnh sang sân; TPS hiệu chỉnh phần méo còn lại. KLT theo chuyển động giữa frame và SIFT neo lại vào frame tham chiếu để hạn chế drift. Khi phép chiếu không xác định được, detection không được dùng như vị trí mét hợp lệ.
Dựng: landmark → điểm chân → pitch; split-view frame neo/frame sau lia; timeline cut cảnh. Ghi cần calibration/landmark, chưa tự động hoàn toàn.
Disclosure 35 landmark, MovingCalibration và phép biến đổi ngược. Residual điểm fit không gọi là accuracy toàn sân; fixed camera là trường hợp gần identity.

## 5. Detection, tracking và identity
**Tiêu đề:** Theo dõi chuyển động trong mét, nối lại những mảnh bị mất.
**Copy:** Detector người và bóng cung cấp quan sát; Kalman dự đoán vị trí/vận tốc, Hungarian ghép quan sát với track và appearance hỗ trợ phân biệt. Các tracklet được nối với ràng buộc thời gian, đội và số áo khi có dữ liệu đọc.
Dựng: một frame detection → trajectory → tracklet trước/sau; case giao cắt/occlusion có lỗi hiện rõ. Không chỉ chọn frame đẹp.
Disclosure model/tile/pass bóng là cấu hình cụ thể; jersey veto là cơ chế nhận dữ liệu số áo, chưa OCR hoàn thiện. Quota 11 mỗi đội theo thời gian không chứng minh luôn có 22 ID chính xác. Tách trọng tài còn hạn chế trên clip 3 phút.

## 6. Analytics — xử lý nhiễu trước khi diễn giải
**Tiêu đề:** Vị trí phải đủ tin cậy trước khi thành chỉ số.
**Copy:** Hampel loại spike và làm mượt quỹ đạo trước khi tính tốc độ, gia tốc và quãng đường. Từ vị trí đội có thể tính chiều rộng, chiều sâu và tâm khối để analyst đọc cấu trúc.
Dựng: raw/smoothed cùng một track thật, cùng trục/thời gian; sau đó width/depth khối đội trên radar. Ngưỡng/smoothing có thể làm mất chuyển động ngắn, cần chọn theo mục đích.
Disclosure formation/press_distance/transitions có module nhưng phụ thuộc input. PPDA cần events; possession hiện lấy từ ballStatus/event feed. Không viết “video tự suy ra toàn bộ event”.

## 7. Quality — công bố kết quả cùng điều kiện
**Tiêu đề:** Biết phần dữ liệu có và phần còn thiếu.
**Copy:** Quality report ghi presence và calibration của lượt chạy. Các tỷ lệ này cho biết dữ liệu xuất được; sai số vị trí và độ đúng identity cần phép đo với ground truth riêng.
Bảng lịch sử F2 ngày 23/09/2026, Brazil–France tactical cam, M2, YOLOv8x6 tile=0/1280 + bóng 1920, stride 1:

| Kết quả | Phạm vi/giới hạn |
|---|---|
| 745/750 frame có bóng (99.3%) | Clip 30 giây; presence, chưa accuracy |
| 3605/3657 frame có calibration có bóng (98.6%) | Clip 3 phút; 80.1% toàn clip do cắt cảnh 34 giây bị loại |
| 81.23% frame có calibration | Clip 3 phút, 139 re-anchor |
| 261 raw track → 179 tracklet | Phân mảnh identity còn nhiều |
| Khoảng 595 giây xử lý/phút trận trên M2 | Offline theo cấu hình có model bóng; không ngoại suy cả trận/GPU |

Dựng: link feasibility §3.4, caption giới hạn sát bảng. Cần report/artifact ghép clip trước public; chưa chạy lại trong lượt soạn nội dung. Không biến số 30 giây thành chất lượng toàn trận.

## 8. Engineering case — lỗi ở ranh giới module
**Tiêu đề:** Detector thấy bóng, bộ lọc vẫn có thể làm mất bóng.
**Copy:** Trong báo cáo F2, dùng cùng lề sân cho cầu thủ và bóng làm mất bóng gần biên. Tách lề bóng đưa presence của clip 30 giây từ 11.7% lên 99.3%. Bài học: kiểm cả phép chiếu và bộ lọc sau detector.
Dựng: before/after cùng clip nếu artifact có; nếu chưa có thì ghi báo cáo lịch sử và link nguồn. Presence tăng không chứng minh sai số vị trí giảm.

## 9. Tactical Query — phần đang phát triển
**Tiêu đề:** Từ dữ liệu vị trí tới tìm chuỗi chiến thuật.
**Copy:** Match State và các đặc trưng đội là nền tảng hiện có. Hướng tiếp theo là suy ra possession, phát hiện chuỗi bằng quy tắc, để analyst chấm và xây chỉ mục nhiều trận.
Dựng: L0 video / L1 Match State / L2 metrics / L3 possession / L4 sequences / L5 query; nhãn rõ phần nền tảng và phần còn thiếu để thành sản phẩm.
**Ví dụ câu hỏi concept:** Tìm các đoạn đối thủ dựng khối phòng ngự thấp.
Concept chỉ minh hoạ ý định; không dựng số kết quả hoặc clip như truy vấn đã chạy. DuckDB và classifier là hướng thiết kế, chưa tích hợp hoàn chỉnh.

## 10. Startup và pilot
**Tiêu đề:** Xây cùng người đọc trận đấu.
**Copy:** footlytics đang phát triển pipeline phân tích video và dữ liệu sân cho analyst. Pilot kỹ thuật tập trung vào một tác vụ, footage có quyền xử lý và đầu ra analyst có thể chấm.
**Quy trình đề xuất:** Chọn tác vụ/camera → chuẩn bị clip → kiểm geometry/tracking/quality → analyst phản hồi → xác định điều kiện mở rộng.
CTA khi có người nhận thật: Trao đổi pilot kỹ thuật.
Không dùng Hà Nội FC/PVF hay đầu mối tiếp cận làm logo khách hàng; chưa có founder/pháp nhân/giá được xác nhận thì không tự điền.

## 11. Kết trang
**Tiêu đề:** Kiểm pipeline trên footage và tác vụ cụ thể.
**Copy:** Chất lượng phụ thuộc camera, calibration, occlusion, đầu vào event và cấu hình chạy. Bản hiện tại là prototype; bước tiếp theo là kiểm đầu ra cùng analyst.
Link đọc: Cơ chế kỹ thuật · Điều kiện thử nghiệm · Nguồn benchmark.
Footer có contact thật khi được cung cấp; chưa tạo chính sách/link rỗng.

## Metadata
**Title:** footlytics — Video, Match State và phân tích bóng đá
**Description:** Prototype chuyển video bóng đá thành dữ liệu sân bằng calibration, bù chuyển động camera và tracking trong mét; nền tảng cho analytics và Tactical Query đang phát triển.
OG dùng asset thật có quyền hoặc đồ hoạ cơ chế có nhãn, không mô tả realtime.

## Quy tắc biên tập
Pipeline, geometry, tracking và evaluation nằm trên landing. Công thức/schema/config được mở ngay tại section liên quan; danh sách dependency không chiếm headline. Media tải theo nhu cầu; mobile không cần hover/play để hiểu. Mỗi số có commit/ngày/clip/cấu hình/mẫu số, mỗi roadmap có nhãn. Mô hình doanh thu, giá giả định, TAM và chiến lược gọi vốn giữ trong hồ sơ startup.

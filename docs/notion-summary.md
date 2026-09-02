# FOOTLYTICS — Tóm tắt ý tưởng (từ Notion, 2026-09-02)

Nguồn: Notion "FOOTLYTICS — MÔ HÌNH KINH DOANH (1)".

- **Định vị:** công ty phần mềm và thuật toán (software-first, algorithm-first). Computer Vision biến video trận đấu thành trạng thái trận đấu (game state) liên tục của 22 cầu thủ và bóng. Không phải công ty camera, wearable hay dashboard.
- **Chuỗi giá trị:** Video -> Computer Vision -> Game state -> Tactical intelligence -> Quy trình analyst -> Quyết định huấn luyện.
- **Hai engine:** Game-State Engine (pitch localisation, calibration, detection, tracking, team classification, ReID, ball tracking, image-to-pitch mapping, trajectory reconstruction, multi-camera fusion; output `timestamp, player_id, team, x, y, ball_x, ball_y, confidence`) và Tactical Intelligence Engine (formation, width/depth/line height, compactness, pressing, build-up, defensive block, off-ball movement, transition, overload, space occupation, passing lanes, line-breaking).
- **Use case V1:** phân tích sau trận (post-match) và phân tích đối thủ (opposition, 3-5 trận gần nhất -> mẫu chiến thuật -> chuỗi tình huống tìm kiếm được). Opposition analysis là mũi nhọn (wedge). Chưa làm live/halftime.
- **Sản phẩm dài hạn:** thư viện trận -> chọn đội -> tactical query ("High press — 18 chuỗi") -> chuỗi tình huống -> video + sân 2D + chỉ số -> xuất clip/báo cáo. Khác biệt: truy vấn theo ngôn ngữ chiến thuật, không chỉ tracking overlay.
- **Thị trường:** Vietnam-first B2B. Beachhead: CLB V.League có analyst (Hà Nội FC qua đầu mối truyền thông), học viện hàng đầu (PVF). Power user: analyst. Economic buyer chưa rõ. Không được nói "Việt Nam chưa có công nghệ phân tích".
- **Doanh thu:** match pack -> thuê bao theo mùa/đội -> multi-team/academy -> bespoke -> data/API. Chỉ số đơn vị: cost per processed match.
- **Moat:** dữ liệu VN + reconstruction ổn định + tactical models + tactical DB lịch sử + tích hợp workflow CLB + vòng lặp phản hồi. Tracking đơn thuần sẽ bị commoditized.
- **GTM:** phỏng vấn -> Hà Nội FC/học viện -> 1-3 trận thật -> prototype -> một tactical output có giá trị -> analyst validation -> pilot -> paid match pack -> subscription. Không mở rộng sales trước khi chứng minh end-to-end.
- **Đội ngũ:** cần technical cofounder (CV, tracking, ML infra), không phải vendor.
- **Blocker B1-B15:** thị trường (B1-B4), dữ liệu (B5-B7), kỹ thuật (B8-B12), sản phẩm (B13-B15).
- **Mục tiêu hiện tại:** không mở rộng đội kỹ thuật cho đến khi game state tái dựng cải thiện một quy trình thật của một analyst thật.

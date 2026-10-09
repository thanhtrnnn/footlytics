# footlytics — Brief landing ưu tiên kỹ thuật
08/10/2026 · Bản 0.2 · Chưa xây UI.
Khung: redesign-existing-projects + impeccable; không dùng aitc-report.

## Quyết định từ người dùng
Landing cần thể hiện nhiều góc kỹ thuật đã chọn lọc. Hướng kỹ thuật đã xác nhận, thay bản 0.1 đưa phần lớn stack/benchmark vào tài liệu phụ.
Đối tượng đề xuất: analyst/CLB, đối tác dữ liệu và người đánh giá kỹ thuật. Công việc đích là phân tích video/đối thủ; landing giải thích cơ chế prototype trước lời mời pilot.

## Luận điểm
Từ điểm ảnh đến dữ liệu sân: geometry và camera motion → tracking trong mét → Match State → analytics có kiểm chất lượng. Tactical Query là phần phát triển tiếp theo và có nhãn riêng.
Góc kỹ thuật phải cho thấy lựa chọn thuật toán, ranh giới module, số đo theo clip và lỗi còn mở. Danh sách dependency không thay câu chuyện này.

## Baseline
- Local /Users/quant/Desktop/Thanh Tran/projects/Footlytics, HEAD f9bbcc1ddf4e92596d1adfc6bff5852b04e9cff1, branch feat/ball-quality-cli. uv.lock có từ trước, giữ nguyên.
- Local remote upstream scalliontor/Footlytics; nguồn canonical mới là thanhtrnnn/footlytics theo technical-plan 1.1 ngày 23/09/2026, commit tài liệu 2f584af8046b0643f4ea7d73093d28a38d90e5a5.
- Local chưa thấy bề mặt web/landing; tìm kiếm remote chưa thấy package.json. Không suy mọi nhánh đều không có web.
- Chưa tự đổi remote, reset hoặc đồng bộ code. Số F2 remote không là benchmark của local HEAD.
- Chưa có footage/radar được chốt đầy đủ quyền và manifest để công bố.

## Cấu trúc đã chọn cho draft
1. Hero kỹ thuật và demo video/radar.
2. Architecture/Match State.
3. Geometry, camera motion và calibration failure.
4. Detection, tracking, identity.
5. Kinematics/tactics.
6. Quality/compute và một engineering lesson.
7. Tactical Query L0–L5 có nhãn giai đoạn.
8. Startup/pilot kỹ thuật.
Mười góc, nguồn, phép đo và demo: [TECHNICAL.md](TECHNICAL.md). Copy đầy đủ: [CONTENT.md](CONTENT.md).

## Chọn lọc
| Đưa lên landing | Mở ngay tại section | Giữ ngoài năng lực hiện có |
|---|---|---|
| Pipeline, DLT/TPS, KLT/SIFT, tracking mét, analytics, quality, giới hạn | Schema, công thức, config/model, manifest, nguồn benchmark, identity case | Realtime, 90 phút đã nghiệm thu, OCR hoàn thiện, tự suy possession, Tactical Query hoàn chỉnh |
| F2 có mẫu số và compute; startup/pilot | Roadmap/adapter đề xuất được gắn nhãn | Logo khách hàng, giá/founder/pháp nhân chưa xác nhận, dataset như footage được tự do công bố |

## Hình thức
Dùng video/radar/đồ thị thật để chọn bố cục; diagram thay đổi theo cơ chế, không tạo grid card cho mọi thuật toán. Technical disclosure là một phần landing, có anchor và đọc được trên mobile.
Chưa chọn palette/font hay gắn nhãn identity đã duyệt. Không sao chép thương hiệu thuế sang bóng đá. Nếu chưa có web stack, đề xuất landing tĩnh nhẹ; chỉ quyết định sau khi kiểm codebase triển khai.
Media tải theo nhu cầu; poster/caption cho người không play, không autoplay âm thanh hoặc phụ thuộc hover.

## Phạm vi và thứ tự tiếp theo
Lượt này cập nhật file nội dung/nguồn/checkpoint, chưa xây UI, cài stack, chạy pipeline hay deploy.
CP2: chọn một clip có quyền → artifact/report khớp commit/config → chốt kênh pilot.
CP3–CP4: concept bằng nội dung thật → build bề mặt web đã chọn. CP5 kiểm kỹ thuật/caption/media/mobile; CP6 release và vòng phản hồi analyst.
Còn thiếu asset/contact không cản draft kỹ thuật, nhưng demo public và CTA cần dữ liệu/điểm nhận thật.

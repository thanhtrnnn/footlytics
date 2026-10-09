# footlytics — nghiệm thu landing kỹ thuật
Cập nhật 08/10/2026. Người dùng đã giao one-shot và chỉ định footlytics.space trên accountCloudflare có sẵn.

| CP | Kết quả hiện tại |
|---|---|
| CP0 | Repo local/canonical và lệch phiên bản đã ghi; uv.lock cũ giữ nguyên, không reset nhánh hoặc đổi remote. |
| CP1 | Đã đưa10 góc kỹ thuật vào landing; các lớp roadmap có nhãn đúng trạng thái. |
| CP2 | F2 canonical 23/09/2026 có ngày/config/mẫu số/compute và giới hạn; không chạy lại local. Sơ đồ tổng hợp có nhãn, không dùng footage chưa xác nhận quyền. |
| CP3 | Film programme hiện tại: cream/field-green/chartreuse, displaycondensed, real video showcase, chapter navigation và About; desktop/mobile đã render và review. |
| CP4 | SemanticHTML/CSS/JS tĩnh trong web/, font selfhost, metadata/OG/favicon/policy/404 và buildpackage riêng. Không thay pipeline Python. |
| CP5 | Browser/width/zoom/keyboard/reducedmotion/axe và kiểm claim đạt ở phạm vi ghi bên dưới. Review cuối ship. |
| CP6 | Đã deploy Cloudflare Pages và CNAMEapex footlytics.space; giữ nguyên4MX/TXT. Guide/rollback trong DEPLOY.md. |

## Bằng chứng QA
- Chromium: khung 1920, 1440, 1024, 768, 390, 375, 320px; overflow0 và các main section lấp đầy parent tại năm kích thước kiểm sâu.
- Zoom200% không overflow; Tab đến skip link; forced-colors và reduced-motion được kiểm.
- Không lỗi runtime/asset/anchor trên bản local; axe không có serious/critical findings. Đây là kiểm tra tự động có phạm vi, không là chứng nhận WCAG đầy đủ.
- Đã xem full-page và hero desktop1440/mobile375; review cuối: ba nhóm finding đều resolved, disposition ship. Không có approved comp hoặc QUALITY BAR riêng; ceiling đối chiếu direction contract/craft floor.
- Detector một lượt: Drake không finding; Foot degraded parser nên chỉ kết quả regex, không tự nhận coverage đầy đủ.
- Mọi raster được phục vụ có provenance; sơ đồ Foot dùng dữ liệu tổng hợp có nhãn.

## Phạm vi và việc vận hành tiếp theo
- CTA hiện có email kickoff@footlytics.space được người dùng cung cấp; chỉ mở email, chưa có form/SLA hoặc xác nhận vận hành mailbox.
- Showcase có bốn video upstream c787ff6 với nguồn SoccerTrackv2/CC BY4.0 và phân biệt suppliedGT/prediction; radar có giới hạn calibration chưa xác minh. Sơ đồ tổng hợp vẫn ghi rõ minh hoạ.
- TacticalQuery/possession/OCR/identity chưa được nâng thành tính năngproduction chỉ vì được mô tả trên landing.
- Chưa bật funnel analytics hoặc nhận feedback bên ngoài; chưa đo lại pipeline hoặc tuyên bố90phút/realtime/accuracy.
- Workingtree chưa commit/push/merge. Thiết kế chỉ áp dụng bề mặt web, không áp thương hiệu cho pipeline Python.

## Nhật ký
08/10: khởi tạo brief/copy/claim ledger, chọn10 góc kỹ thuật. Sau đó build/render/QA và đóng ba nhóm reviewerfindings. Đã xuất bản theo yêu cầu người dùng; hosting thêm no-transform để tránh beacon tự chèn xung đột CSP.


## Real-video showcase redesign — 09/10/2026
- [x] User inspiration VGF and Studio DUY inspected visually; replacement world documented.
- [x] Four upstream c787ff6 MP4 exports and posters preserved; cropped10s hero explicitlyGT.
- [x] Player selection/configuration/limits, loading/error/retry, no-autoplay and synthetic lab controls implemented.
- [x] Desktop/mobile visual review; 320–1920 responsive overflow, reduced motion, forced colors and axe checks. All4 durations/playback and injected error recovery passed.
- [x] Provenance scan6 rasters0 missing; all videos below25MiB.
- [x] Fresh finish review disposition: ship, no material fixes. A generic fresh agent performed the full skill reviewer role.
- [x] Canonical design documentation refresh (DESIGN.md and JSON v2 sidecar).
- [x] Production publish cb7705d2 and live media/header smoke checks; HTTP200, Range206, all4 clips played, axe0, no JS/CSP errors.


## Startup About và UI refinement — 09/10/2026
- [x] Shared navigation/contact, mobile hierarchy/touch targets, chapter reading density và trang /about mới theo identity đang dùng.
- [x] Email kickoff@footlytics.space trong trang chủ/About/footer; trạng thái prototype/validation/roadmap hoặc trial/unfinished workflow tách rõ.
- [x] Local 4 routes:200, axe0; overflow0 ở320/375/768/1024/1440/1920; About dùng được không JavaScript.
- [x] Static build đạt; hero/video selector phát được, skip-link/forced-colors qua; không sửa pipeline hoặc uv.lock.
- [x] Fresh independent finish reviewer: disposition ship; generic agent thay named skill reviewer, không có material fixes.
- [x] Live bốn route:200, HTML byte-match, axe0, JS/CSP0, responsive375–1920; Foot10s model video phát và trả Range206.
- [x] Production `8577374b-3b25-414f-8bdc-fb1ca86b4a77`; rollback ghi trong DEPLOY.md.
- Phạm vi: UI/mailto đã hoàn thành; vận hành mailbox và mapping domain riêng Drake chưa được thực hiện trong tác vụ này.
- [x] Fresh generic documenter cập nhật scoped DESIGN.md và JSONv2 sidecar theo nguồn UI đã hoàn thành; canonical identity giữ nguyên.


## Responsive và domain — 09/10/2026
- [x] Fourpublicsurfaces homepage/About, phone/tablet/desktop riêng; fonts/prose/touch/keyboardscrolltables/safearea hoàn thành.
- [x] Chromium local/live320–2560px, axe0/runtime0; landscape/touch44×44/roottextscale200% đạt. Không có physical-device hoặc Safari QA (WebKitbinary chưa có).
- [x] Reviewship và freshgenericdocumenter cập nhật source-based design docs/JSONv2.
- [x] Published production `ed2e29f9-95d4-448a-b966-0b98f9946ee3`; guides/rollback trong DEPLOY.md.
- [x] Dx Figgle primary loaded, OTFbyteidentical; completeVietnamese fallback; nativehero/modelclip liveplayback/Range206.

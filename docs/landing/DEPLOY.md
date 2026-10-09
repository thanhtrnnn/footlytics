# Cloudflare Pages — footlytics

Public: https://footlytics.space
Pages: https://footlytics-space.pages.dev
Project: `footlytics-space`; production branch Pages: `main`.

## Cập nhật
Landing nằm trong `web/`; pipeline Python được giữ nguyên. Build đóng gói chỉ tài nguyên public, không gửi tài liệu thiết kế/kế hoạch lên hosting.

```sh
python3 scripts/build_landing.py
npx --yes wrangler@4.148.0 pages deploy dist-web --project-name footlytics-space --branch main
```

Đặt token qua biến môi trường `CLOUDFLARE_API_TOKEN`, account ID qua `CLOUDFLARE_ACCOUNT_ID`; không ghi token vào repo.
Quyền: Account / Cloudflare Pages / Edit; khi quản lý domain, Zone / DNS / Edit và Zone / Zone / Read, chỉ scope `footlytics.space`.

## Domain và bảo mật
Apex CNAME proxied: `footlytics.space` → `footlytics-space.pages.dev`. Custom domain được gắn vào Pages trước khi đổi DNS. MX/TXT được giữ nguyên.
`_headers` giữ CSP tài nguyên cùng origin và `Cache-Control: public, max-age=0, must-revalidate, no-transform` cho HTML, ngăn proxy tự chèn beacon vào payload. Đây không phải analytics funnel; trang không cài form hoặc thu footage.
Tham khảo: https://developers.cloudflare.com/web-analytics/get-started/

## Rollback
Cloudflare Pages Deployments → chọn production trước → Rollback. Bản hiện tại: `8577374b-3b25-414f-8bdc-fb1ca86b4a77` (09/10/2026 UTC+7). Bản trước để rollback: `cb7705d2-2f4f-453a-b8b7-49d1879b8e1e`. Bản đầu `335a35d5-73a9-4951-834c-91e611d91016` có beacon tự chèn bị CSP chặn; ưu tiên redeploy package hiện tại thay vì quay lại cấu hình header cũ.
Khôi phục hosting web trước Pages: thay đúng bản ghi CNAME apex hiện tại bằng A `171.226.10.154`, proxied true, TTL auto. Không sửa các MX/TXT.
Nguồn working tree trên `feat/ball-quality-cli`, base HEAD `f9bbcc1ddf4e92596d1adfc6bff5852b04e9cff1`; landing chưa commit. F2 trên trang là report canonical 23/09/2026, không phải kết quả chạy lại code local.


## Showcase release
Four original videos from c787ff6 plus a labelled10s GT hero crop. All files fit Pages25MiB per-file upload limit. Native players load only on visitor action; range responses206/video-mp4 checked on all5 assets. Homepage byte hash matches package, no analytics beacon injection. Live desktop/mobile playback, geometry controls, keyboard skip link, reduced motion selection and axe0 violations checked. Temporary token file removed after publish. Existing DNS/MX/TXT unchanged in this release.


## Startup About và tinh chỉnh UI — 09/10/2026
- Production: `8577374b-3b25-414f-8bdc-fb1ca86b4a77`; immutable URL: https://8577374b.footlytics-space.pages.dev.
- About: https://footlytics.space/about/.
- Rollback production trước: `cb7705d2-2f4f-453a-b8b7-49d1879b8e1e` (https://cb7705d2.footlytics-space.pages.dev).
- Trang chủ và About cùng gói build; About hoạt động với JavaScript tắt. Navigation/footer có email người dùng xác nhận, chỉ là mailto.
- Live: HTML khớp byte với package, bốn route HTTP200, axe0, không lỗi JS/CSP, viewport375/1024/1440/1920 không overflow/underfill.
- Không sửa DNS/MX/TXT hoặc tạo mailbox; không thay portal/backend. Token tạm đã xoá sau deploy. Working tree chưa commit/push.


## Responsive, font và domain — 09/10/2026
Production mới nhất: `ed2e29f9-95d4-448a-b966-0b98f9946ee3`; immutable: https://ed2e29f9.footlytics-space.pages.dev. Public: https://footlytics.space, About: https://footlytics.space/about/.
Deployment liền trước: `8577374b-3b25-414f-8bdc-fb1ca86b4a77` (https://8577374b.footlytics-space.pages.dev); dùng Pages Deployments → Rollback nếu cần.
Kiểm live4routes200, HTMLbyte-match, CSS/fontmatch, axe0, JS/CSP0, responsive320–2560px đạt. Review độc lập ship; tài liệu scoped đã cập nhật. Token tạm đã xoá, working tree chưa commit/push.
Dx Figgle OTF selfhost/raw unchanged; user xác nhận WebFontlicense. Chữ không được font hỗ trợ dùng complete ReplayDisplay role. Rollback trước lượt này:8577374b-3b25-414f-8bdc-fb1ca86b4a77.


## Browser verification — 09/10/2026
Production: `b6840b80-d94a-4d9c-a76c-1795d3f2b5c9` (https://b6840b80.footlytics-space.pages.dev). Build uses content-hashed CSS/JS URLs across public HTML to avoid returning-browser stale assets. Verified new stylesheet and Dx Figgle in actual IAB; hero/prediction playback and tracking controls work. Hero media links use light hover/focus contrast. No DNS changes. Token removed.


## Bricolage typography — 09/10/2026
Production `4c3ba63b-f664-4b83-80c5-f004c194dc4a` (https://4c3ba63b.footlytics-space.pages.dev). User selected Bricolage Grotesque. Headings/brand 550, technology labels 500; Vietnamese supported in the same face. Self-hosted raw variable TTF with OFL license; shared CSS and both home/About font preloads updated.

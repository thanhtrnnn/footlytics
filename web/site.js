const pitch = document.querySelector('#projection-pitch');
const groupIds = ['field-lines', 'pitch-guides', 'players-home', 'players-away', 'calibration-anchors', 'track-lines'];
const buttons = [...document.querySelectorAll('[data-layer]')];
const caption = document.querySelector('#layer-caption');
const unit = document.querySelector('#desk-unit');
const captions = {
  image: 'Minh hoạ cơ chế · dữ liệu tổng hợp. Phối cảnh làm vị trí và tỷ lệ chuyển động trong điểm ảnh thay đổi; cần calibration trước khi dùng mét.',
  pitch: 'Minh hoạ cơ chế · dữ liệu tổng hợp. Các điểm được đưa về hệ toạ độ sân; không phải bản chạy pipeline.',
  track: 'Minh hoạ cơ chế · dữ liệu tổng hợp. Quỹ đạo biểu diễn chuyển động trong mét; không phải ID hay kinematics đã đo trên một trận thật.',
};
buttons.forEach(button => button.addEventListener('click', () => {
  const layer = button.dataset.layer;
  buttons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  pitch.dataset.layer = layer;
  const perspective = layer === 'image' ? 'translate(64 86) skewX(-14) scale(.91 .65)' : '';
  groupIds.forEach(id => {
    const element = document.getElementById(id);
    if (perspective) element.setAttribute('transform', perspective);
    else element.removeAttribute('transform');
  });
  const ball = pitch.querySelector('.ball');
  if (perspective) ball.setAttribute('transform', perspective);
  else ball.removeAttribute('transform');
  caption.textContent = captions[layer];
  unit.textContent = layer === 'image' ? 'Image space · minh hoạ' : 'Toạ độ sân · mét';
}));
// Real-video controls; the pitch illustration below remains a separate, labelled demo.
const reel = document.querySelector('#hero-video');
const reelButton = document.querySelector('#reel-toggle');
const reelStatus = document.querySelector('#reel-status');
function reelLabel(playing) {
  reelButton.setAttribute('aria-pressed', String(playing));
  reelButton.querySelector('span').textContent = playing ? 'Tạm dừng trích đoạn' : 'Phát trích đoạn · 10s';
  reelButton.querySelector('path').setAttribute('d', playing ? 'M7 5h3v14H7zM14 5h3v14h-3z' : 'm9 5 10 7-10 7z');
}
reelButton.addEventListener('click', async () => {
  if (!reel.paused) { reel.pause(); return; }
  reelStatus.textContent = 'Đang tải trích đoạn…';
  if (reel.error) reel.load();
  try { await reel.play(); }
  catch { reelStatus.textContent = 'Chưa phát được trích đoạn. Thử lại hoặc mở video đầy đủ ở showcase bên dưới.'; }
});
reel.addEventListener('playing', () => { reelLabel(true); reelStatus.textContent = ''; });
reel.addEventListener('pause', () => reelLabel(false));
reel.addEventListener('error', () => { reelLabel(false); reelStatus.textContent = 'Video chưa tải được. Thử lại hoặc mở showcase đầy đủ bên dưới.'; });
new IntersectionObserver(entries => {
  if (!entries[0].isIntersecting) reel.pause();
}, { threshold: .05 }).observe(reel);
document.addEventListener('visibilitychange', () => { if (document.hidden) reel.pause(); });

const demos = {
  gt40: { file: 'gt-video-radar-40s', title: 'Video & radar đồng bộ', kind: 'Ground truth · 40 giây · 25 fps', description: 'Footage thật, bbox và số áo từ annotations, cùng radar và quỹ đạo. Hai góc nhìn của cùng một đoạn trận đấu.', limit: 'Dùng ground truth cung cấp sẵn, không đo accuracy của model.' },
  gt240: { file: 'gt-radar-4min', title: 'Theo trận đấu trong bốn phút', kind: 'Ground truth · 4 phút · 25 fps', description: 'Radar toàn sân, vị trí và trails từ annotations. Cửa sổ dài hơn để quan sát cách đội hình thay đổi theo thời gian.', limit: 'Chỉ hiển thị ground truth. Radar dùng giả định kích thước sân legacy chưa được xác minh.' },
  model25: { file: 'pipeline-detection-10s', title: 'Khi pipeline tự nhìn trận đấu', kind: 'Model prediction · 10 giây · 25 fps', description: 'Nhận diện và chiếu lên radar với confidence 0.25, pitch margin 2m. Xem trực tiếp những quan sát model xuất ra.', limit: 'Còn bỏ sót người, false positive và ID phân mảnh. Chưa có ground-truth scoring cho lượt chạy này.' },
  model10: { file: 'pipeline-detection-low-confidence-10s', title: 'Nhận nhiều hơn. Kiểm lỗi kỹ hơn.', kind: 'Model prediction · 10 giây · 25 fps', description: 'Lượt chạy với confidence 0.10, pitch margin 6m. Ngưỡng nhận rộng hơn cần được kiểm cùng bộ lọc và chất lượng tracking.', limit: 'Cả confidence và pitch margin đều thay đổi. Không phải so sánh có kiểm soát chỉ một tham số.' },
};
const demoVideo = document.querySelector('#demo-video');
const demoStatus = document.querySelector('#demo-status');
const retry = document.querySelector('#demo-retry');
const selectors = [...document.querySelectorAll('[data-demo]')];
selectors.forEach(button => button.addEventListener('click', event => {
  if (button.getAttribute('aria-pressed') === 'true') return;
  const demo = demos[button.dataset.demo];
  demoVideo.pause();
  selectors.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  demoVideo.src = `/assets/demos/${demo.file}.mp4`;
  demoVideo.poster = `/assets/demos/${demo.file}.jpg`;
  document.querySelector('#demo-title').textContent = demo.title;
  document.querySelector('#demo-kind').textContent = demo.kind;
  document.querySelector('#demo-description').textContent = demo.description;
  document.querySelector('#demo-limit').textContent = demo.limit;
  document.querySelector('#demo-file').href = demoVideo.src;
  retry.hidden = true;
  demoStatus.textContent = `Đã chọn: ${demo.title}. Chọn Phát để tải và xem.`;
  demoVideo.load();
  if (innerWidth <= 640 && event.detail > 0) {
    demoVideo.scrollIntoView({ block: 'start', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
  }
}));
demoVideo.addEventListener('waiting', () => { demoStatus.textContent = 'Đang tải video…'; });
demoVideo.addEventListener('playing', () => { demoStatus.textContent = ''; retry.hidden = true; });
demoVideo.addEventListener('loadedmetadata', () => { demoStatus.textContent = 'Video sẵn sàng. Dùng điều khiển để phát, tua hoặc mở toàn màn hình.'; });
demoVideo.addEventListener('error', () => {
  demoStatus.textContent = 'Video chưa tải được. Tải lại hoặc chọn “Mở video riêng” để xem trực tiếp.';
  retry.hidden = false;
});
retry.addEventListener('click', async () => {
  retry.hidden = true;
  demoStatus.textContent = 'Đang tải lại video…';
  demoVideo.load();
  try { await demoVideo.play(); }
  catch { demoStatus.textContent = 'Chưa phát được video. Thử “Mở video riêng” hoặc chọn video khác.'; retry.hidden = false; }
});

"""Landmark clicking for calibration.

This is the only manual step in the whole pipeline, and for a fixed camera it
happens once per venue rather than once per match -- so it is worth doing
carefully. Aim for 8-15 landmarks spread across the *whole* frame: measured on
a synthetic camera, 4 clustered points give ~1 m of error while 20 spread ones
give ~0.2 m (scripts/test_geometry.py).
"""

from __future__ import annotations

import base64
import io
import json
from typing import Optional

import numpy as np

from .homography import Calibration
from .pitch import Pitch, DEFAULT_PITCH

# Suggested clicking order: the corners pin down the projective part, the
# penalty boxes stabilise each end, the halfway line fixes the middle.
SUGGESTED = [
    "corner_LT", "corner_RT", "corner_RB", "corner_LB",
    "halfway_T", "halfway_B",
    "pen_area_LT_front", "pen_area_LB_front",
    "pen_area_RT_front", "pen_area_RB_front",
    "pen_area_LT_goalline", "pen_area_LB_goalline",
    "pen_area_RT_goalline", "pen_area_RB_goalline",
    "centre_spot",
]

HELP = {
    "corner_LT": "top-left corner flag (left goal, near touchline)",
    "corner_RT": "top-right corner flag",
    "corner_RB": "bottom-right corner flag",
    "corner_LB": "bottom-left corner flag",
    "halfway_T": "halfway line meets the top touchline",
    "halfway_B": "halfway line meets the bottom touchline",
    "pen_area_LT_front": "LEFT penalty box, front line, top corner",
    "pen_area_LB_front": "LEFT penalty box, front line, bottom corner",
    "pen_area_RT_front": "RIGHT penalty box, front line, top corner",
    "pen_area_RB_front": "RIGHT penalty box, front line, bottom corner",
    "pen_area_LT_goalline": "LEFT penalty box meets the goal line, top",
    "pen_area_LB_goalline": "LEFT penalty box meets the goal line, bottom",
    "pen_area_RT_goalline": "RIGHT penalty box meets the goal line, top",
    "pen_area_RB_goalline": "RIGHT penalty box meets the goal line, bottom",
    "centre_spot": "the centre spot",
}


def show_frame_grid(frame: np.ndarray, figsize=(20, 11), step: int = 200):
    """Display a frame with a labelled pixel grid.

    The always-works fallback: read the coordinates off the axes and type them
    into a dict. Ugly, but it never breaks, and every fancier widget below can
    and does break depending on the notebook frontend.
    """
    import matplotlib.pyplot as plt

    h, w = frame.shape[:2]
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(frame)
    ax.set_xticks(range(0, w, step)); ax.set_yticks(range(0, h, step))
    ax.grid(color="yellow", alpha=0.35, lw=0.6)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig, ax


def _png_data_uri(frame: np.ndarray, max_width: int = 1600) -> tuple[str, float]:
    """Encode a frame as a data URI, downscaled for the browser. Returns the
    URI and the scale factor needed to map clicks back to full resolution."""
    from PIL import Image

    img = Image.fromarray(frame)
    scale = min(max_width / img.width, 1.0)
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), scale


def draw_candidates(frame: np.ndarray, candidates: list[dict], radius: int = 6) -> np.ndarray:
    """Mark calib-assist candidates (dicts with id, x, y) on a copy of the frame.

    Used to pre-draw `footlytics calib-assist` proposals in the clicker, so the operator
    clicks on a detected line intersection instead of estimating a corner by eye at 4K.
    """
    import cv2

    out = np.ascontiguousarray(frame).copy()
    for c in candidates:
        x, y = int(round(c["x"])), int(round(c["y"]))
        cv2.circle(out, (x, y), radius, (255, 0, 0), 2)
        cv2.circle(out, (x, y), 2, (255, 0, 0), -1)          # filled centre: the exact pixel to click
        cv2.putText(out, f"P{c['id']}", (x + radius + 2, y - radius), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return out


def click_landmarks(frame: np.ndarray, landmarks: Optional[list[str]] = None,
                    max_width: int = 1600, candidates: Optional[list[dict]] = None):
    """Click landmarks directly on the frame, in a notebook.

    `candidates`: calib-assist proposals (`{"id", "x", "y"}` dicts) drawn on the frame first.

    Renders the frame in an HTML canvas, walks you through the landmark list,
    and stores the result in `footlytics.geometry.annotate.LAST_CLICKS`.
    Zoom with the scroll wheel, since a corner flag is a few pixels wide at 4K
    displayed in a browser.
    """
    from IPython.display import HTML, display

    names = landmarks or SUGGESTED
    if candidates:
        frame = draw_candidates(frame, candidates)
    uri, scale = _png_data_uri(frame, max_width)
    payload = json.dumps({
        "src": uri, "scale": scale,
        "names": names, "help": {n: HELP.get(n, n) for n in names},
    })

    html = """
<div id="fl-wrap" style="font:13px system-ui;color:#e6edf3;background:#0d1117;padding:10px;border-radius:8px">
  <div id="fl-prompt" style="margin-bottom:6px;font-size:15px"></div>
  <div style="margin-bottom:6px">
    <button id="fl-skip">skip this one</button>
    <button id="fl-undo">undo</button>
    <button id="fl-done">done - print result</button>
    <span id="fl-count" style="margin-left:10px;opacity:.8"></span>
    <span style="margin-left:10px;opacity:.6">scroll = zoom, drag = pan</span>
  </div>
  <canvas id="fl-cv" width="1000" height="560"
          style="border:1px solid #30363d;cursor:crosshair;background:#000"></canvas>
  <pre id="fl-out" style="white-space:pre-wrap;background:#161b22;padding:8px;
       border-radius:6px;margin-top:8px;max-height:220px;overflow:auto"></pre>
</div>
<script>
(function(){
  const D = __PAYLOAD__;
  const cv = document.getElementById('fl-cv'), ctx = cv.getContext('2d');
  const img = new Image(); let i = 0, pts = {};
  let z = 1, ox = 0, oy = 0, drag = null, moved = false;

  function fit(){ z = Math.min(cv.width/img.width, cv.height/img.height); ox = 0; oy = 0; }
  function draw(){
    ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,cv.width,cv.height);
    ctx.setTransform(z,0,0,z,ox,oy); ctx.drawImage(img,0,0);
    ctx.setTransform(1,0,0,1,0,0);
    for (const [n,p] of Object.entries(pts)){
      const x = p[0]*D.scale*z+ox, y = p[1]*D.scale*z+oy;
      ctx.strokeStyle='#ffd60a'; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.moveTo(x-7,y); ctx.lineTo(x+7,y);
      ctx.moveTo(x,y-7); ctx.lineTo(x,y+7); ctx.stroke();
      ctx.fillStyle='#ffd60a'; ctx.font='10px system-ui'; ctx.fillText(n, x+9, y-4);
    }
    const c = document.getElementById('fl-count');
    c.textContent = Object.keys(pts).length + ' / ' + D.names.length + ' placed';
  }
  function prompt_(){
    const el = document.getElementById('fl-prompt');
    if (i >= D.names.length){ el.innerHTML = '<b>All landmarks visited.</b> Press "done".'; return; }
    const n = D.names[i];
    el.innerHTML = 'Click: <b style="color:#ffd60a">' + n + '</b> &mdash; ' + D.help[n];
  }
  cv.addEventListener('mousedown', e => { drag = [e.offsetX, e.offsetY, ox, oy]; moved = false; });
  window.addEventListener('mouseup', () => { drag = null; });
  cv.addEventListener('mousemove', e => {
    if (!drag) return;
    if (Math.abs(e.offsetX-drag[0])+Math.abs(e.offsetY-drag[1]) > 3) moved = true;
    ox = drag[2] + e.offsetX - drag[0]; oy = drag[3] + e.offsetY - drag[1]; draw();
  });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.15 : 1/1.15;
    ox = e.offsetX - (e.offsetX-ox)*f; oy = e.offsetY - (e.offsetY-oy)*f; z *= f; draw();
  }, {passive:false});
  cv.addEventListener('click', e => {
    if (moved || i >= D.names.length) return;
    // back to ORIGINAL full-resolution pixels: undo pan, zoom, then downscale
    const x = ((e.offsetX-ox)/z)/D.scale, y = ((e.offsetY-oy)/z)/D.scale;
    pts[D.names[i]] = [Math.round(x*100)/100, Math.round(y*100)/100];
    i++; prompt_(); draw(); emit();
  });
  document.getElementById('fl-skip').onclick = () => { i++; prompt_(); draw(); };
  document.getElementById('fl-undo').onclick = () => {
    if (i > 0){ i--; delete pts[D.names[i]]; prompt_(); draw(); emit(); }
  };
  document.getElementById('fl-done').onclick = emit;
  function emit(){
    const txt = 'IMAGE_POINTS = ' + JSON.stringify(pts, null, 2)
        .replace(/\\[\\n\\s+/g,'(').replace(/,\\n\\s+/g,', ').replace(/\\n\\s+\\]/g,')')
        .replace(/\\[/g,'(').replace(/\\]/g,')');
    document.getElementById('fl-out').textContent = txt;
    if (window.google && google.colab && google.colab.kernel){
      google.colab.kernel.invokeFunction('footlytics.clicks', [JSON.stringify(pts)], {});
    }
  }
  img.onload = () => { fit(); prompt_(); draw(); };
  img.src = D.src;
})();
</script>
""".replace("__PAYLOAD__", payload)

    try:                                    # let Colab push results back to Python
        from google.colab import output as _colab_output

        def _receive(js: str):
            global LAST_CLICKS
            LAST_CLICKS = {k: tuple(v) for k, v in json.loads(js).items()}
        _colab_output.register_callback("footlytics.clicks", _receive)
    except Exception:
        pass

    display(HTML(html))
    print("Click each prompted landmark. When finished press 'done'.")
    print("In Colab the points also land in footlytics.geometry.annotate.LAST_CLICKS;")
    print("elsewhere, copy the IMAGE_POINTS block printed under the canvas.")


LAST_CLICKS: dict[str, tuple[float, float]] = {}


def build_calibration(image_points: dict, frame: np.ndarray,
                      pitch: Pitch = DEFAULT_PITCH, use_tps: bool = False,
                      verbose: bool = True) -> Calibration:
    """Fit and immediately grade a calibration, so a bad one cannot pass quietly."""
    h, w = frame.shape[:2]
    lm = pitch.landmarks()
    cal = Calibration.from_correspondences(image_points, lm, (w, h), use_tps=use_tps)
    ok, msg = cal.verdict(lm)
    if verbose:
        r = cal.reprojection_error(lm)
        print(f"calibration from {r['n_points']} landmarks: {msg}")
        print(f"  mean {r['mean_m']:.3f} m | median {r['median_m']:.3f} m | max {r['max_m']:.3f} m")
        worst = sorted(r["per_point_m"].items(), key=lambda kv: -kv[1])[:3]
        print("  worst points: " + ", ".join(f"{n} {e:.2f} m" for n, e in worst))
        if not ok:
            print("  -> do not build analytics on this. Re-click the worst points.")
    return cal


def overlay_pitch(frame: np.ndarray, cal: Calibration, pitch: Pitch = DEFAULT_PITCH,
                  figsize=(18, 10)):
    """Project the pitch model back onto the frame -- the real calibration check.

    Numbers can look fine while the mapping is subtly wrong. If these lines sit
    on the painted lines, the calibration is right; if they drift at one end,
    it is not.
    """
    import matplotlib.pyplot as plt
    from .pitch import CENTRE_CIRCLE_R, PENALTY_AREA_DEPTH, PENALTY_AREA_WIDTH

    L, W = pitch.half_l, pitch.half_w
    segs = [
        [(-L, -W), (L, -W)], [(-L, W), (L, W)],
        [(-L, -W), (-L, W)], [(L, -W), (L, W)],
        [(0, -W), (0, W)],
    ]
    for sx in (-1, 1):
        pa_x, pa_y = sx * (L - PENALTY_AREA_DEPTH), PENALTY_AREA_WIDTH / 2
        segs += [[(sx * L, -pa_y), (pa_x, -pa_y)], [(sx * L, pa_y), (pa_x, pa_y)],
                 [(pa_x, -pa_y), (pa_x, pa_y)]]
    circle = [(CENTRE_CIRCLE_R * np.cos(a), CENTRE_CIRCLE_R * np.sin(a))
              for a in np.linspace(0, 2 * np.pi, 80)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(frame)
    for s in segs:
        p = cal.pitch_to_image(s)
        ax.plot(p[:, 0], p[:, 1], color="#ffd60a", lw=2.0)
    c = cal.pitch_to_image(circle)
    ax.plot(c[:, 0], c[:, 1], color="#ffd60a", lw=2.0)
    for n, (px, py) in cal.named_points.items():
        ax.plot(px, py, "o", ms=6, mfc="none", mec="#ff006e", mew=2)
    ax.set_title("pitch model projected onto the frame -- yellow should sit on the paint")
    ax.axis("off")
    plt.tight_layout()
    return fig, ax

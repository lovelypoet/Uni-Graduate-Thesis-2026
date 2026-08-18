#!/usr/bin/env python3
"""
PCB Component Labeling Tool v2
- Chạy: python3 pcb_labeler_v2.py
- Mở browser: http://localhost:5000
- Click + kéo để crop component, gõ label để lưu

NÂNG CẤP v2:
  • Context margin: tự động lấy rộng quanh vùng chọn để THẤY PAD
    (giải quyết vấn đề component nằm ngang/chéo cần reference)
  • Square crop: tùy chọn crop vuông (tốt cho CNN input)
  • Zoom in/out ảnh để chọn component nhỏ chính xác hơn
  • Lưu kèm ảnh gốc context để review lại
  • Hiện kích thước crop (px) real-time
"""

import os
import sys
import json
import base64
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# ── Cấu hình ──────────────────────────────────────────────
# Chọn folder lưu qua tham số dòng lệnh:
#   python3 pcb_labeler_v2.py            → lưu vào dataset/
#   python3 pcb_labeler_v2.py dataset_v2 → lưu vào dataset_v2/
OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
CLASSES = {
    "o": "ok", "s": "shift", "r": "rotate", "m": "missing",
    "t": "tombstone", "i": "tilt", "b": "bridge",
}
# ──────────────────────────────────────────────────────────

for cls in CLASSES.values():
    (OUTPUT_DIR / "train" / cls).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / cls).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test" / cls).mkdir(parents=True, exist_ok=True)

counters = {cls: len(list((OUTPUT_DIR / "train" / cls).glob("*.jpg")))
                 + len(list((OUTPUT_DIR / "val" / cls).glob("*.jpg")))
                 + len(list((OUTPUT_DIR / "test" / cls).glob("*.jpg")))
            for cls in CLASSES.values()}

HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>PCB Labeler v2</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif; }
  .header { background: #16213e; padding: 12px 20px; display: flex; align-items: center; gap: 20px; border-bottom: 2px solid #0f3460; }
  .header h1 { font-size: 18px; color: #e94560; }
  .header .badge { background:#4ecca3; color:#000; font-size:11px; padding:2px 8px; border-radius:10px; font-weight:bold; }
  .main { display: flex; height: calc(100vh - 56px); }
  .sidebar { width: 280px; background: #16213e; padding: 16px; display: flex; flex-direction: column; gap: 14px; overflow-y: auto; border-right: 1px solid #0f3460; padding-bottom: 40px; }
  .canvas-area { flex: 1; display: flex; align-items: center; justify-content: center; overflow: auto; padding: 16px; position: relative; }
  #canvas { cursor: crosshair; border: 2px solid #0f3460; }
  .section-title { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #888; margin-bottom: 4px; }
  .upload-btn { background: #e94560; color: white; border: none; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-size: 14px; width: 100%; font-weight: bold; }
  .upload-btn:hover { background: #c73652; }
  .class-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .class-btn { padding: 8px 6px; border: 2px solid #333; border-radius: 6px; cursor: pointer; font-size: 12px; text-align: center; background: #0f3460; color: #eee; transition: all 0.15s; }
  .class-btn:hover { border-color: #e94560; transform: scale(1.03); }
  .class-btn.active { border-color: #4ecca3; background: #1a5a4a; color: #4ecca3; }
  .class-btn .key { display: inline-block; background: #333; border-radius: 3px; padding: 1px 5px; font-size: 10px; font-weight: bold; margin-right: 4px; color: #ffd700; }
  .stats { background: #0f3460; border-radius: 8px; padding: 10px; }
  .stat-row { display: flex; justify-content: space-between; font-size: 12px; padding: 2px 0; border-bottom: 1px solid #1a3a6a; }
  .stat-row:last-child { border: none; }
  .stat-count { color: #4ecca3; font-weight: bold; }
  .controls { display: flex; flex-direction: column; gap: 6px; }
  .btn { padding: 8px; border: 1px solid #333; border-radius: 6px; cursor: pointer; font-size: 12px; background: #0f3460; color: #eee; text-align: center; }
  .btn:hover { background: #1a4a8a; }
  .btn.danger { border-color: #e94560; }
  .btn.danger:hover { background: #4a1a2a; }
  .info-box { background: #0a2040; border: 1px solid #1a4a8a; border-radius: 8px; padding: 10px; font-size: 11px; line-height: 1.7; color: #aaa; }
  .info-box b { color: #4ecca3; }
  .status-bar { position: fixed; bottom: 0; left: 0; right: 0; background: #0f3460; padding: 6px 16px; font-size: 12px; color: #4ecca3; border-top: 1px solid #1a4a8a; z-index: 100; }
  #overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.7); display: flex; align-items: center; justify-content: center; z-index: 50; }
  .save-popup { background: #16213e; border: 2px solid #4ecca3; border-radius: 12px; padding: 24px 32px; text-align: center; min-width: 300px; }
  .save-popup h3 { color: #4ecca3; margin-bottom: 6px; }
  .save-popup .sub { font-size:11px; color:#888; margin-bottom:10px; }
  .save-popup .preview { width: 220px; height: 165px; object-fit: contain; border: 1px solid #333; background: #000; margin: 10px auto; display: block; image-rendering: pixelated; }
  .save-popup .label-buttons { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }
  .save-popup .lbl-btn { padding: 8px; border: 1px solid #333; border-radius: 6px; cursor: pointer; font-size: 12px; background: #0f3460; color: #eee; }
  .save-popup .lbl-btn:hover { background: #1a5a4a; border-color: #4ecca3; }
  .save-popup .cancel-btn { margin-top: 10px; padding: 6px 20px; background: #333; border: none; border-radius: 6px; color: #aaa; cursor: pointer; font-size: 12px; }
  .toast { position: fixed; top: 20px; right: 20px; background: #4ecca3; color: #000; padding: 10px 20px; border-radius: 8px; font-weight: bold; font-size: 14px; z-index: 999; opacity: 0; transition: opacity 0.3s; pointer-events: none; }
  .toast.show { opacity: 1; }
  .no-image { color: #555; text-align: center; font-size: 14px; }
  .no-image .icon { font-size: 48px; margin-bottom: 12px; }
  .slider-row { display:flex; align-items:center; gap:8px; font-size:12px; }
  .slider-row input[type=range] { flex:1; }
  .slider-val { color:#4ecca3; font-weight:bold; min-width:38px; text-align:right; }
  .toggle-row { display:flex; align-items:center; gap:8px; font-size:12px; cursor:pointer; }
  .toggle-row input { width:16px; height:16px; cursor:pointer; }
  .zoom-controls { display:flex; gap:6px; }
  .zoom-controls .btn { flex:1; }
</style>
</head>
<body>

<div class="header">
  <h1>🔬 PCB Component Labeler</h1>
  <span class="badge">v2</span>
  <span style="color:#888; font-size:13px;">Kéo để crop → context margin tự lấy pad → chọn label</span>
</div>

<div class="main">
  <div class="sidebar">
    <div>
      <div class="section-title">Ảnh PCB</div>
      <input type="file" id="fileInput" accept="image/*" style="display:none">
      <button class="upload-btn" onclick="document.getElementById('fileInput').click()">📁 Mở ảnh PCB</button>
      <div id="filename" style="font-size:11px; color:#888; margin-top:6px; word-break:break-all;"></div>
    </div>

    <!-- NEW: Crop settings -->
    <div>
      <div class="section-title">⚙️ Cài đặt crop (mới)</div>
      <div class="slider-row">
        <span>Context</span>
        <input type="range" id="marginSlider" min="0" max="100" value="30">
        <span class="slider-val" id="marginVal">30%</span>
      </div>
      <div style="font-size:10px; color:#777; margin:2px 0 8px;">Lấy rộng quanh vùng chọn để thấy pad → model có reference</div>
      <label class="toggle-row">
        <input type="checkbox" id="squareToggle" checked>
        <span>Crop vuông (tốt cho CNN)</span>
      </label>
    </div>

    <!-- NEW: Zoom -->
    <div>
      <div class="section-title">🔍 Zoom ảnh</div>
      <div class="zoom-controls">
        <div class="btn" onclick="zoomBy(0.8)">➖</div>
        <div class="btn" onclick="fitZoom()">Fit</div>
        <div class="btn" onclick="zoomBy(1.25)">➕</div>
      </div>
    </div>

    <div>
      <div class="section-title">Phím tắt / Label</div>
      <div class="class-grid">
        <div class="class-btn" data-key="o" onclick="selectClass('ok','o')"><span class="key">O</span>OK</div>
        <div class="class-btn" data-key="s" onclick="selectClass('shift','s')"><span class="key">S</span>Shift</div>
        <div class="class-btn" data-key="r" onclick="selectClass('rotate','r')"><span class="key">R</span>Rotate</div>
        <div class="class-btn" data-key="m" onclick="selectClass('missing','m')"><span class="key">M</span>Missing</div>
        <div class="class-btn" data-key="t" onclick="selectClass('tombstone','t')"><span class="key">T</span>Tombstone</div>
        <div class="class-btn" data-key="i" onclick="selectClass('tilt','i')"><span class="key">I</span>Tilt/Nghiêng</div>
        <div class="class-btn" data-key="b" onclick="selectClass('bridge','b')"><span class="key">B</span>Bridge</div>
      </div>
    </div>

    <div>
      <div class="section-title">Số ảnh đã lưu</div>
      <div class="stats" id="stats">
        <div class="stat-row"><span>ok</span><span class="stat-count" id="cnt-ok">0</span></div>
        <div class="stat-row"><span>shift</span><span class="stat-count" id="cnt-shift">0</span></div>
        <div class="stat-row"><span>rotate</span><span class="stat-count" id="cnt-rotate">0</span></div>
        <div class="stat-row"><span>missing</span><span class="stat-count" id="cnt-missing">0</span></div>
        <div class="stat-row"><span>tombstone</span><span class="stat-count" id="cnt-tombstone">0</span></div>
        <div class="stat-row"><span>tilt</span><span class="stat-count" id="cnt-tilt">0</span></div>
        <div class="stat-row"><span>bridge</span><span class="stat-count" id="cnt-bridge">0</span></div>
        <div class="stat-row" style="border-top:1px solid #4ecca3; margin-top:4px; padding-top:4px;"><span><b>Total</b></span><span class="stat-count" id="cnt-total">0</span></div>
      </div>
    </div>

    <div class="controls">
      <div class="section-title">Thao tác</div>
      <div class="btn" onclick="undoLast()">↩ Undo ảnh cuối (U)</div>
      <div class="btn danger" onclick="clearCanvas()">✕ Xóa vùng chọn (Esc)</div>
    </div>

    <div class="info-box">
      <b>Mẹo về Context margin:</b><br>
      Kéo khung sát component → context margin tự lấy rộng ra để thấy <b>pad xung quanh</b>. Model nhờ đó phân biệt được component nằm-ngang-đúng-chỗ (OK) vs lệch-khỏi-pad (shift).<br><br>
      <b>Phím tắt:</b><br>O=ok · S=shift · R=rotate<br>M=missing · T=tombstone · I=tilt<br>B=bridge · W=wrong · U=undo · Esc=cancel
    </div>
  </div>

  <div class="canvas-area" id="canvasArea">
    <div class="no-image" id="noImage"><div class="icon">🖼️</div>Mở ảnh PCB để bắt đầu label</div>
    <canvas id="canvas" style="display:none"></canvas>

    <div id="overlay" style="display:none;">
      <div class="save-popup">
        <h3>Chọn label cho component</h3>
        <div class="sub" id="cropInfo"></div>
        <img id="cropPreview" class="preview" src="">
        <div class="label-buttons">
          <div class="lbl-btn" onclick="saveWithLabel('ok')">✅ OK</div>
          <div class="lbl-btn" onclick="saveWithLabel('shift')">↔ Shift</div>
          <div class="lbl-btn" onclick="saveWithLabel('rotate')">⟳ Rotate</div>
          <div class="lbl-btn" onclick="saveWithLabel('missing')">❌ Missing</div>
          <div class="lbl-btn" onclick="saveWithLabel('tombstone')">⬆ Tombstone</div>
          <div class="lbl-btn" onclick="saveWithLabel('tilt')">⤡ Tilt</div>
          <div class="lbl-btn" onclick="saveWithLabel('bridge')">🌉 Bridge</div>
        </div>
        <button class="cancel-btn" onclick="cancelCrop()">Hủy (Esc)</button>
      </div>
    </div>
  </div>
</div>

<div class="status-bar" id="statusBar">Mở ảnh PCB để bắt đầu...</div>
<div class="toast" id="toast"></div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let img = null;
let isDrawing = false;
let startX, startY, endX, endY;
let currentCrop = null;
let selectedClass = null;
let counters = {};
let lastSaved = null;
let scale = 1;        // current display scale
let fitScale = 1;     // scale that fits screen

document.getElementById('marginSlider').addEventListener('input', e => {
  document.getElementById('marginVal').textContent = e.target.value + '%';
});

document.getElementById('fileInput').addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  document.getElementById('filename').textContent = file.name;
  const reader = new FileReader();
  reader.onload = ev => {
    img = new Image();
    img.onload = () => { fitZoom(); document.getElementById('noImage').style.display='none'; canvas.style.display='block'; };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
});

function fitZoom() {
  if (!img) return;
  const maxW = document.getElementById('canvasArea').clientWidth - 40;
  const maxH = document.getElementById('canvasArea').clientHeight - 40;
  fitScale = Math.min(1, maxW / img.width, maxH / img.height);
  scale = fitScale;
  applyZoom();
}
function zoomBy(f) { if(!img) return; scale = Math.max(0.1, Math.min(8, scale*f)); applyZoom(); }
function applyZoom() {
  canvas.width = img.width * scale;
  canvas.height = img.height * scale;
  redraw();
  setStatus(`Ảnh: ${img.width}x${img.height}px | Zoom: ${Math.round(scale*100)}% | Kéo để chọn component`);
}

canvas.addEventListener('mousedown', e => {
  if (!img) return;
  isDrawing = true;
  const r = canvas.getBoundingClientRect();
  startX = e.clientX - r.left; startY = e.clientY - r.top;
  endX = startX; endY = startY;
});
canvas.addEventListener('mousemove', e => {
  if (!isDrawing) return;
  const r = canvas.getBoundingClientRect();
  endX = e.clientX - r.left; endY = e.clientY - r.top;
  redraw();
  // selection box (vùng bạn kéo)
  ctx.strokeStyle = '#4ecca3'; ctx.lineWidth = 2; ctx.setLineDash([5,3]);
  ctx.strokeRect(startX, startY, endX-startX, endY-startY);
  ctx.fillStyle = 'rgba(78,204,163,0.1)';
  ctx.fillRect(startX, startY, endX-startX, endY-startY);
  // context margin preview (vùng thực sự sẽ crop - màu vàng)
  const box = computeCropBox();
  if (box) {
    ctx.strokeStyle = '#ffd700'; ctx.lineWidth = 1.5; ctx.setLineDash([2,2]);
    ctx.strokeRect(box.dx*scale, box.dy*scale, box.dw*scale, box.dh*scale);
  }
  ctx.setLineDash([]);
});
canvas.addEventListener('mouseup', e => {
  if (!isDrawing) return;
  isDrawing = false;
  if (Math.abs(endX-startX) < 8 || Math.abs(endY-startY) < 8) return;
  const box = computeCropBox();
  if (!box) return;
  const tmp = document.createElement('canvas');
  tmp.width = box.dw; tmp.height = box.dh;
  const tctx = tmp.getContext('2d');
  tctx.fillStyle = '#000'; tctx.fillRect(0,0,box.dw,box.dh); // nền đen nếu crop vượt mép ảnh
  tctx.drawImage(img, box.sx, box.sy, box.sw, box.sh, box.ox, box.oy, box.sw, box.sh);
  currentCrop = tmp.toDataURL('image/jpeg', 0.95);

  document.getElementById('cropInfo').textContent =
    `Crop ${Math.round(box.dw)}×${Math.round(box.dh)}px · context +${document.getElementById('marginSlider').value}%` +
    (document.getElementById('squareToggle').checked ? ' · vuông' : '');

  if (selectedClass) { saveWithLabel(selectedClass); return; }
  document.getElementById('cropPreview').src = currentCrop;
  document.getElementById('overlay').style.display = 'flex';
  setStatus('Chọn label cho component...');
});

// Tính vùng crop thực tế: vùng chọn + context margin, optionally vuông
function computeCropBox() {
  if (!img) return null;
  // vùng chọn theo toạ độ ảnh gốc
  let x0 = Math.min(startX,endX)/scale, y0 = Math.min(startY,endY)/scale;
  let w = Math.abs(endX-startX)/scale, h = Math.abs(endY-startY)/scale;
  if (w < 1 || h < 1) return null;

  const margin = parseInt(document.getElementById('marginSlider').value)/100;
  // nới rộng theo margin
  let cx = x0 + w/2, cy = y0 + h/2;
  w = w * (1 + margin*2);
  h = h * (1 + margin*2);

  if (document.getElementById('squareToggle').checked) {
    const s = Math.max(w, h); w = s; h = s;
  }
  let dx = cx - w/2, dy = cy - h/2;   // toạ độ crop (có thể âm/vượt mép)

  // phần thực sự nằm trong ảnh (source)
  let sx = Math.max(0, dx), sy = Math.max(0, dy);
  let sx2 = Math.min(img.width, dx+w), sy2 = Math.min(img.height, dy+h);
  let sw = sx2 - sx, sh = sy2 - sy;
  if (sw <= 0 || sh <= 0) return null;
  // offset khi vẽ vào canvas đích (nếu crop vượt mép)
  let ox = sx - dx, oy = sy - dy;
  return { dx, dy, dw:w, dh:h, sx, sy, sw, sh, ox, oy };
}

function selectClass(cls, key) {
  selectedClass = cls;
  document.querySelectorAll('.class-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`.class-btn[data-key="${key}"]`).classList.add('active');
  setStatus(`Class đang chọn: ${cls.toUpperCase()} | Kéo để crop (tự lưu ngay)`);
}

function saveWithLabel(label) {
  if (!currentCrop) return;
  document.getElementById('overlay').style.display = 'none';
  fetch('/save', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({image: currentCrop, label}) })
  .then(r=>r.json()).then(data => {
    lastSaved = {label, filename: data.filename};
    counters[label] = (counters[label]||0)+1;
    updateStats(); showToast(`✅ ${label} #${counters[label]}`);
    setStatus(`Lưu: ${data.filename}`); currentCrop=null; redraw();
  });
}
function cancelCrop(){ document.getElementById('overlay').style.display='none'; currentCrop=null; redraw(); }
function undoLast(){
  if(!lastSaved) return;
  fetch('/undo',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(lastSaved)}).then(r=>r.json()).then(()=>{
    counters[lastSaved.label]=Math.max(0,(counters[lastSaved.label]||1)-1);
    updateStats(); showToast(`↩ undo ${lastSaved.label}`); lastSaved=null;
  });
}
function clearCanvas(){ document.getElementById('overlay').style.display='none'; currentCrop=null; isDrawing=false; redraw(); }
function redraw(){ if(!img) return; ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(img,0,0,canvas.width,canvas.height); }
function updateStats(){
  const cs=['ok','shift','rotate','missing','tombstone','tilt','bridge']; let t=0;
  cs.forEach(c=>{const n=counters[c]||0; const el=document.getElementById('cnt-'+c); if(el)el.textContent=n; t+=n;});
  document.getElementById('cnt-total').textContent=t;
}
fetch('/counts').then(r=>r.json()).then(d=>{counters=d; updateStats();});
function setStatus(m){ document.getElementById('statusBar').textContent=m; }
function showToast(m){ const t=document.getElementById('toast'); t.textContent=m; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),1800); }

document.addEventListener('keydown', e => {
  if (e.target.tagName==='INPUT') return;
  const key=e.key.toLowerCase();
  const km={o:'ok',s:'shift',r:'rotate',m:'missing',t:'tombstone',i:'tilt',b:'bridge'};
  if (document.getElementById('overlay').style.display!=='none'){
    if(key==='escape'){cancelCrop();return;}
    if(km[key]){saveWithLabel(km[key]);return;}
  }
  if(key==='escape'){clearCanvas();return;}
  if(key==='u'){undoLast();return;}
  if(key==='+'||key==='='){zoomBy(1.25);return;}
  if(key==='-'){zoomBy(0.8);return;}
  if(km[key]){selectClass(km[key],key);return;}
});
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif path == '/counts':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(counters).encode())
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(length))
        path = urlparse(self.path).path
        if path == '/save':
            label = body['label']
            img_bytes = base64.b64decode(body['image'].split(',')[1])
            counters[label] = counters.get(label, 0) + 1
            cnt = counters[label]
            # 70/15/15 split: trong mỗi 7 ảnh → 5 train, 1 val, 1 test
            m = cnt % 7
            if m == 0:
                split = 'val'
            elif m == 6:
                split = 'test'
            else:
                split = 'train'
            filename = f"{label}_{cnt:04d}.jpg"
            save_path = OUTPUT_DIR / split / label / filename
            with open(save_path, 'wb') as f:
                f.write(img_bytes)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'filename': str(save_path), 'count': cnt}).encode())
        elif path == '/undo':
            try:
                os.remove(body['filename'])
                lab = body['label']
                if counters.get(lab, 0) > 0: counters[lab] -= 1
                res = {'ok': True}
            except Exception as e:
                res = {'ok': False, 'error': str(e)}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode())


if __name__ == '__main__':
    port = 5000
    server = HTTPServer(('localhost', port), Handler)
    print(f"""
+==========================================+
|      PCB Component Labeling Tool  v2      |
+==========================================+
|  Mo browser: http://localhost:{port}        |
|  Anh luu vao: ./dataset/                 |
|  Moi: context margin + crop vuong + zoom |
|  Dung: Ctrl+C                            |
+==========================================+
    """)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDa dung. Dataset luu tai ./dataset/")
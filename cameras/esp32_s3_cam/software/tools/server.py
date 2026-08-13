#!/usr/bin/env python3
"""Image collection server — receives JPEGs POSTed by the board.

Stdlib only. Saves into dataset/incoming/<X-Filename>, prefixing a
timestamp so re-used board sequence numbers never collide.

Usage: server.py [port]        (default 8077)
"""
import http.server
import os
import pathlib
import sys
import time

# Dataset lives outside the repo (bulk storage). Resolution order:
# DATASET_ROOT env var (set by the containerized make targets), then
# ~/.dusty/config.toml [paths] dataset_root, then this default.
def _dataset_root():
    env = os.environ.get('DATASET_ROOT')
    if env:
        return env
    try:
        from dusty.config import load
        return load(required=False).get('paths', {}).get('dataset_root') \
            or '/hd2/datasets/wavesharecam'
    except Exception:
        return '/hd2/datasets/wavesharecam'


DATASET_ROOT = pathlib.Path(_dataset_root())
OUT = DATASET_ROOT / 'incoming'


def _gray_png(raw, w, h):
    """Encode 8-bit grayscale bytes as PNG (stdlib only)."""
    import struct, zlib
    rows = bytearray()
    for y in range(h):
        rows.append(0)
        rows += raw[y * w:(y + 1) * w]
    def chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data
                + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(bytes(rows), 6))
            + chunk(b'IEND', b''))


FIRMWARE = pathlib.Path(__file__).resolve().parent / 'firmware' / 'person_detection.bin'


def firmware_version(path):
    """Read the version string from the esp_app_desc_t at offset 32."""
    import struct
    with open(path, 'rb') as f:
        f.seek(32)
        desc = f.read(80)
    magic = struct.unpack('<I', desc[:4])[0]
    if magic != 0xABCD5432:
        return None
    return desc[16:48].split(b'\x00')[0].decode()


LIVE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>cam live</title>
<style>
 body { background:#111; color:#eee; font-family:system-ui,sans-serif;
        display:flex; flex-direction:column; align-items:center; gap:14px;
        padding-top:4vh; margin:0; }
 img  { width:min(70vw,480px); image-rendering:pixelated;
        border:6px solid #333; border-radius:8px; }
 img.person { border-color:#2e9e4f; }
 #score { font-size:2rem; font-weight:600; }
 #meta  { color:#888; font-size:0.9rem; }
</style></head><body>
<img id="img" src="/latest.png" alt="live frame">
<div id="score">waiting for samples…</div>
<div id="meta"></div>
<script>
async function tick() {
  try {
    const r = await fetch('/latest.json', {cache:'no-store'});
    if (r.ok) {
      const m = await r.json();
      const p = Math.round(m.person_score * 100);
      document.getElementById('score').textContent =
        (p >= 60 ? 'PERSON ' : 'no person ') + p + '%';
      document.getElementById('img').className = p >= 60 ? 'person' : '';
      document.getElementById('meta').textContent =
        'seq ' + m.seq + ' · ' + m.received + ' · ' + m.device;
      const img = document.getElementById('img');
      img.src = '/latest.png?seq=' + m.seq;
    }
  } catch (e) {}
  setTimeout(tick, 300);
}
tick();
</script></body></html>""".encode("utf-8")


class Handler(http.server.BaseHTTPRequestHandler):
    latest_png = None
    latest_meta = None

    def do_GET(self):
        if self.path == '/' or self.path == '/live':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(LIVE_HTML)))
            self.end_headers()
            self.wfile.write(LIVE_HTML)
        elif self.path.startswith('/latest.png'):
            data = Handler.latest_png
            if not data:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/latest.json':
            import json
            meta = Handler.latest_meta
            if not meta:
                self.send_response(404); self.end_headers(); return
            body = json.dumps(meta).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/firmware/version':
            ver = firmware_version(FIRMWARE) if FIRMWARE.exists() else None
            if not ver:
                self.send_response(404); self.end_headers(); return
            body = ver.encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/firmware.bin':
            if not FIRMWARE.exists():
                self.send_response(404); self.end_headers(); return
            data = FIRMWARE.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            print(f'served firmware {firmware_version(FIRMWARE)} ({len(data)}B)', flush=True)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path == '/sample':
            return self._do_sample()
        if self.path != '/upload':
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get('Content-Length', 0))
        if not 0 < length < 5_000_000:
            self.send_response(400); self.end_headers(); return
        name = pathlib.Path(self.headers.get('X-Filename', 'img.jpg')).name
        data = self.rfile.read(length)
        OUT.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime('%Y%m%d-%H%M%S')
        path = OUT / f'{stamp}-{name}'
        path.write_bytes(data)
        print(f'saved {path} ({length}B)', flush=True)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ok')

    def _do_sample(self):
        """Raw grayscale frame + prediction headers -> PNG + JSON sidecar."""
        import json
        length = int(self.headers.get('Content-Length', 0))
        w, _, h = self.headers.get('X-Size', '96x96').partition('x')
        w, h = int(w), int(h)
        if length != w * h or not 0 < length < 1_000_000:
            self.send_response(400); self.end_headers(); return
        raw = self.rfile.read(length)
        meta = {
            'seq': int(self.headers.get('X-Seq', 0)),
            'device': self.headers.get('X-Device', 'unknown'),
            'person_score': float(self.headers.get('X-Person-Score', -1)),
            'no_person_score': float(self.headers.get('X-No-Person-Score', -1)),
            'width': w, 'height': h,
            'received': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        out = OUT.parent / 'samples'
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{meta['device'][-4:]}-{meta['seq']:06d}"
        png = _gray_png(raw, w, h)
        (out / f'{stem}.png').write_bytes(png)
        (out / f'{stem}.json').write_text(json.dumps(meta, indent=1))
        Handler.latest_png = png
        Handler.latest_meta = meta
        print(f"sample {stem} person={meta['person_score']:.2f}", flush=True)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ok')

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8077
    print(f'collection server on 0.0.0.0:{port} -> {OUT}', flush=True)
    http.server.ThreadingHTTPServer(('', port), Handler).serve_forever()

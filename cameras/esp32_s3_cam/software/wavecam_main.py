# main.py — wavecam sound recorder
# Tiny HTTP server: record from the dual mics (ES7210) to WAV on flash,
# list/play/delete recordings from a browser page. Runs after boot.py
# (Wi-Fi + WebREPL). Pins per LESSONS.md: I2C sda=8 scl=7; I2S sck=11
# ws=12 din=13; MCLK via LEDC PWM on 10 (this build's I2S has no mck).
import gc
import os
import socket
import struct
import time
from machine import Pin, I2C, I2S, PWM

RATE = 16000
RECS = "/recs"
MAX_SEC = 15

i2c = I2C(1, sda=Pin(8), scl=Pin(7), freq=100000)
ES = 0x40


def _wr(reg, val):
    i2c.writeto_mem(ES, reg, bytes([val]))


def _rd(reg):
    return i2c.readfrom_mem(ES, reg, 1)[0]


def _upd(reg, mask, val):
    _wr(reg, (_rd(reg) & (0xFF ^ mask)) | (val & mask))


def es7210_init(gain=12):  # 12 = 34.5dB PGA
    _wr(0x00, 0xFF)
    _wr(0x00, 0x41)
    _wr(0x01, 0x3F)
    _wr(0x09, 0x30)
    _wr(0x0A, 0x30)
    _wr(0x23, 0x2A)
    _wr(0x22, 0x0A)
    _wr(0x20, 0x0A)
    _wr(0x21, 0x2A)
    _upd(0x08, 0x01, 0x00)
    _wr(0x40, 0x43)
    _wr(0x41, 0x70)
    _wr(0x42, 0x70)
    _wr(0x07, 0x20)
    _wr(0x02, 0xC1)
    _wr(0x04, 0x01)
    _wr(0x05, 0x00)
    _wr(0x11, (_rd(0x11) & 0x1F) | 0x60)
    _wr(0x11, _rd(0x11) & 0xFC)
    for i in range(4):
        _upd(0x43 + i, 0x10, 0x00)
    _wr(0x4B, 0xFF)
    _wr(0x4C, 0xFF)
    for greg in (0x43, 0x44):
        _upd(0x01, 0x0B, 0x00)
        _wr(0x4B, 0x00)
        _upd(greg, 0x10, 0x10)
        _upd(greg, 0x0F, gain)
    _wr(0x12, 0x00)
    _wr(0x06, 0x00)
    _wr(0x40, 0x43)
    for preg in (0x47, 0x48, 0x49, 0x4A):
        _wr(preg, 0x08)


def wav_header(nbytes):
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + nbytes, b"WAVE",
                       b"fmt ", 16, 1, 2, RATE, RATE * 4, 4, 16,
                       b"data", nbytes)


def record(path, sec):
    mclk = PWM(Pin(10), freq=RATE * 256)
    mclk.duty_u16(32768)
    es7210_init()
    audio = I2S(0, sck=Pin(11), ws=Pin(12), sd=Pin(13),
                mode=I2S.RX, bits=16, format=I2S.STEREO, rate=RATE,
                ibuf=32000)
    try:
        nbytes = sec * RATE * 4
        buf = bytearray(nbytes)
        mv = memoryview(buf)
        junk = bytearray(6400)
        audio.readinto(junk)  # discard codec settling
        i = 0
        while i < nbytes:
            i += audio.readinto(mv[i:min(i + 6400, nbytes)])
    finally:
        audio.deinit()
        mclk.deinit()
    with open(path, "wb") as f:
        f.write(wav_header(nbytes))
        for j in range(0, nbytes, 8192):
            f.write(mv[j:j + 8192])
    del mv, buf
    gc.collect()


def next_name():
    n = 0
    for f in os.listdir(RECS):
        if f.startswith("rec") and f.endswith(".wav"):
            try:
                n = max(n, int(f[3:-4]))
            except ValueError:
                pass
    return "rec%03d.wav" % (n + 1)


def safe(name):
    return name and "/" not in name and ".." not in name


PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>wavecam recorder</title><style>
body{font-family:system-ui,sans-serif;max-width:480px;margin:2em auto;
padding:0 1em;background:#111;color:#eee}
h1{font-size:1.3em}button{font-size:1.1em;padding:.6em 1.4em;border:0;
border-radius:8px;background:#c33;color:#fff;cursor:pointer}
button:disabled{background:#666}select{font-size:1.1em;padding:.5em;
border-radius:8px}li{margin:.8em 0;list-style:none}ul{padding:0}
audio{width:100%%;margin-top:.3em}a{color:#f88}#st{margin-left:1em;color:#8f8}
.row{display:flex;align-items:center;gap:.6em}</style></head><body>
<h1>&#127908; wavecam recorder</h1>
<div class="row"><button id="rec" onclick="go()">Record</button>
<select id="sec"><option>3</option><option selected>5</option>
<option>10</option><option>15</option></select><span>sec</span>
<span id="st"></span></div>
<ul>%s</ul>
<script>
async function go(){
  let b=document.getElementById('rec'),s=document.getElementById('st');
  b.disabled=true;s.textContent='recording...';
  try{await fetch('/record?sec='+document.getElementById('sec').value);
      location.reload();}
  catch(e){s.textContent='error: '+e;b.disabled=false;}
}
</script></body></html>"""


def page_items():
    items = []
    try:
        files = sorted(os.listdir(RECS), reverse=True)
    except OSError:
        files = []
    for f in files:
        if f.endswith(".wav"):
            kb = os.stat(RECS + "/" + f)[6] // 1024
            items.append(
                '<li><div class="row"><b>%s</b> (%dKB) '
                '<a href="/del?f=%s">delete</a></div>'
                '<audio controls preload="none" src="/rec/%s"></audio></li>'
                % (f, kb, f, f))
    return "".join(items) or "<li>no recordings yet</li>"


def send(cl, code, ctype, body):
    cl.send("HTTP/1.0 %s\r\nContent-Type: %s\r\nContent-Length: %d\r\n"
            "Connection: close\r\n\r\n" % (code, ctype, len(body)))
    cl.send(body)


def handle(cl):
    cl.settimeout(5)
    req = cl.recv(1024)
    line = req.split(b"\r\n", 1)[0].decode()
    parts = line.split(" ")
    if len(parts) < 2:
        return
    path = parts[1]
    if path == "/":
        send(cl, "200 OK", "text/html", PAGE % page_items())
    elif path.startswith("/record"):
        sec = 5
        if "sec=" in path:
            try:
                sec = int(path.split("sec=")[1].split("&")[0])
            except ValueError:
                pass
        sec = max(1, min(MAX_SEC, sec))
        name = next_name()
        print("recording %ds -> %s" % (sec, name))
        record(RECS + "/" + name, sec)
        send(cl, "200 OK", "application/json", '{"file":"%s"}' % name)
    elif path.startswith("/rec/"):
        name = path[5:]
        if not safe(name):
            send(cl, "400 Bad Request", "text/plain", "bad name")
            return
        try:
            size = os.stat(RECS + "/" + name)[6]
        except OSError:
            send(cl, "404 Not Found", "text/plain", "no such file")
            return
        cl.send("HTTP/1.0 200 OK\r\nContent-Type: audio/wav\r\n"
                "Content-Length: %d\r\nConnection: close\r\n\r\n" % size)
        with open(RECS + "/" + name, "rb") as f:
            buf = bytearray(4096)
            while True:
                n = f.readinto(buf)
                if not n:
                    break
                cl.send(buf[:n] if n < len(buf) else buf)
    elif path.startswith("/del?f="):
        name = path[7:]
        if safe(name):
            try:
                os.remove(RECS + "/" + name)
            except OSError:
                pass
        cl.send("HTTP/1.0 303 See Other\r\nLocation: /\r\n"
                "Connection: close\r\n\r\n")
    else:
        send(cl, "404 Not Found", "text/plain", "not found")


def main():
    try:
        os.mkdir(RECS)
    except OSError:
        pass
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(socket.getaddrinfo("0.0.0.0", 80)[0][-1])
    s.listen(2)
    s.settimeout(2)  # keep ctrl-C/WebREPL responsive
    print("recorder serving on :80")
    while True:
        try:
            cl, remote = s.accept()
        except OSError:
            continue
        try:
            handle(cl)
        except Exception as e:
            print("request error:", e)
        finally:
            cl.close()


main()

#!/usr/bin/env python3
"""Pump audio logger server.

The wavecam (Waveshare ESP32-S3-CAM) POSTs continuous 8 kHz mono s16le PCM
in 5 s chunks to /ingest. This server:
  - appends audio into per-minute .pcm files under /hd2/pumpaudio/audio/
  - maintains a 1 Hz RMS loudness log per day (u16 slots, 0xFFFF = no data)
  - serves a timeline UI: loudness chart, threshold slider -> pump on/off
    event list, click-to-listen
  - deletes day data older than RETAIN_DAYS

Stdlib only. Runs as a systemd user service on :8090 (pattern: plantlog).
"""
import json
import math
import os
import re
import shutil
import socket
import struct
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

DATA = "/hd2/pumpaudio"
PORT = 8090
MQTT_HOST = "127.0.0.1"     # mosquitto (sensorhub compose stack)
MQTT_PORT = 1883
MQTT_TOPIC = "home/pump/rms"
MQTT_USER = os.environ.get("MQTT_USER", "pump")
# from sensorhub/.env (EnvironmentFile= in pumpaudio.service)
MQTT_PASS = os.environ.get("MQTT_PASS") or os.environ.get("MQTT_PASS_PUMP")
RATE = 8000                 # stored/served format: 8 kHz mono s16le
RATE_IN = 16000             # wire format from the board: 16 kHz STEREO s16le
CHUNK_SEC = 5
CHUNK_BYTES = RATE_IN * 2 * 2 * CHUNK_SEC
RETAIN_DAYS = 30
NODATA = 0xFFFF

lock = threading.Lock()
# per-boot stream time base: boot_id -> (seq0, t0)  [t(seq) = t0 + (seq-seq0)*CHUNK_SEC]
streams = {}
max_seen = {}   # boot_id -> highest seq seen


def day_dir(date):
    return os.path.join(DATA, "audio", date)


def level_path(date):
    return os.path.join(DATA, "levels", date + ".u16")


def ensure_level_file(date):
    p = level_path(date)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(struct.pack("<H", NODATA) * 86400)
    return p


def write_level(date, sec_of_day, rms):
    p = ensure_level_file(date)
    with open(p, "r+b") as f:
        f.seek(sec_of_day * 2)
        f.write(struct.pack("<H", min(rms, NODATA - 1)))


SEC_BYTES = RATE * 2
MINUTE_BYTES = 60 * SEC_BYTES


def append_audio(date, hhmm, sec_in_min, pcm_sec):
    """Write the second at its wall-clock position in a fixed 60 s minute file
    (missing seconds stay silent) so player timestamps align with real time.
    Legacy compact files (pre-2026-08-09-evening) keep append behavior."""
    d = day_dir(date)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, hhmm + ".pcm")
    if not os.path.exists(p):
        with open(p, "wb") as f:
            f.write(b"\x00" * MINUTE_BYTES)
    if os.path.getsize(p) == MINUTE_BYTES:
        with open(p, "r+b") as f:
            f.seek(sec_in_min * SEC_BYTES)
            f.write(pcm_sec)
    else:                       # legacy compact file mid-minute
        with open(p, "ab") as f:
            f.write(pcm_sec)


MARKS_PATH = os.path.join(DATA, "marks.json")


def load_marks():
    """Reference sound marks: [{id, date, t0, t1, label, ts}] — user-tagged
    clip ranges (pump-on click, hum, pump-off...) for fingerprint training."""
    try:
        with open(MARKS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def save_marks(marks):
    tmp = MARKS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(marks, f, indent=1)
    os.replace(tmp, MARKS_PATH)


def chunk_time(boot, seq, arrival):
    """Map (boot, seq) to the wall-clock time the chunk STARTED recording."""
    with lock:
        base = streams.get(boot)
        predicted = None
        if base:
            predicted = base[1] + (seq - base[0]) * CHUNK_SEC
        fresh = seq >= max_seen.get(boot, -1)
        if base is None or (fresh and predicted is not None
                            and abs(predicted - (arrival - CHUNK_SEC)) > 15):
            streams[boot] = (seq, arrival - CHUNK_SEC)
            predicted = arrival - CHUNK_SEC
        if fresh:
            max_seen[boot] = seq
        return predicted


def _mqtt_str(s):
    b = s.encode()
    return struct.pack(">H", len(b)) + b


def _mqtt_len(n):
    out = b""
    while True:
        d, n = n % 128, n // 128
        out += bytes([d | (0x80 if n else 0)])
        if not n:
            return out


def mqtt_publish(topic, payload):
    """Minimal MQTT 3.1.1 QoS0 publish, one connection per call. Stdlib only;
    failures are logged and dropped -- audio ingest must never depend on the bus."""
    try:
        s = socket.create_connection((MQTT_HOST, MQTT_PORT), timeout=5)
        try:
            flags = 0xC2 if MQTT_PASS else 0x02   # user+pass+clean session
            var = (_mqtt_str("MQTT") + bytes([0x04, flags]) + b"\x00\x3c"
                   + _mqtt_str("pump-server"))
            if MQTT_PASS:
                var += _mqtt_str(MQTT_USER) + _mqtt_str(MQTT_PASS)
            s.sendall(b"\x10" + _mqtt_len(len(var)) + var)
            ack = s.recv(4)
            if len(ack) < 4 or ack[0] != 0x20 or ack[3] != 0:
                raise OSError("bad CONNACK %r" % ack)
            body = _mqtt_str(topic) + payload.encode()
            s.sendall(b"\x30" + _mqtt_len(len(body)) + body)
            s.sendall(b"\xe0\x00")
        finally:
            s.close()
    except OSError as e:
        print("mqtt publish failed:", e, flush=True)


# minute aggregation of per-second RMS for the bus (1 msg/min, not 1/sec)
minute_agg = {"min": None, "vals": []}


def note_rms(t, rms):
    """Accumulate a second's RMS; when its minute rolls over, return the
    finished (epoch_minute, values) to publish. Call with `lock` held."""
    m = int(t) // 60
    if minute_agg["min"] is None:
        minute_agg["min"] = m
    if m == minute_agg["min"]:
        minute_agg["vals"].append(rms)
        return None
    if m < minute_agg["min"]:   # late replay of an already-closed minute
        return None
    done = (minute_agg["min"], minute_agg["vals"])
    minute_agg["min"], minute_agg["vals"] = m, [rms]
    return done


def downmix(pcm_stereo16k):
    """16 kHz stereo s16le -> 8 kHz mono s16le: left channel, averaged pairs."""
    n = len(pcm_stereo16k) // 2          # total shorts (L R L R ...)
    s = struct.unpack("<%dh" % n, pcm_stereo16k)
    out = [0] * (n // 4)                 # one output per 2 stereo frames
    for i in range(len(out)):
        out[i] = (s[i * 4] + s[i * 4 + 2]) // 2   # avg left of frames 2i, 2i+1
    return struct.pack("<%dh" % len(out), *out)


def ingest(boot, seq, pcm, arrival):
    t0 = chunk_time(boot, seq, arrival)
    # auto-detect wire format by chunk size: 16 kHz stereo (legacy firmware,
    # 64000 B/s -> downmix here) vs 8 kHz mono (small-chunk firmware, 16000 B/s)
    stereo = len(pcm) % (RATE_IN * 2 * 2) == 0 and len(pcm) > 0
    in_sec_bytes = RATE_IN * 2 * 2 if stereo else RATE * 2
    n_sec = len(pcm) // in_sec_bytes
    for k in range(n_sec):
        t = t0 + k
        lt = datetime.fromtimestamp(t)
        date = lt.strftime("%Y-%m-%d")
        sec_of_day = lt.hour * 3600 + lt.minute * 60 + lt.second
        raw = pcm[k * in_sec_bytes:(k + 1) * in_sec_bytes]
        seg = downmix(raw) if stereo else raw
        samples = struct.unpack("<%dh" % RATE, seg)
        rms = int(math.sqrt(sum(v * v for v in samples) / RATE))
        with lock:
            write_level(date, sec_of_day, rms)
            append_audio(date, lt.strftime("%H-%M"), lt.second, seg)
            closed = note_rms(t, rms)
        if closed:
            m, vals = closed
            mqtt_publish(MQTT_TOPIC, json.dumps({
                "v": round(sum(vals) / len(vals), 1), "max": max(vals),
                "n": len(vals), "ts": m * 60}))


def rotate():
    while True:
        cutoff = (datetime.now() - timedelta(days=RETAIN_DAYS)).strftime("%Y-%m-%d")
        for sub, is_dir in (("audio", True), ("levels", False)):
            root = os.path.join(DATA, sub)
            if not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                date = name[:10]
                if re.match(r"\d{4}-\d{2}-\d{2}$", date) and date < cutoff:
                    p = os.path.join(root, name)
                    shutil.rmtree(p) if is_dir else os.remove(p)
                    print("rotated out", p, flush=True)
        time.sleep(6 * 3600)


PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>pump listener</title><style>
body{font-family:system-ui,sans-serif;margin:1.2em;background:#111;color:#eee}
h1{font-size:1.2em}select,input,button{font-size:1em}
#chart{width:100%;height:260px;background:#1a1a1a;border-radius:8px;cursor:crosshair}
.row{display:flex;gap:1em;align-items:center;flex-wrap:wrap;margin:.6em 0}
#events{margin-top:.6em}table{border-collapse:collapse}
td,th{padding:.25em .8em;border-bottom:1px solid #333;text-align:left;vertical-align:top}
#hover{color:#8cf;min-width:14em}audio{width:100%;margin-top:.6em}
.on{color:#8f8}.chip{display:inline-block;background:#2a3a4a;border-radius:9px;
padding:.05em .55em;margin:.1em .15em;font-size:.85em;white-space:nowrap}
.chip b{color:#8cf;font-weight:normal}.fpchip{background:#3d2a52}
.fpchip b{color:#d9f}#search{background:#222;color:#eee;
border:1px solid #444;border-radius:6px;padding:.3em .6em;width:14em}
#zoominfo{color:#fc6}</style></head><body>
<h1>&#128266; pump listener</h1>
<div class="row">
  <select id="day"></select>
  <label>threshold <input type="range" id="thr" min="1" max="2000" value="100">
  <span id="thrv">100</span></label>
  <input id="search" placeholder="search labels (e.g. pump, water)">
  <button id="markbtn" title="drag on chart to tag a reference sound">&#9998; mark</button>
  <button onclick="dlView()" title="download the visible time range as wav (zoom/pan first; max 2h)">&#8681; wav</button>
  <span id="cov"></span>
  <span id="zoominfo"></span>
  <span id="hover"></span>
</div>
<div class="row" id="markbar" style="display:none">
  <span id="markrange" style="color:#d9f"></span>
  <button onclick="saveMark('pump-on')">pump-on</button>
  <button onclick="saveMark('pump-hum')">pump-hum</button>
  <button onclick="saveMark('pump-off')">pump-off</button>
  <input id="marklabel" placeholder="custom label" size="14">
  <button onclick="saveMark()">save</button>
  <button onclick="dlMark()" title="download selection as wav">&#8681; download</button>
  <button onclick="cancelMark()">cancel</button>
</div>
<canvas id="chart" height="260"></canvas>
<audio id="player" controls preload="none"></audio>
<div id="marksdiv"></div>
<div id="events"></div>
<script>
let lv=null, date=null, labelData=null, view=[0,86400], dragStart=null;
let clipInfo=null;   // {map:[wall-clock sec per audio-second]} for the loaded clip
let marks=[], markMode=false, pendingMark=null;
const cv=document.getElementById('chart'), cx=cv.getContext('2d');
const thr=document.getElementById('thr'), thrv=document.getElementById('thrv');
const player=document.getElementById('player');
const search=document.getElementById('search');

async function loadDays(){
  const days=await (await fetch('/days')).json();
  const sel=document.getElementById('day');
  sel.innerHTML=days.map(d=>`<option>${d}</option>`).join('');
  if(days.length){sel.value=days[days.length-1];loadDay(sel.value);}
  sel.onchange=()=>loadDay(sel.value);
}
async function loadDay(d){
  if(d!==date)clipInfo=null;
  date=d; view=[0,86400];
  const buf=await (await fetch('/levels/'+d+'?_='+Date.now())).arrayBuffer();
  lv=new Uint16Array(buf);
  labelData=null;
  try{const r=await fetch('/labels/'+d+'?_='+Date.now());
    if(r.ok)labelData=await r.json();}catch(e){}
  await loadMarks();
  draw();
}
async function loadMarks(){
  try{marks=await (await fetch('/marks?date='+date+'&_='+Date.now())).json();}
  catch(e){marks=[];}
  renderMarks();
}
function xToSec(x){return Math.floor(view[0]+x/cv.width*(view[1]-view[0]));}
function secToX(s){return (s-view[0])/(view[1]-view[0])*cv.width;}
function fmt(s){const h=String(Math.floor(s/3600)).padStart(2,'0'),
  m=String(Math.floor(s%3600/60)).padStart(2,'0'),
  ss=String(Math.floor(s%60)).padStart(2,'0');return h+':'+m+':'+ss;}
function draw(chartOnly){
  if(!lv)return;
  cv.width=cv.clientWidth;
  const W=cv.width,H=cv.height,T=+thr.value,span=view[1]-view[0];
  cx.clearRect(0,0,W,H);
  let maxv=200;
  for(let s=view[0];s<view[1];s++){const v=lv[s];if(v!==65535&&v>maxv)maxv=v;}
  const events=findEvents(T);
  let lastIdx=-1;
  for(let s=86399;s>=0;s--){if(lv[s]!==65535){lastIdx=s;break;}}
  let covered=0;
  for(let s=0;s<=lastIdx;s++)if(lv[s]!==65535)covered++;
  document.getElementById('cov').textContent = lastIdx>=0 ?
    'coverage '+Math.round(covered*100/(lastIdx+1))+'%' : '';
  document.getElementById('zoominfo').textContent =
    span<86400 ? fmt(view[0])+' - '+fmt(view[1])+' (dblclick to reset)'
    : 'scroll=zoom · shift+scroll=pan · drag=select · ✎+drag=mark';
  for(let x=0;x<W;x++){
    const a=Math.floor(view[0]+x/W*span), b=Math.max(a+1,Math.floor(view[0]+(x+1)/W*span));
    let m=-1,n=0;
    for(let s=a;s<b;s++){const v=lv[s];if(v!==65535){n++;if(v>m)m=v;}}
    if(a>lastIdx){cx.fillStyle='#181818';cx.fillRect(x,0,1,H);continue;}
    const frac=n/(b-a);
    if(n===0){cx.fillStyle='#202020';cx.fillRect(x,0,1,H);}
    if(frac===0){cx.fillStyle='#c33';cx.fillRect(x,0,1,5);}
    else if(frac<0.6){cx.fillStyle='#c73';cx.fillRect(x,0,1,5);}
    if(n===0)continue;
    const h=Math.max(1,Math.round(m/maxv*(H-8)));
    cx.fillStyle=m>=T?'#e66':'#4a8';
    cx.fillRect(x,H-h,1,h);
  }
  const ty=H-Math.round(T/maxv*(H-8));
  cx.strokeStyle='#fc6';cx.setLineDash([4,4]);
  cx.beginPath();cx.moveTo(0,ty);cx.lineTo(W,ty);cx.stroke();cx.setLineDash([]);
  // adaptive time ticks
  cx.fillStyle='#777';cx.font='10px sans-serif';
  const step=span>21600?7200:span>7200?1800:span>1800?600:60;
  for(let s=Math.ceil(view[0]/step)*step;s<=view[1];s+=step){
    const x=secToX(s);
    cx.fillRect(x,H-4,1,4);
    cx.fillText(span>21600?fmt(s).slice(0,2):fmt(s).slice(0,5),x+2,H-6);
  }
  for(const m of marks){                       // reference marks: purple bands
    const xa=secToX(m.t0),xb=secToX(m.t1);
    if(xb<0||xa>W)continue;
    cx.fillStyle='rgba(190,130,255,.20)';
    cx.fillRect(xa,0,Math.max(2,xb-xa),H);
    cx.fillStyle='#b7f';
    cx.fillRect(xa,H-5,Math.max(2,xb-xa),5);
    if(xb-xa>30||span<1800){
      cx.fillStyle='#d9f';cx.font='10px sans-serif';
      cx.fillText(m.label,xa+2,26);
    }
  }
  if(pendingMark){
    const xa=secToX(pendingMark.t0),xb=secToX(pendingMark.t1);
    cx.fillStyle='rgba(255,180,80,.25)';
    cx.fillRect(xa,0,Math.max(2,xb-xa),H);
  }
  if(dragStart!==null&&dragCur!==null){
    cx.fillStyle=markMode?'rgba(255,180,80,.25)':'rgba(140,200,255,.15)';
    const x0=Math.min(dragStart,dragCur),x1=Math.max(dragStart,dragCur);
    cx.fillRect(x0,0,x1-x0,H);
  }
  if(clipInfo&&clipInfo.map.length){
    // loaded-clip extent band along the top
    const m=clipInfo.map;
    const xa=secToX(m[0]),xb=secToX(m[m.length-1]+1);
    cx.fillStyle='rgba(140,200,255,.35)';
    cx.fillRect(Math.max(0,xa),0,Math.min(W,xb)-Math.max(0,xa),3);
    // playhead: map audio time -> wall clock (handles gap-skipped clips)
    const t=player.currentTime||0, fi=Math.floor(t),
      i=Math.min(m.length-1,fi), sec=m[i]+(fi===i?t-fi:1);
    const x=secToX(sec);
    if(x>=0&&x<=W){
      cx.strokeStyle='#fff';cx.beginPath();cx.moveTo(x,0);cx.lineTo(x,H);cx.stroke();
      cx.fillStyle='#fff';
      cx.beginPath();cx.moveTo(x-5,0);cx.lineTo(x+5,0);cx.lineTo(x,8);cx.fill();
      cx.fillStyle='#8cf';cx.font='11px sans-serif';
      cx.fillText(fmt(sec),x+7,14);
    }
  }
  if(!chartOnly)renderEvents(events);
}
function findEvents(T){
  const ev=[];let start=-1,last=-1;
  for(let s=0;s<86400;s++){
    const v=lv[s];if(v===65535)continue;
    if(v>=T){if(start<0)start=s;last=s;}
    else if(start>=0&&s-last>5){ev.push([start,last]);start=-1;}
  }
  if(start>=0)ev.push([start,last]);
  return ev.filter(e=>e[1]-e[0]>=3);
}
function labelsFor(a,b){
  if(!labelData)return null;
  const hits=[];
  for(const le of labelData.events)
    if(le.start<=b&&le.end>=a&&le.top)
      for(const [name,p] of le.top)
        if(!hits.some(h=>h[0]===name))hits.push([name,p]);
  hits.sort((x,y)=>y[1]-x[1]);
  return hits.slice(0,4);
}
function fpFor(a,b){
  if(!labelData)return [];
  const hits=[];
  for(const le of labelData.events)
    if(le.start<=b&&le.end>=a&&le.fp)
      for(const [name,p] of le.fp)
        if(!hits.some(h=>h[0]===name))hits.push([name,p]);
  hits.sort((x,y)=>y[1]-x[1]);
  return hits;
}
function fpSegsFor(a,b){
  if(!labelData)return [];
  const segs=[];
  for(const le of labelData.events)
    if(le.start<=b&&le.end>=a&&le.fpseg)
      for(const s of le.fpseg)segs.push(s);
  return segs;
}
function renderEvents(ev){
  const el=document.getElementById('events');
  const q=search.value.trim().toLowerCase();
  let rows=[];
  for(const e of [...ev].reverse()){   // newest first
    const lab=labelsFor(e[0],e[1]);
    const fps=fpFor(e[0],e[1]);
    const labTxt=((lab?lab.map(l=>l[0]).join(' '):'')+' '
      +fps.map(f=>f[0]).join(' ')).toLowerCase();
    if(q&&!labTxt.includes(q))continue;
    const chips=fps.map(f=>
      `<span class="chip fpchip">${f[0]} <b>${f[1].toFixed(2)}</b></span>`).join('')
      +(lab?lab.map(l=>
      `<span class="chip">${l[0]} <b>${Math.round(l[1]*100)}%</b></span>`).join('')
      :'<span style="color:#666">unclassified</span>');
    const pk=peaksOf(e[0],e[1]).map(p=>
      `<a href="#" onclick="listen(${p[1]});return false" title="rms ${p[0]}">${fmt(p[1]).slice(3)}</a>`)
      .join(' ');
    const fseg=fpSegsFor(e[0],e[1]).map(s=>
      `<a href="#" onclick="listenMark(${s[1]},${s[2]-1});return false"
        title="${s[0]} ${s[3]}" style="color:#c9f">${fmt(s[1]).slice(3)} ${s[3].toFixed(2)}</a>`)
      .join(' ');
    rows.push(`<tr><td class="on">${fmt(e[0])}</td><td>${fmt(e[1])}</td>
      <td>${Math.round((e[1]-e[0])/60*10)/10} min</td><td>${chips}</td>
      <td><a href="#" onclick="listen(${e[0]},${e[1]});return false">listen</a>
      <a href="#" onclick="zoomTo(${e[0]},${e[1]});return false">zoom</a><br>
      <span style="color:#888">peaks:</span> ${pk}
      ${fseg?`<br><span style="color:#a7f">fp hits:</span> ${fseg}`:''}</td></tr>`);
  }
  if(!rows.length){el.innerHTML=q?'<p>no events match "'+q+'"</p>'
    :'<p>no events above threshold</p>';return;}
  el.innerHTML='<table><tr><th>on</th><th>off</th><th>dur</th><th>sounds (PANNs)</th>'
    +'<th></th></tr>'+rows.join('')+'</table>';
}
function zoomTo(a,b){
  const pad=Math.max(30,(b-a)*0.5);
  view=[Math.max(0,a-pad),Math.min(86400,b+pad)];draw();
}
let clipUrl=null,listenSeq=0;
async function loadClip(t0,t1,skip){
  const map=[];
  for(let s=t0;s<t1;s++)if(!skip||lv[s]!==65535)map.push(s);
  // fetch whole clip as a blob so the <audio> element is fully seekable
  // (streamed WAV w/o Range support snaps back to 0 on seek)
  const my=++listenSeq;
  const r=await fetch(`/clip/${date}?t0=${t0}&t1=${t1}&skip=${skip?1:0}`);
  const blob=await r.blob();
  if(my!==listenSeq)return;   // a newer click superseded this fetch
  clipInfo={map};
  if(clipUrl)URL.revokeObjectURL(clipUrl);
  clipUrl=URL.createObjectURL(blob);
  player.src=clipUrl;
  player.onloadedmetadata=()=>{player.play();};
  player.load();
}
function listen(a,b){
  if(b===undefined){          // chart/peak click: play onward from that moment
    const t0=Math.max(0,a-1);loadClip(t0,Math.min(86400,t0+120),false);
  }else{                      // event: gap-free playback of the whole event
    const t0=Math.max(0,a-2);loadClip(t0,Math.min(86400,Math.min(b+2,t0+590)),true);
  }
}
function listenMark(a,b){     // tight wall-clock playback of a reference mark
  loadClip(Math.max(0,a-1),Math.min(86400,b+1),false);
}
function seekOrListen(s){
  // click inside the loaded clip = seek (audio-editor style); outside = new clip
  if(clipInfo){
    const idx=clipInfo.map.indexOf(s);
    if(idx>=0){player.currentTime=idx;player.play();return;}
  }
  listen(s);
}
function phLoop(){
  if(!player.paused&&!player.ended){draw(true);requestAnimationFrame(phLoop);}
}
player.addEventListener('play',()=>requestAnimationFrame(phLoop));
player.addEventListener('pause',()=>draw(true));
player.addEventListener('seeked',()=>draw(true));
function peaksOf(a,b){
  const cand=[];
  for(let s=a;s<=b;s++){const v=lv[s];if(v!==65535)cand.push([v,s]);}
  cand.sort((x,y)=>y[0]-x[0]);
  const picks=[];
  for(const [v,s] of cand){
    if(picks.length>=3)break;
    if(picks.every(p=>Math.abs(p[1]-s)>5))picks.push([v,s]);
  }
  return picks.sort((x,y)=>x[1]-y[1]);
}
let dragCur=null;
cv.addEventListener('mousedown',e=>{dragStart=e.offsetX;dragCur=null;});
cv.addEventListener('mousemove',e=>{
  if(dragStart!==null){dragCur=e.offsetX;draw();}
  const s=xToSec(e.offsetX),v=lv?lv[s]:0;
  document.getElementById('hover').textContent=
    fmt(s)+'  rms '+(v===65535?'(no data)':v);});
cv.addEventListener('mouseup',e=>{
  if(markMode&&dragStart!==null&&dragCur!==null&&Math.abs(dragCur-dragStart)>3){
    const a=xToSec(Math.min(dragStart,dragCur)),b=xToSec(Math.max(dragStart,dragCur));
    pendingMark={t0:a,t1:Math.max(b+1,a+1)};
    document.getElementById('markrange').textContent=
      fmt(pendingMark.t0)+' - '+fmt(pendingMark.t1)+' ('+(pendingMark.t1-pendingMark.t0)+'s)';
    document.getElementById('markbar').style.display='flex';
    listenMark(pendingMark.t0,pendingMark.t1-1);   // audition the selection
  }else if(dragStart!==null&&dragCur!==null&&Math.abs(dragCur-dragStart)>12){
    const a=xToSec(Math.min(dragStart,dragCur)),b=xToSec(Math.max(dragStart,dragCur));
    if(b-a>=10)view=[a,b];
  }else if(dragStart!==null){seekOrListen(xToSec(e.offsetX));}
  dragStart=null;dragCur=null;draw();});
const markbtn=document.getElementById('markbtn');
markbtn.onclick=()=>{
  markMode=!markMode;
  markbtn.style.background=markMode?'#a5f':'';
  if(!markMode)cancelMark();
};
async function saveMark(preset){
  const label=preset||document.getElementById('marklabel').value.trim();
  if(!pendingMark||!label)return;
  await fetch('/marks',{method:'POST',
    body:JSON.stringify({date,t0:pendingMark.t0,t1:pendingMark.t1,label})});
  document.getElementById('marklabel').value='';
  cancelMark();
  await loadMarks();draw();
}
function dlRange(t0,t1){
  const hms=s=>fmt(s).replace(/:/g,'');
  const a=document.createElement('a');
  a.href=`/clip/${date}?t0=${t0}&t1=${t1}&skip=1`;
  a.download=`pump_${date}_${hms(t0)}-${hms(t1)}.wav`;
  document.body.appendChild(a);a.click();a.remove();
}
function dlMark(){ if(pendingMark)dlRange(pendingMark.t0,pendingMark.t1); }
function dlView(){
  const t0=Math.max(0,Math.floor(view[0])), t1=Math.min(86400,Math.ceil(view[1]));
  if(t1-t0>7200){
    document.getElementById('hover').textContent='⚠ zoom in to 2h or less to download';
    return;
  }
  dlRange(t0,t1);
}
function cancelMark(){
  pendingMark=null;
  document.getElementById('markbar').style.display='none';
  draw();
}
async function delMark(id){
  await fetch('/marks/delete',{method:'POST',body:JSON.stringify({id})});
  await loadMarks();draw();
}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');}
function renderMarks(){
  const el=document.getElementById('marksdiv');
  if(!marks.length){el.innerHTML='';return;}
  el.innerHTML='<table><tr><th style="color:#b7f">reference marks</th><th>start</th>'
    +'<th>end</th><th>dur</th><th></th></tr>'
    +marks.map(m=>`<tr><td style="color:#d9f">${esc(m.label)}</td>
      <td>${fmt(m.t0)}</td><td>${fmt(m.t1)}</td><td>${m.t1-m.t0}s</td>
      <td><a href="#" onclick="listenMark(${m.t0},${m.t1-1});return false">listen</a>
      <a href="#" onclick="zoomTo(${m.t0},${m.t1});return false">zoom</a>
      <a href="#" onclick="delMark(${m.id});return false" style="color:#f77">del</a>
      </td></tr>`).join('')+'</table>';
}
cv.addEventListener('dblclick',()=>{view=[0,86400];draw();});
cv.addEventListener('wheel',e=>{
  e.preventDefault();
  if(!lv)return;
  const span=view[1]-view[0];
  let a,b;
  if(e.shiftKey||Math.abs(e.deltaX)>Math.abs(e.deltaY)){
    // pan (shift+scroll or trackpad horizontal)
    const d=(e.deltaX||e.deltaY)/cv.width*span;
    a=view[0]+d;b=view[1]+d;
  }else{
    // zoom centered on cursor
    const s=xToSec(e.offsetX), f=e.deltaY>0?1.3:1/1.3;
    const ns=Math.min(86400,Math.max(10,span*f));
    a=s-(s-view[0])*ns/span;b=a+ns;
  }
  if(a<0){b-=a;a=0;}
  if(b>86400){a-=b-86400;b=86400;}
  view=[Math.max(0,Math.round(a)),Math.min(86400,Math.round(b))];
  if(view[1]<=view[0])view[1]=view[0]+1;
  draw();
},{passive:false});
thr.oninput=()=>{thrv.textContent=thr.value;draw();};
search.oninput=()=>draw();
window.onresize=draw;
function localDate(){const d=new Date();
  return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')
    +'-'+String(d.getDate()).padStart(2,'0');}
async function refresh(){
  // refresh data without touching zoom/clip state (works while zoomed in)
  if(date!==localDate()||dragStart!==null)return;
  const buf=await (await fetch('/levels/'+date+'?_='+Date.now())).arrayBuffer();
  lv=new Uint16Array(buf);
  try{const r=await fetch('/labels/'+date+'?_='+Date.now());
    if(r.ok)labelData=await r.json();}catch(e){}
  await loadMarks();
  draw();
}
setInterval(refresh,30000);
loadDays();
</script></body></html>"""


def wav_header(nbytes):
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + nbytes, b"WAVE",
                       b"fmt ", 16, 1, 1, RATE, RATE * 2, 2, 16,
                       b"data", nbytes)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html", PAGE.encode())
        elif path == "/days":
            root = os.path.join(DATA, "levels")
            days = sorted(f[:10] for f in os.listdir(root)) if os.path.isdir(root) else []
            self._send(200, "application/json", json.dumps(days).encode())
        elif path == "/marks":
            q = parse_qs(urlparse(self.path).query)
            marks = load_marks()
            if "date" in q:
                marks = [m for m in marks if m["date"] == q["date"][0]]
            self._send(200, "application/json", json.dumps(marks).encode())
        elif path.startswith("/labels/"):
            date = path[8:]
            p = os.path.join(DATA, "labels", date + ".json")
            if re.match(r"\d{4}-\d{2}-\d{2}$", date) and os.path.exists(p):
                with open(p, "rb") as f:
                    self._send(200, "application/json", f.read())
            else:
                self._send(404, "text/plain", b"no labels")
        elif path.startswith("/levels/"):
            date = path[8:]
            p = level_path(date)
            if re.match(r"\d{4}-\d{2}-\d{2}$", date) and os.path.exists(p):
                with open(p, "rb") as f:
                    self._send(200, "application/octet-stream", f.read())
            else:
                self._send(404, "text/plain", b"no data")
        elif path.startswith("/clip/"):
            # /clip/<date>?t0=SEC&t1=SEC — exact wall-clock audio extraction,
            # gaps as silence; works for both aligned and legacy compact files
            date = path[6:]
            q = parse_qs(urlparse(self.path).query)
            if not re.match(r"\d{4}-\d{2}-\d{2}$", date):
                self._send(404, "text/plain", b"bad date")
                return
            try:
                t0 = max(0, int(q["t0"][0]))
                t1 = min(86400, int(q["t1"][0]))
            except (KeyError, ValueError):
                self._send(400, "text/plain", b"need t0,t1")
                return
            if t1 <= t0 or t1 - t0 > 7200:
                self._send(400, "text/plain", b"bad range (max 7200s)")
                return
            skip_gaps = q.get("skip", ["0"])[0] == "1"
            try:
                with open(level_path(date), "rb") as f:
                    lv = struct.unpack("<86400H", f.read())
            except OSError:
                self._send(404, "text/plain", b"no data")
                return
            silence = b"\x00" * SEC_BYTES
            out = bytearray()
            fcache = {}
            for s in range(t0, t1):
                mkey = s // 60
                if mkey not in fcache:
                    fcache.clear()      # sequential scan: one minute-file live
                    p = os.path.join(day_dir(date),
                                     "%02d-%02d.pcm" % (mkey // 60, mkey % 60))
                    if os.path.exists(p):
                        with open(p, "rb") as f:
                            fcache[mkey] = f.read()
                    else:
                        fcache[mkey] = None
                blob = fcache[mkey]
                sec_pcm = None
                if blob is not None and lv[s] != NODATA:
                    if len(blob) == MINUTE_BYTES:      # aligned format
                        off = (s % 60) * SEC_BYTES
                        sec_pcm = blob[off:off + SEC_BYTES]
                    else:                              # legacy compact format
                        present = [k for k in range(60)
                                   if lv[mkey * 60 + k] != NODATA]
                        try:
                            idx = present.index(s % 60)
                            if (idx + 1) * SEC_BYTES <= len(blob):
                                sec_pcm = blob[idx * SEC_BYTES:(idx + 1) * SEC_BYTES]
                        except ValueError:
                            pass
                if sec_pcm:
                    out += sec_pcm
                elif not skip_gaps:
                    out += silence
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(44 + len(out)))
            self.end_headers()
            self.wfile.write(wav_header(len(out)))
            self.wfile.write(bytes(out))
        elif path.startswith("/audio/"):
            m = re.match(r"/audio/(\d{4}-\d{2}-\d{2})/(\d{2}-\d{2})\.wav$", path)
            if not m:
                self._send(404, "text/plain", b"bad path")
                return
            p = os.path.join(day_dir(m.group(1)), m.group(2) + ".pcm")
            if not os.path.exists(p):
                self._send(404, "text/plain", b"no audio for that minute")
                return
            with open(p, "rb") as f:
                pcm = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(44 + len(pcm)))
            self.end_headers()
            self.wfile.write(wav_header(len(pcm)))
            self.wfile.write(pcm)
        elif path == "/fw":
            # serve staged firmware; one-shot: consume update.flag on fetch so
            # a board that fails mid-install doesn't loop (re-touch to retry)
            fw = os.path.join(DATA, "fw_stage.py")
            flag = os.path.join(DATA, "update.flag")
            if not os.path.exists(fw):
                self._send(404, "text/plain", b"nothing staged")
                return
            with open(fw, "rb") as f:
                body = f.read()
            if os.path.exists(flag):
                os.remove(flag)
            print("serving staged firmware (%d bytes)" % len(body), flush=True)
            self._send(200, "text/plain", body)
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/marks":
            n = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(min(n, 4096)))
                date, label = body["date"], str(body["label"]).strip()[:80]
                t0, t1 = int(body["t0"]), int(body["t1"])
                assert re.match(r"\d{4}-\d{2}-\d{2}$", date)
                assert 0 <= t0 < t1 <= 86400 and label
            except Exception:
                self._send(400, "text/plain", b"need date,t0,t1,label")
                return
            with lock:
                marks = load_marks()
                mid = max([m["id"] for m in marks], default=0) + 1
                marks.append({"id": mid, "date": date, "t0": t0, "t1": t1,
                              "label": label, "ts": int(time.time())})
                save_marks(marks)
            print("mark #%d %s %s [%d,%d)" % (mid, label, date, t0, t1),
                  flush=True)
            self._send(200, "application/json", json.dumps({"id": mid}).encode())
            return
        if u.path == "/marks/delete":
            n = int(self.headers.get("Content-Length", "0"))
            try:
                mid = int(json.loads(self.rfile.read(min(n, 4096)))["id"])
            except Exception:
                self._send(400, "text/plain", b"need id")
                return
            with lock:
                marks = load_marks()
                save_marks([m for m in marks if m["id"] != mid])
            self._send(200, "text/plain", b"ok")
            return
        if u.path == "/scan":
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(min(n, 4096)).decode(errors="replace")
            line = "%s %s\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), body)
            with open(os.path.join(DATA, "scans.log"), "a") as f:
                f.write(line)
            print("scan report:", body, flush=True)
            self._send(200, "text/plain", b"ok")
            return
        if u.path != "/ingest":
            self._send(404, "text/plain", b"not found")
            return
        q = parse_qs(u.query)
        boot = q.get("boot", ["?"])[0]
        seq = int(q.get("seq", ["0"])[0])
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0 or n > CHUNK_BYTES * 2:
            self._send(400, "text/plain", b"bad length")
            return
        pcm = self.rfile.read(n)
        flag = os.path.join(DATA, "reset.flag")
        if os.path.exists(flag):
            os.remove(flag)
            print("ordering board reset", flush=True)
            self._send(418, "text/plain", b"reset")
            return
        try:
            ingest(boot, seq, pcm, time.time())
            # 210 = "new firmware staged, fetch /fw" (board treats it as 200+update)
            if os.path.exists(os.path.join(DATA, "update.flag")):
                self._send(210, "text/plain", b"update available")
            else:
                self._send(200, "text/plain", b"ok")
        except Exception as e:
            print("ingest error:", e, flush=True)
            self._send(500, "text/plain", b"ingest error")


def main():
    os.makedirs(DATA, exist_ok=True)
    threading.Thread(target=rotate, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("pump server on :%d, data in %s" % (PORT, DATA), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

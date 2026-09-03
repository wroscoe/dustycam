#!/usr/bin/env python3
"""casereview - browse the camera enclosure renders and pin comments on them.

A review app for the enclosure work in cameras/*/hardware/case/. It serves the
rendered views, lets you click anywhere on one to drop a numbered pin with a
note, and writes every pin to comments.json beside this file. That JSON is the
point: it lives in the repo next to the CAD source, versions with it, and Claude
reads it directly to see what you marked up.

Stdlib only -- no Flask, nothing to install. Run:

    python3 tools/casereview/serve.py            # http://localhost:8102
    python3 tools/casereview/serve.py --port N
    python3 tools/casereview/serve.py --host 0.0.0.0   # reachable on the tailnet

Renders are discovered, not configured: any cameras/<cam>/hardware/case/renders/
file named <target>.<view>.png shows up automatically, so a new camera's case
appears here as soon as it has renders.
"""
import argparse
import json
import re
import shutil
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                       # .../dustycam
COMMENTS = HERE / "comments.json"
PAGE = HERE / "index.html"

# <target>.<view>.png, e.g. goouuu_cam_case_body.bottom.png
RENDER_RE = re.compile(r"^(?P<target>[A-Za-z0-9_]+)\.(?P<view>[a-z]+)\.png$")

# The deployed pose differs per board, so the same face means different things.
# Spelling it out here keeps the UI honest about what you are looking at.
VIEW_ORDER = ["iso", "front", "bottom", "back", "top", "side"]
VIEW_HELP = {
    "goouuu": {
        "iso": "Three-quarter view",
        "front": "FRONT (deployed) - board +Z, the lid; camera looks out here",
        "bottom": "FLOOR (deployed) - board -X, both USB-C ports open downward",
        "back": "BACK (deployed) - board -Z, the 1/4\"-20 tripod pad",
        "top": "ROOF (deployed) - board +X, the WROOM antenna end",
        "side": "Side - board -Y",
    },
    "waveshare": {
        "iso": "Three-quarter view",
        "front": "FRONT (deployed) - board +Z, the lid; camera looks out here",
        "bottom": "FLOOR (deployed) - board -Y, USB-C + lead slot open downward",
        "back": "BACK (deployed) - board -Z, the 1/4\"-20 tripod pad",
        "top": "ROOF (deployed) - board +Y",
        "side": "Side - board -X",
    },
}
DESIGNS = {
    "goouuu": "GOOUUU ESP32-S3-CAM",
    "wsc": "Waveshare ESP32-S3-CAM-GC0308",
}
DESIGN_KEY = {"goouuu": "goouuu", "wsc": "waveshare"}     # -> VIEW_HELP key

_lock = threading.Lock()


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def design_of(target):
    return "goouuu" if target.startswith("goouuu") else "wsc"


def pretty(target):
    t = target.replace("_cam_case_", " ").replace("wsc", "waveshare")
    return t.replace("goouuu", "GOOUUU").replace("_", " ")


def scan_renders():
    """Every cameras/*/hardware/case/renders/<target>.<view>.png in the repo."""
    found = {}
    for d in sorted((REPO / "cameras").glob("*/hardware/case/renders")):
        for p in sorted(d.glob("*.png")):
            m = RENDER_RE.match(p.name)
            if not m:
                continue
            t, v = m["target"], m["view"]
            found.setdefault(t, {"target": t, "label": pretty(t),
                                 "design": design_of(t), "views": {}})
            found[t]["views"][v] = str(p.relative_to(REPO))
    targets = []
    for t in sorted(found, key=lambda k: (design_of(k), "assembly" in k, k)):
        e = found[t]
        order = [v for v in VIEW_ORDER if v in e["views"]]
        order += [v for v in sorted(e["views"]) if v not in VIEW_ORDER]
        e["order"] = order
        e["help"] = VIEW_HELP.get(DESIGN_KEY[e["design"]], {})
        targets.append(e)
    return targets


def load():
    if not COMMENTS.exists():
        return []
    try:
        data = json.loads(COMMENTS.read_text())
        return data.get("comments", []) if isinstance(data, dict) else data
    except json.JSONDecodeError:
        # Never let a hand-edit typo wipe the file; keep it and start clean.
        shutil.copy(COMMENTS, COMMENTS.with_suffix(".json.bad"))
        return []


def save(comments):
    """Write sorted + indented so the file diffs cleanly in git."""
    comments = sorted(comments, key=lambda c: (c["target"], c["view"], c["created"]))
    for i, c in enumerate(comments, 1):
        c["n"] = i
    payload = {
        "_comment": "Pin comments on the camera-case renders. Written by "
                    "tools/casereview/serve.py; safe to hand-edit or delete.",
        "updated": now(),
        "comments": comments,
    }
    tmp = COMMENTS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(COMMENTS)                    # atomic, so a crash cannot truncate
    return comments


class Handler(BaseHTTPRequestHandler):
    server_version = "casereview"

    def log_message(self, fmt, *a):
        pass                                  # quiet; this is a local tool

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.read_text(), "text/html; charset=utf-8")
        if path == "/api/state":
            return self._send(200, {"targets": scan_renders(),
                                    "comments": load(),
                                    "designs": DESIGNS})
        if path.startswith("/render/"):
            rel = path[len("/render/"):]
            f = (REPO / rel).resolve()
            # containment check: never serve outside the repo
            if not str(f).startswith(str(REPO) + "/") or not f.is_file():
                return self._send(404, {"error": "no such render"})
            if f.suffix.lower() != ".png":
                return self._send(403, {"error": "png only"})
            return self._send(200, f.read_bytes(), "image/png")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or "{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "bad json"})

        with _lock:
            comments = load()
            if path == "/api/comment":
                text = (body.get("text") or "").strip()
                if not text:
                    return self._send(400, {"error": "empty comment"})
                c = {
                    "id": body.get("id") or f"c{int(datetime.now().timestamp()*1000)}",
                    "target": body["target"],
                    "view": body["view"],
                    "x": round(float(body["x"]), 4),
                    "y": round(float(body["y"]), 4),
                    "text": text[:2000],
                    "status": "open",
                    "created": now(),
                }
                comments.append(c)
                return self._send(200, {"comments": save(comments)})

            if path == "/api/update":
                cid = body.get("id")
                for c in comments:
                    if c["id"] == cid:
                        if "text" in body:
                            c["text"] = (body["text"] or "").strip()[:2000]
                        if "status" in body:
                            c["status"] = body["status"]
                        if "x" in body and "y" in body:
                            c["x"] = round(float(body["x"]), 4)
                            c["y"] = round(float(body["y"]), 4)
                        return self._send(200, {"comments": save(comments)})
                return self._send(404, {"error": "no such comment"})

            if path == "/api/delete":
                cid = body.get("id")
                kept = [c for c in comments if c["id"] != cid]
                if len(kept) == len(comments):
                    return self._send(404, {"error": "no such comment"})
                return self._send(200, {"comments": save(kept)})

        return self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8102)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to reach it from the tailnet")
    a = ap.parse_args()
    n = sum(len(t["views"]) for t in scan_renders())
    print(f"casereview: {n} renders, {len(load())} comments -> {COMMENTS}")
    print(f"  http://{'localhost' if a.host == '127.0.0.1' else a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plant logger home server.

Receives sensor readings POSTed by the FeatherS3D, stores them in SQLite,
and serves a dashboard. Stdlib only - no dependencies.

  POST /api/reading   {"device":..., "ts":..., "soil_moist":..., ...}
                      -> {"ok": true, "epoch": <server unix time>}   (board clock sync)
  GET  /api/data?hours=168&limit=5000
  GET  /api/latest
  GET  /               dashboard
"""
import json
import os
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "plantlog.db")
DASH_PATH = os.path.join(BASE, "dashboard.html")
PORT = 8087

NUM_FIELDS = ["soil_moist", "soil_temp", "amb_light", "batt_v", "batt_pct",
              "batt_rate", "rssi", "uptime_s", "seq"]
COLS = ["ts", "received_at", "device"] + NUM_FIELDS + ["vbus"]


def open_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS readings(
        id INTEGER PRIMARY KEY,
        ts INTEGER NOT NULL,
        received_at INTEGER NOT NULL,
        device TEXT,
        soil_moist REAL, soil_temp REAL, amb_light REAL,
        batt_v REAL, batt_pct REAL, batt_rate REAL,
        rssi INTEGER, uptime_s INTEGER, seq INTEGER,
        vbus INTEGER)""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(ts)")
    con.execute("CREATE TABLE IF NOT EXISTS names(device TEXT PRIMARY KEY, name TEXT)")
    con.execute("""CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY,
        ts INTEGER NOT NULL,
        device TEXT NOT NULL,
        text TEXT NOT NULL,
        created_at INTEGER NOT NULL)""")
    return con


class Handler(BaseHTTPRequestHandler):
    server_version = "PlantLog/1.0"

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/name":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                device, name = str(body["device"]), str(body["name"])[:60]
            except (ValueError, KeyError, json.JSONDecodeError):
                return self._send(400, {"error": "bad json"})
            con = open_db()
            with con:
                con.execute("INSERT INTO names(device,name) VALUES(?,?) "
                            "ON CONFLICT(device) DO UPDATE SET name=excluded.name",
                            (device, name))
            con.close()
            return self._send(200, {"ok": True})
        if path == "/api/event":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                device, text = str(body["device"]), str(body["text"]).strip()[:500]
                if not text:
                    raise ValueError
            except (ValueError, KeyError, json.JSONDecodeError):
                return self._send(400, {"error": "bad json"})
            now = int(time.time())
            ts = body.get("ts")
            if not isinstance(ts, (int, float)) or ts <= 0:
                ts = now
            con = open_db()
            with con:
                cur = con.execute("INSERT INTO events(ts,device,text,created_at) VALUES(?,?,?,?)",
                                  (int(ts), device, text, now))
                eid = cur.lastrowid
            con.close()
            return self._send(200, {"ok": True, "id": eid})
        if path == "/api/event_delete":
            try:
                length = int(self.headers.get("Content-Length", 0))
                eid = int(json.loads(self.rfile.read(length))["id"])
            except (ValueError, KeyError, json.JSONDecodeError):
                return self._send(400, {"error": "bad json"})
            con = open_db()
            with con:
                con.execute("DELETE FROM events WHERE id=?", (eid,))
            con.close()
            return self._send(200, {"ok": True})
        if path != "/api/reading":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            rec = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "bad json"})

        now = int(time.time())
        ts = rec.get("ts")
        # trust the board's clock only if it's plausibly synced
        if not isinstance(ts, (int, float)) or abs(ts - now) > 7 * 86400:
            ts = now
        row = [int(ts), now, str(rec.get("device", ""))]
        for f in NUM_FIELDS:
            v = rec.get(f)
            row.append(float(v) if isinstance(v, (int, float)) else None)
        row.append(1 if rec.get("vbus") else 0)

        con = open_db()
        with con:
            con.execute(
                "INSERT INTO readings(%s) VALUES(%s)" % (",".join(COLS), ",".join("?" * len(COLS))),
                row)
        con.close()
        self._send(200, {"ok": True, "epoch": now})

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            try:
                with open(DASH_PATH, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                return self._send(500, {"error": "dashboard.html missing"})
        if url.path == "/api/events":
            q = parse_qs(url.query)
            hours = float(q.get("hours", ["168"])[0])
            since = int(time.time() - hours * 3600) if hours > 0 else 0
            con = open_db()
            cur = con.execute(
                "SELECT id, ts, device, text FROM events WHERE ts >= ? ORDER BY ts ASC LIMIT 2000",
                (since,))
            evs = [dict(zip(["id", "ts", "device", "text"], r)) for r in cur.fetchall()]
            con.close()
            return self._send(200, {"events": evs})
        if url.path == "/api/devices":
            con = open_db()
            cur = con.execute(
                "SELECT %s FROM readings r WHERE id = "
                "(SELECT id FROM readings WHERE device = r.device ORDER BY ts DESC, id DESC LIMIT 1) "
                "GROUP BY device" % ",".join(COLS))
            devices = [dict(zip(COLS, r)) for r in cur.fetchall()]
            named = dict(con.execute("SELECT device, name FROM names").fetchall())
            con.close()
            for d in devices:
                d["name"] = named.get(d["device"])
            return self._send(200, {"devices": devices})
        if url.path == "/api/latest":
            con = open_db()
            cur = con.execute("SELECT %s FROM readings ORDER BY ts DESC LIMIT 1" % ",".join(COLS))
            r = cur.fetchone()
            con.close()
            return self._send(200, dict(zip(COLS, r)) if r else {})
        if url.path == "/api/data":
            q = parse_qs(url.query)
            hours = float(q.get("hours", ["168"])[0])
            limit = int(q.get("limit", ["20000"])[0])
            since = int(time.time() - hours * 3600) if hours > 0 else 0
            con = open_db()
            cur = con.execute(
                "SELECT %s FROM readings WHERE ts >= ? ORDER BY ts ASC LIMIT ?" % ",".join(COLS),
                (since, limit))
            rows = [dict(zip(COLS, r)) for r in cur.fetchall()]
            con.close()
            return self._send(200, {"rows": rows})
        self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass  # keep journal quiet; data is in the DB


if __name__ == "__main__":
    open_db().close()
    print("plantlog server on 0.0.0.0:%d, db=%s" % (PORT, DB_PATH), flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

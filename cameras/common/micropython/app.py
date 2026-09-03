"""app: the live loop, shared by every always-on MicroPython camera.
docs/camera_standard.md §2: Boot, Connect, Announce, then Sense/Watch/
Capture/Judge/Record/Deliver/Report/Serve/Rest. Bundled last.

Board facts and board_sensors() come from the camera's board.py, bundled
first. Deployed as /flash/app.py by the loader (ota_main.py -> app.run(ota.poll));
`poll` is the loader's OTA poll, which control.py retires by taking over
its port, so it is a no-op here.
"""
import gc
import time

import machine
import network
import secrets

from board import *
from config import *
from uplink import *
from spool import *
from otapull import *
from motion import *
from camera import *
from control import *

EPOCH_OFFSET = 946684800 if time.time() < 1_000_000_000 else 0  # 2000 vs 1970 epoch
BOOT_COUNT_FILE = '/flash/boot_count.txt'

def _boot_count():
    n = 0
    try:
        with open(BOOT_COUNT_FILE) as f:
            n = int(f.read().strip() or 0)
    except (OSError, ValueError):
        pass
    n += 1
    try:
        with open(BOOT_COUNT_FILE, 'w') as f:
            f.write(str(n))
    except OSError:
        pass
    return n


def _ip():
    try:
        return network.WLAN(network.STA_IF).ifconfig()[0]
    except OSError:
        return ''


def build_meta(ts, seq, w, h, why, frac, buffered):
    """Standard meta (§4). `ip` is how ota_push.py and the device page find
    the board; `ts` is the server's timestamp of record."""
    return ('{"ts": %d, "seq": %d, "w": %d, "h": %d, "v": "%s", "cfg": %d, "ip": "%s", '
            '"mode": "%s", "why": "%s", "diff": %.5f, "gate": %.4f, "heartbeat": %s, "buffered": %s}'
            % (ts, seq, w, h, APP_VERSION, CFG.get('cfg', 0), _ip(), STATE['mode'], why,
               frac, CFG['diff_min_frac'],
               'true' if why == 'heartbeat' else 'false',
               'true' if buffered else 'false'))


def sensors():
    """Sense stage: the board's readings (board.py: board_sensors()) + RSSI."""
    vals = {}
    bs = globals().get('board_sensors')
    if bs:
        try:
            vals.update(bs())
        except Exception as e:
            print('board_sensors error', repr(e))
    try:
        vals['rssi'] = network.WLAN(network.STA_IF).status('rssi')
    except (OSError, ValueError):
        pass
    return vals


def run(poll=lambda: None):
    # --- Boot
    rolled_back = fw_boot_check(APP_VERSION)
    cfg_init(TUNING)
    boot_count = _boot_count()
    w, h = preview_init()
    gate = MotionGate(w, h)
    wlan = network.WLAN(network.STA_IF)
    sd = sd_ready()
    st = {'sent': 0, 'skipped': 0, 'failed': 0, 'buffered': 0, 'seq': 0,
          'pending': pending_count() if sd else 0,
          'last_upload': 0, 'last_telemetry': 0, 'last_wifi': 0, 'last_config': 0,
          'setup_cfg_done': -1, 'next_why': 'boot'}    # why for the first frame with no reference
    t_boot = time.ticks_ms()
    print('boot %d  v%s  cfg %s  sd %s (%d pending)  tuning %s'
          % (boot_count, APP_VERSION, CFG.get('cfg'), 'ready' if sd else 'none', st['pending'], TUNING))

    def telemetry_vals():
        dmax, dmean = gate.window()
        vals = {'uptime_s': time.ticks_diff(time.ticks_ms(), t_boot) // 1000,
                'mem_free': gc.mem_free(), 'boot_count': boot_count,
                'frames_sent': st['sent'], 'frames_skipped': st['skipped'],
                'upload_failures': st['failed'], 'pending_files': st['pending'],
                'diff_max': dmax, 'diff_mean': dmean, 'diff_gate': CFG['diff_min_frac'],
                'mode': MODE_NUM.get(STATE['mode'], 0), 'cfg': CFG.get('cfg', 0)}
        if STATE['mode'] == 'setup':
            vals['focus_score'] = round(STATE['score'], 1)
            vals['focus_best'] = round(STATE['best'], 1)
        vals.update(sensors())
        return vals

    def tick(force=False):
        if force or time.time() - st['last_telemetry'] >= CFG['telemetry_s']:
            st['last_telemetry'] = time.time()
            vals = telemetry_vals()
            body = '{%s}' % ', '.join('"%s": %s' % kv for kv in vals.items())
            ok = post_json('/telemetry/%s' % secrets.DEVICE, body)
            print('telemetry %s  rssi=%s pending=%d' % ('OK' if ok else 'FAIL', vals.get('rssi'), st['pending']))

    def wifi_up():
        if wlan.isconnected():
            return True
        if time.time() - st['last_wifi'] < WIFI_RETRY_S:
            return False
        st['last_wifi'] = time.time()
        try:
            wlan.active(True)
            wlan.connect(secrets.WIFI_SSID, secrets.WIFI_PASS)
            end = time.time() + 8
            while not wlan.isconnected() and time.time() < end:
                time.sleep_ms(300)
        except OSError:
            pass
        return wlan.isconnected()

    def refresh():
        """Serve stage: config pull, then firmware check (may reset)."""
        st['last_config'] = time.time()
        changed = cfg_pull(TUNING)
        fw_check(APP_VERSION)
        return changed

    def idle():
        poll()
        control_poll()

    def deliver(why, frac):
        """Capture -> Record -> Deliver (or spool + drain). True if delivered."""
        ts = time.time() + EPOCH_OFFSET
        st['seq'] += 1
        data, cw, ch = capture()              # aliases the frame buffer until restore_preview()
        nbytes = len(data)
        delivered = spooled = False
        try:
            if wifi_up():
                if st['pending']:
                    # backlog first: this frame joins the spool, then everything drains in order
                    meta = build_meta(ts, st['seq'], cw, ch, why, frac, True)
                    if sd and st['pending'] < MAX_PENDING and buffer_frame(data, meta):
                        st['pending'] += 1
                        st['buffered'] += 1
                        spooled = True
                    restore_preview()               # `data` is invalid from here on
                    print('draining %d pending...' % st['pending'])
                    if drain_pending(idle, APP_VERSION):
                        st['sent'] += st['pending']
                        st['pending'] = 0
                        delivered = True
                        print('drain complete')
                        if time.localtime()[0] < 2021:
                            try:
                                import ntptime
                                ntptime.settime()
                            except Exception:
                                pass
                    else:
                        st['pending'] = pending_count()
                        print('drain interrupted, %d left' % st['pending'])
                else:
                    meta = build_meta(ts, st['seq'], cw, ch, why, frac, False)
                    if post_blob('frame', data, meta):
                        st['sent'] += 1
                        delivered = True
            if not delivered:
                st['failed'] += 1
                if spooled:
                    st['last_upload'] = time.time()         # buffered counts for the heartbeat timer
                elif sd and st['pending'] < MAX_PENDING:
                    meta = build_meta(ts, st['seq'], cw, ch, why, frac, True)
                    if buffer_frame(data, meta):
                        st['pending'] += 1
                        st['buffered'] += 1
                        st['last_upload'] = time.time()
            else:
                st['last_upload'] = time.time()
                fw_mark_valid()                              # a pending install proved itself
        finally:
            restore_preview()
        print('%s  %s %dx%d %dkB diff=%.4f  sent=%d skipped=%d failed=%d pending=%d' %
              ('upload OK' if delivered else 'offline->sd', why, cw, ch, nbytes // 1024, frac,
               st['sent'], st['skipped'], st['failed'], st['pending']))
        return delivered

    def shoot():
        restore_preview()
        ok = deliver('manual', 0.0)
        gate.reset()
        return ok

    HOOKS.update({'tick': tick, 'shoot': shoot, 'refresh': refresh, 'sensors': sensors,
                  'status': lambda: {'version': APP_VERSION, 'seq': st['seq'], 'sent': st['sent'],
                                     'pending': st['pending'], 'failed': st['failed'], 'sd': sd,
                                     'rolled_back': rolled_back or '', 'fw_pending': fw_pending(),
                                     'last_error': LAST_ERROR[0], 'last_capture': LAST_CAPTURE[0]}})
    control_init(BUTTON_NAMES, LED_NAME)

    # --- Connect / Announce / first Serve
    wifi_up()
    tick(force=True)
    refresh()

    # --- the loop
    while True:
        t0 = time.ticks_ms()
        try:
            tick()
            import sensor
            img = sensor.snapshot()
            had_ref = gate.have_ref
            frac = gate.diff(img, CFG['diff_l_thresh'])
            heartbeat = time.time() - st['last_upload'] >= CFG['heartbeat_s']
            if frac >= CFG['diff_min_frac'] or heartbeat:
                gate.commit(img)
                if not had_ref:
                    why = st['next_why']          # boot, or the confirmation frame after setup
                elif frac >= CFG['diff_min_frac']:
                    why = 'motion'
                else:
                    why = 'heartbeat'
                st['next_why'] = 'manual'
                deliver(why, frac)
            else:
                st['skipped'] += 1
                print('skip  diff=%.5f  sent=%d skipped=%d pending=%d' %
                      (frac, st['sent'], st['skipped'], st['pending']))
            if time.time() - st['last_config'] >= CFG.get('config_s', CFG['heartbeat_s']):
                refresh()
            if CFG.get('mode') == 'setup' and st['setup_cfg_done'] != CFG.get('cfg'):
                st['setup_cfg_done'] = CFG.get('cfg')
                setup_session(None, CFG['setup_secs'], 'config')
                gate.reset()
                st['last_upload'] = time.time()
        except Exception as e:
            st['failed'] += 1
            print('loop error:', repr(e))
            try:
                restore_preview()
            except Exception as e2:
                print('restore failed:', repr(e2))
        # --- Rest: keep the preview fresh and the control plane responsive
        while time.ticks_diff(time.ticks_ms(), t0) < CFG['period_s'] * 1000:
            import sensor
            sensor.snapshot()
            poll()
            if control_poll():
                gate.reset()                   # scene/exposure changed: no false motion
                st['last_upload'] = time.time()
                t0 = time.ticks_ms()
            time.sleep_ms(100)

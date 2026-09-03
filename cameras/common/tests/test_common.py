"""Pure-logic tests for cameras/common/micropython (no board)."""
import re
import subprocess
import sys
from pathlib import Path

import config
import spool
import focus
import control

COMMON = Path(__file__).resolve().parents[1] / 'micropython'
CAMERAS = Path(__file__).resolve().parents[2]

DEFAULTS = {'period_s': 10, 'diff_min_frac': 0.04, 'heartbeat_s': 300, 'capture_framesize': 'WQXGA2',
            'setup_secs': 300}


def test_cfg_init_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'CFG_FILE', str(tmp_path / 'none.json'))
    cfg = config.cfg_init(DEFAULTS)
    assert cfg['period_s'] == 10 and cfg['cfg'] == 0 and cfg['mode'] == 'live'


def test_cfg_apply_coerces_and_ignores_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'CFG_FILE', str(tmp_path / 'none.json'))
    config.cfg_init(DEFAULTS)
    changed = config.cfg_apply({'cfg': '3', 'period_s': '15', 'diff_min_frac': 1,
                                'capture_framesize': 'VGA', 'bogus': 1, 'mode': 'setup'}, DEFAULTS)
    assert set(changed) == {'cfg', 'period_s', 'diff_min_frac', 'capture_framesize', 'mode'}
    assert config.CFG['period_s'] == 15 and isinstance(config.CFG['period_s'], int)
    assert config.CFG['diff_min_frac'] == 1.0 and isinstance(config.CFG['diff_min_frac'], float)
    assert 'bogus' not in config.CFG and config.CFG['cfg'] == 3
    assert config.cfg_apply({'period_s': 15}, DEFAULTS) == []      # no change -> nothing reported
    assert config.cfg_apply({'period_s': 'ten'}, DEFAULTS) == []   # bad value -> ignored


def test_cfg_init_reads_stored_file(tmp_path, monkeypatch):
    f = tmp_path / 'config.json'
    f.write_text('{"cfg": 7, "heartbeat_s": 120, "nope": 1}')
    monkeypatch.setattr(config, 'CFG_FILE', str(f))
    cfg = config.cfg_init(DEFAULTS)
    assert cfg['cfg'] == 7 and cfg['heartbeat_s'] == 120 and 'nope' not in cfg


def test_spool_name_is_unique_per_second():
    taken = {'/sdcard/pending/0000001000.jpg', '/sdcard/pending/0000001000_1.jpg'}
    assert spool.spool_name(1000, lambda p: p in taken) == '/sdcard/pending/0000001000_2'
    assert spool.spool_name(1001, lambda p: p in taken) == '/sdcard/pending/0000001001'


def test_status_json_encoder():
    j = control._j({'a': 1, 'b': 2.5, 'c': True, 'd': 'x', 'e': [1, 'y'], 'f': {'g': None}})
    assert j == '{"a": 1, "b": 2.5, "c": true, "d": "x", "e": [1, "y"], "f": {"g": "None"}}'


def test_setup_page_has_watchdog_and_no_token():
    html = control.setup_page({'secs': '120'})
    assert '/stream?secs=120' in html and 'setInterval' in html and 'token' not in html


def test_focus_roi_inside_vga():
    x, y, w, h = focus.FOCUS_ROI
    assert 0 <= x and x + w <= 640 and 0 <= y and y + h <= 480


def test_bundle_rt1062_compiles_and_strips_imports(tmp_path):
    out = tmp_path / 'app.py'
    subprocess.run([sys.executable, str(COMMON / 'bundle.py'), str(CAMERAS / 'openmv_rt1062'), '-o', str(out)],
                   check=True, capture_output=True)
    text = out.read_text()
    compile(text, 'app.py', 'exec')
    assert not re.search(r'^from (uplink|spool|config|otapull|motion|camera|focus|control|board) import \*', text, re.M)
    assert 'import secrets' in text and 'import sensor' in text
    assert text.index("APP_VERSION = ") < text.index('def post_blob(') < text.index('def run(')
    assert 'def board_sensors' in text

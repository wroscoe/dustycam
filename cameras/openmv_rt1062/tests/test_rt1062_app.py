"""Host tests for the RT1062 app: meta builder and trigger vocabulary.
Runs the shared stubs from cameras/common/tests/conftest.py."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'common' / 'tests'))
import boardstubs  # noqa: E402  (installs the board stubs)
sys.path.insert(0, str(HERE.parent / 'software' / 'app'))
sys.path.insert(0, str(HERE.parents[1] / 'common' / 'micropython'))

for _m in ('board', 'app'):            # per-camera modules: never reuse another camera's
    sys.modules.pop(_m, None)
import board  # noqa: E402
import config  # noqa: E402
import app  # noqa: E402

STANDARD_META = {'ts', 'seq', 'w', 'h', 'v', 'cfg', 'ip', 'mode', 'why', 'diff', 'gate', 'heartbeat', 'buffered'}


def test_meta_has_every_standard_key(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'CFG_FILE', str(tmp_path / 'c.json'))
    config.cfg_init(board.TUNING)
    m = json.loads(app.build_meta(1700000000, 3, 2592, 1944, 'motion', 0.0521, False))
    assert set(m) == STANDARD_META
    assert m['v'] == board.APP_VERSION and m['ip'] == '10.0.0.5' and m['mode'] == 'live'
    assert m['heartbeat'] is False and m['buffered'] is False and m['seq'] == 3
    hb = json.loads(app.build_meta(1, 4, 640, 480, 'heartbeat', 0.0, True))
    assert hb['heartbeat'] is True and hb['buffered'] is True


def test_why_vocabulary():
    src = (HERE.parents[1] / 'common' / 'micropython' / 'app.py').read_text()
    for why in ('boot', 'motion', 'heartbeat', 'manual'):
        assert "'%s'" % why in src


def test_tuning_keys_match_manifest():
    import tomllib
    manifest = tomllib.loads((HERE.parent / 'camera.toml').read_text())
    assert set(manifest['tuning']) == set(board.TUNING)

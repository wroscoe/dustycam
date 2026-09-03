"""Host tests for the N6 app: board facts, meta builder, tuning keys.
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


def test_board_facts():
    assert board.CAPTURE_MODE == 'rgb565'            # the N6 CSI rejects JPEG output
    assert board.TUNING['capture_framesize'] == 'HD'  # 1280x800 native
    assert board.APP_VERSION.endswith('-n6')


def test_meta_has_every_standard_key(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'CFG_FILE', str(tmp_path / 'c.json'))
    config.cfg_init(board.TUNING)
    m = json.loads(app.build_meta(1700000000, 2, 1280, 800, 'motion', 0.0071, False))
    assert set(m) == STANDARD_META and m['v'] == board.APP_VERSION and m['gate'] == 0.005


def test_tuning_keys_match_manifest():
    import tomllib
    manifest = tomllib.loads((HERE.parent / 'camera.toml').read_text())
    assert set(manifest['tuning']) == set(board.TUNING)

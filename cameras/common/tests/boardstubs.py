"""Host-side stubs so the MicroPython modules import under CPython (import from a test as `boardstubs`).
Board modules (sensor, image, machine, network, secrets) are replaced by
minimal fakes; pure logic (config merge, spool naming, meta building,
bundling) is what the tests exercise."""
import sys
import types
from pathlib import Path

COMMON = Path(__file__).resolve().parents[1] / 'micropython'
sys.path.insert(0, str(COMMON))


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _Pin:
    IN = 0
    PULL_UP = 1
    board = types.SimpleNamespace(SW=1, CHG=2)

    def __init__(self, *a, **k):
        pass

    def value(self):
        return 1


class _Img:
    def __init__(self, *a, **k):
        pass


_stub('sensor', RGB565=1, GRAYSCALE=2, JPEG=3, VGA=4, WQXGA2=5,
      reset=lambda: None, set_pixformat=lambda *a: None, set_framesize=lambda *a: None,
      skip_frames=lambda **k: None, width=lambda: 640, height=lambda: 480, snapshot=lambda: _Img())
_stub('image', Image=_Img)
_stub('machine', Pin=_Pin, LED=lambda *a: None, reset=lambda: None)
_stub('network', STA_IF=0, WLAN=lambda *a: types.SimpleNamespace(
    ifconfig=lambda: ('10.0.0.5',), isconnected=lambda: False, status=lambda *a: -50))
_stub('secrets', DEVICE='testcam', SERVER_HOST='127.0.0.1', SERVER_PORT=1, SERVER_TLS=False,
      BLOB_TOKEN='t', OTA_PORT=8266, OTA_TOKEN='x', WIFI_SSID='s', WIFI_PASS='p')

# time.ticks_* exist only on MicroPython
import time as _time
if not hasattr(_time, 'ticks_ms'):
    _time.ticks_ms = lambda: int(_time.time() * 1000)
    _time.ticks_diff = lambda a, b: a - b
    _time.sleep_ms = lambda ms: None

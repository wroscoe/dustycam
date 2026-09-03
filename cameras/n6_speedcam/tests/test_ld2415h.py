"""Host-side tests for the LD2415H frame parser and command builders.

Run from the repo root:  python -m pytest cameras/n6_speedcam/tests
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "software"))

import pytest
import ld2415h as R


def test_parse_frame_examples_from_datasheet():
    s = R.parse_frame(b"V+001.8\r\n")
    assert s.value == pytest.approx(1.8) and s.approaching is True
    s = R.parse_frame(b"V-001.9\r\n")
    assert s.value == pytest.approx(1.9) and s.approaching is False
    assert R.parse_frame(b"V+240.0\r\n").value == pytest.approx(240.0)


@pytest.mark.parametrize("bad", [
    b"V+01.3\r\n\r", b"X+012.3\r\n", b"V*012.3\r\n", b"V+0123.\r\n",
    b"V+012,3\r\n", b"V+012.3\n\r", b"V+012.3\r", b"",
])
def test_parse_frame_rejects_malformed(bad):
    assert R.parse_frame(bad) is None


def test_parser_handles_split_and_junk():
    p = R.Parser()
    assert p.feed(b"V+0") == []
    out = p.feed(b"12.3\r\nV-")
    assert [(s.value, s.approaching) for s in out] == [(12.3, True)]
    out = p.feed(b"004.0\r\n\x00\xffgarbageV+100.5\r\n")
    assert [(s.value, s.approaching) for s in out] == [(4.0, False), (100.5, True)]
    assert p.feed(b"") == []


def test_parser_resyncs_on_false_start():
    p = R.Parser()
    # a stray 'V' followed by a real frame
    out = p.feed(b"VV+050.0\r\n")
    assert [s.value for s in out] == [50.0]


def test_parser_bounds_buffer():
    p = R.Parser()
    p.feed(b"V" + b"x" * 200)
    assert len(p._buf) <= 4 * R.FRAME_LEN


def test_commands_match_datasheet_examples():
    # datasheet §6.4: 43 46 01 03 0a 05 0d 0a and 43 46 02 01 02 00 0d 0a
    assert R.cmd_detection(min_kmh=3, angle_deg=10, sensitivity=5) == bytes.fromhex("4346 01 03 0a 05 0d0a")
    assert R.cmd_output(direction=1, rate=2, unit=0) == bytes.fromhex("4346 02 01 02 00 0d0a")
    # defaults from §6.3
    assert R.cmd_detection() == bytes.fromhex("4346 01 01 00 05 0d0a")
    assert R.cmd_output() == bytes.fromhex("4346 02 00 01 00 0d0a")
    assert R.cmd_antivibration(0) == bytes.fromhex("4346 03 00 00 00 0d0a")
    assert R.cmd_trigger(0, 0) == bytes.fromhex("4346 04 00 00 00 0d0a")
    assert R.CMD_READ_SETTINGS == bytes.fromhex("4346 07 00 00 00 00 00 00 00 00 00 00")


@pytest.mark.parametrize("call", [
    lambda: R.cmd_detection(sensitivity=0),
    lambda: R.cmd_detection(sensitivity=16),
    lambda: R.cmd_output(direction=3),
    lambda: R.cmd_output(unit=5),
    lambda: R.cmd_antivibration(0x71),
    lambda: R.cmd_trigger(hold_s=300),
])
def test_commands_reject_out_of_range(call):
    with pytest.raises(ValueError):
        call()


class FakeUART:
    def __init__(self, incoming=b""):
        self.incoming = incoming
        self.written = b""

    def any(self):
        return len(self.incoming)

    def read(self, n):
        data, self.incoming = self.incoming[:n], self.incoming[n:]
        return data

    def write(self, data):
        self.written += data


def test_radar_reads_and_configures_over_fake_uart():
    u = FakeUART(b"V+033.2\r\nV+034.1\r\n")
    r = R.Radar(uart=u)
    s = r.read()
    assert s.value == pytest.approx(34.1) and r.last is s
    assert r.read() is None
    r.configure(min_kmh=5, sensitivity=7, direction=R.Radar.APPROACHING, antivibration=3)
    r.set_trigger(hold_s=2, threshold_kmh=25)
    assert u.written == (R.cmd_detection(5, 0, 7) + R.cmd_output(1, 1, 0)
                         + R.cmd_antivibration(3) + R.cmd_trigger(2, 25))

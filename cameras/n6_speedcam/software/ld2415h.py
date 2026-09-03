"""HLK-LD2415H 24 GHz speed radar — MicroPython driver (OpenMV N6 UART3).

Protocol (hardware/radar/README.md): 9600 8N1, 3.3 V TTL. The radar streams
9-byte ASCII frames ``V+012.3\\r\\n`` while a target is tracked (``+`` coming
toward the radar, ``-`` going away) and sends nothing otherwise, so "no
target" is a timeout, not a message. Settings are ``CF`` + function code +
three parameter bytes + CRLF and persist in the module.

The parser is plain Python with no MicroPython dependencies so the tests in
``../tests/`` run on the host; only ``Radar`` touches ``machine.UART``.

    from ld2415h import Radar
    r = Radar(uart_id=3, tx="P4", rx="P5")        # OpenMV N6: P4 = UART3 TX, P5 = RX
    r.configure(min_kmh=3, sensitivity=5, direction=Radar.BOTH, unit=Radar.KMH)
    r.set_trigger(hold_s=2, threshold_kmh=20)     # PULSE/AA -> carrier -> N6 P11 wake
    while True:
        s = r.read()                              # Speed(value, approaching) or None
        if s: print(s.value, "km/h", "coming" if s.approaching else "going")
"""

try:
    from collections import namedtuple
except ImportError:  # pragma: no cover
    from ucollections import namedtuple  # type: ignore

Speed = namedtuple("Speed", ("value", "approaching", "raw"))

FRAME_LEN = 9
HEAD = b"CF"
TAIL = b"\r\n"


class Parser:
    """Feed it bytes, get ``Speed`` objects. Tolerates junk between frames."""

    def __init__(self):
        self._buf = b""

    def feed(self, data):
        """Append bytes; return a list of complete Speed frames found."""
        out = []
        if not data:
            return out
        self._buf += data
        while True:
            start = self._buf.find(b"V")
            if start < 0:
                self._buf = b""
                break
            if start:
                self._buf = self._buf[start:]
            if len(self._buf) < FRAME_LEN:
                break
            frame = self._buf[:FRAME_LEN]
            speed = parse_frame(frame)
            if speed is None:
                # not a frame after all: drop this 'V' and rescan
                self._buf = self._buf[1:]
                continue
            out.append(speed)
            self._buf = self._buf[FRAME_LEN:]
        # never let garbage grow without bound
        if len(self._buf) > 4 * FRAME_LEN:
            self._buf = self._buf[-FRAME_LEN:]
        return out


def parse_frame(frame):
    """``b'V+012.3\\r\\n'`` -> Speed(12.3, True, frame); None if malformed."""
    if len(frame) != FRAME_LEN or frame[0:1] != b"V" or frame[7:9] != TAIL:
        return None
    sign = frame[1:2]
    if sign not in (b"+", b"-"):
        return None
    body = frame[2:7]                        # b'012.3'
    if body[3:4] != b"." or not (body[:3].isdigit() and body[4:5].isdigit()):
        return None
    value = int(body[:3]) + int(body[4:5]) / 10.0
    return Speed(value, sign == b"+", bytes(frame))


# ---- command builders (pure functions, testable on the host) -------------------------------
def _cmd(fn, p1=0, p2=0, p3=0):
    for p in (fn, p1, p2, p3):
        if not 0 <= p <= 0xFF:
            raise ValueError("parameter out of range: %r" % (p,))
    return HEAD + bytes((fn, p1, p2, p3)) + TAIL


def cmd_detection(min_kmh=1, angle_deg=0, sensitivity=5):
    """0x01: minimum reported speed, beam/road angle compensation, sensitivity 1..15."""
    if not 1 <= sensitivity <= 0x0F:
        raise ValueError("sensitivity is 1..15")
    return _cmd(0x01, min_kmh, angle_deg, sensitivity)


def cmd_output(direction=0, rate=1, unit=0):
    """0x02: direction 0 both / 1 approaching / 2 receding; rate 0 (~22 fps) .. n; unit 0 km/h 1 mph 2 m/s."""
    if direction not in (0, 1, 2) or unit not in (0, 1, 2):
        raise ValueError("direction and unit are 0..2")
    return _cmd(0x02, direction, rate, unit)


def cmd_antivibration(coeff=0):
    """0x03: 0x00..0x70; bigger ignores more small back-and-forth motion (and misses short moves)."""
    if not 0 <= coeff <= 0x70:
        raise ValueError("coefficient is 0..0x70")
    return _cmd(0x03, coeff)


def cmd_trigger(hold_s=0, threshold_kmh=0):
    """0x04: PULSE/AA output hold time (s) and the speed that fires it; 0 disables."""
    return _cmd(0x04, hold_s, threshold_kmh)


CMD_READ_SETTINGS = HEAD + bytes((0x07,)) + bytes(10)  # 13-byte form, standard mode only


class Radar:
    """The radar on a MicroPython UART. Construct with the OpenMV pin names."""
    BOTH, APPROACHING, RECEDING = 0, 1, 2
    KMH, MPH, MPS = 0, 1, 2

    def __init__(self, uart_id=3, tx="P4", rx="P5", baud=9600, uart=None):
        if uart is None:
            from machine import UART, Pin  # MicroPython only
            uart = UART(uart_id, baud, tx=Pin(tx), rx=Pin(rx), timeout=0)
        self.uart = uart
        self.parser = Parser()
        self.last = None            # most recent Speed
        self.last_ms = None         # ticks_ms when it arrived

    # -- reading ------------------------------------------------------------------------
    def poll(self):
        """Drain the UART; return every new Speed (possibly several per call)."""
        n = self.uart.any()
        if not n:
            return []
        frames = self.parser.feed(self.uart.read(n))
        if frames:
            self.last = frames[-1]
            try:
                import time
                self.last_ms = time.ticks_ms()
            except (ImportError, AttributeError):  # host
                self.last_ms = None
        return frames

    def read(self):
        """Newest Speed since the last call, or None."""
        frames = self.poll()
        return frames[-1] if frames else None

    def idle_ms(self, now_ms):
        """Milliseconds since the last frame; None if never seen one."""
        if self.last_ms is None:
            return None
        import time
        return time.ticks_diff(now_ms, self.last_ms)

    # -- configuring --------------------------------------------------------------------
    def send(self, cmd):
        self.uart.write(cmd)

    def configure(self, min_kmh=1, angle_deg=0, sensitivity=5,
                  direction=0, rate=1, unit=0, antivibration=None):
        self.send(cmd_detection(min_kmh, angle_deg, sensitivity))
        self.send(cmd_output(direction, rate, unit))
        if antivibration is not None:
            self.send(cmd_antivibration(antivibration))

    def set_trigger(self, hold_s, threshold_kmh):
        self.send(cmd_trigger(hold_s, threshold_kmh))

    def read_settings(self, wait_ms=300):
        """Standard-mode settings dump as bytes (``No.: 20210726 v3.0 X1:.. ...``)."""
        self.send(CMD_READ_SETTINGS)
        import time
        time.sleep_ms(wait_ms)
        n = self.uart.any()
        return self.uart.read(n) if n else b""

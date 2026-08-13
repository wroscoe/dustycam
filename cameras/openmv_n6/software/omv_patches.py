"""Bug-fix monkeypatches for the `openmv` PyPI package (v1.0.7).

Fix 1: Transport.recv_packet resets its timeout clock whenever an event
packet arrives. The camera emits frame-ready events continuously while
streaming, so a lost command response makes recv_packet spin forever.
Patched version: events no longer reset the clock, so the configured
timeout is honored and the library's resync/retry logic can kick in.

Fix 2: Camera._send_cmd_wait_resp calls sys.exit(0) on any unexpected
exception (e.g. a serial glitch), silently killing the host process.
Patched version raises instead, so callers can reconnect.

Import this module once, before creating Camera objects.
"""

import struct
import time

from openmv import camera as _camera
from openmv import transport as _transport
from openmv.camera import Camera, Opcode
from openmv.constants import Flags, Status
from openmv.exceptions import (
    ChecksumException,
    OMVException,
    ResyncException,
    SequenceException,
    TimeoutException,
)
from openmv.transport import Transport


def _recv_packet(self, poll_events=False):
    if not self.serial or not self.serial.is_open:
        raise OMVException("Serial connection not open")

    fragments = bytearray()
    start_time = time.time()

    while time.time() - start_time < self.timeout:
        if self.serial.in_waiting > 0:
            self.buf.extend(self.serial.read(self.serial.in_waiting))

        if not (packet := self._process()):
            if poll_events:
                return
            time.sleep(0.001)
            continue

        self.stats['received'] += 1
        self.log(packet=packet, direction="Recv")

        if (packet['flags'] & Flags.RTX) and (self.sequence != packet['sequence']):
            if packet['flags'] & Flags.ACK_REQ:
                self.send_packet(packet['opcode'], packet['channel'],
                                 Flags.ACK, sequence=packet['sequence'])
            continue

        if packet['flags'] & Flags.ACK_REQ:
            self.send_packet(packet['opcode'], packet['channel'], Flags.ACK)

        if packet['flags'] & Flags.EVENT:
            self.event_callback(packet['channel'], 0xFFFF if not packet['length']
                                else struct.unpack('<H', packet['payload'])[0])
            # PATCH: do NOT reset start_time here
            continue

        self.sequence = (self.sequence + 1) & 0xFF

        if packet['flags'] & Flags.FRAGMENT:
            fragments.extend(packet['payload'])
            start_time = time.time()  # fragments legitimately extend the wait
            continue

        if packet['flags'] & Flags.NAK:
            status = struct.unpack('<H', packet['payload'][:2])[0]
            if status == Status.CHECKSUM:
                raise ChecksumException("")
            elif status == Status.SEQUENCE:
                raise SequenceException("")
            elif status == Status.TIMEOUT:
                raise TimeoutException("")
            elif status != Status.BUSY:
                raise OMVException(f"Command failed with status: {Status(status).name}")
            return False

        if fragments:
            fragments.extend(packet['payload'])
            packet['payload'] = bytes(fragments)
            packet['length'] = len(fragments)

        return True if not packet['length'] else bytes(packet['payload'])

    if not poll_events:
        raise TimeoutException("Packet receive timeout")


def _send_cmd_wait_resp(self, opcode, channel=0, data=b''):
    if not self.is_connected():
        raise OMVException("Not connected")

    if opcode in [Opcode.SYS_RESET, Opcode.SYS_BOOT]:
        self.transport.send_packet(opcode, channel, 0, data)
        self.disconnect()
        return None

    try:
        self.transport.send_packet(opcode, channel, 0, data)
        return self.transport.recv_packet()
    except OMVException:
        self._resync()
        raise ResyncException()
    # PATCH: no sys.exit(0) catch-all; let other exceptions propagate


Transport.recv_packet = _recv_packet
Camera._send_cmd_wait_resp = _send_cmd_wait_resp

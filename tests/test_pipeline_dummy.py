from dustycam.frame import FramePacket
from dustycam.node import SourceNode, Node, SinkNode
from dustycam.runner import Runner

import numpy as np


class DummySource(SourceNode):
    def __init__(self, n=3):
        super().__init__(name="DummySource")
        self._counter = 0
        self._n = n

    def next_packet(self):
        if self._counter >= self._n:
            return None
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        pkt = FramePacket(frame_id=self._counter, timestamp=float(self._counter), image=img)
        self._counter += 1
        return pkt

    def forward(self, packet: FramePacket) -> FramePacket:
        return packet


class A(Node):
    def forward(self, packet: FramePacket) -> FramePacket:
        pkt = packet.copy_shallow()
        pkt.ocr_results.append("x")
        return pkt


class B(Node):
    def forward(self, packet: FramePacket) -> FramePacket:
        pkt = packet.copy_shallow()
        pkt.ocr_results.append("y")
        return pkt


class CollectSink(SinkNode):
    def __init__(self):
        super().__init__(name="CollectSink")
        self.received = []

    def forward(self, packet: FramePacket) -> FramePacket:
        return packet

    def consume(self, packet: FramePacket) -> None:
        self.received.append(packet.ocr_results)


def test_dummy_chain_runs():
    src = DummySource(n=2)
    a = A()
    b = B()
    sink = CollectSink()

    src.connect(a).connect(b).connect(sink)
    runner = Runner([src], [sink])
    while runner.run_once():
        pass

    assert sink.received == [["x", "y"], ["x", "y"]]

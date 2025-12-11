from time import time
import numpy as np

from dustycam.frame_packet import FramePacket
from dustycam.node import SourceNode, Node, SinkNode
from dustycam.dag_runner import Runner


class DummySource(SourceNode):
    def __init__(self):
        super().__init__(name="DummySource")
        self._counter = 0

    def next_packet(self):
        if self._counter >= 5:
            return None
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        pkt = FramePacket(frame_id=self._counter, timestamp=time(), image=img)
        self._counter += 1
        return pkt

    def forward(self, packet: FramePacket) -> FramePacket:
        return packet


class NodeA(Node):
    def forward(self, packet: FramePacket) -> FramePacket:
        pkt = packet.copy_shallow()
        # Attach dummy metadata
        pkt.ocr_results.append("hello")
        return pkt


class NodeB(Node):
    def forward(self, packet: FramePacket) -> FramePacket:
        pkt = packet.copy_shallow()
        if pkt.ocr_results:
            pkt.ocr_results[0] = pkt.ocr_results[0] + " world"
        return pkt


class PrintSink(SinkNode):
    def forward(self, packet: FramePacket) -> FramePacket:
        return packet

    def consume(self, packet: FramePacket) -> None:
        print(f"[frame {packet.frame_id}] ocr_results={packet.ocr_results}")


def main():
    src = DummySource()
    a = NodeA()
    b = NodeB()
    sink = PrintSink()

    src.connect(a).connect(b).connect(sink)

    runner = Runner(sources=[src], sinks=[sink])
    while runner.run_once():
        pass


if __name__ == "__main__":
    main()

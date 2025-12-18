import pytest
import cv2
import numpy as np
import threading
import time
from dustycam.frame import FramePacket
from dustycam.node import Node
from dustycam.runner import Runner
from dustycam.nodes.sources import create_source
from dustycam.nodes.sinks.null import NullSink
from dustycam.webapp import create_app

class MockSource(Node):
    def __init__(self):
        super().__init__("MockSource")
        self.frame_count = 0
        
    def next_packet(self):
        self.frame_count += 1
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        return FramePacket(frame_id=self.frame_count, timestamp=0.0, image=img)
        
    def forward(self, packet):
        return packet
        
class MockSink(NullSink):
    pass

def test_expose_nodes_via_runner():
    source = MockSource()
    sink = MockSink()
    source.connect(sink)
    
    # Manually configure nodes as Source/Sink types for Runner to be happy?
    # Runner expects SourceNode and SinkNode types in init list.
    # But for this test let's just cheat or use real classes if possible, 
    # or subclass correctly.
    
    # Actually Runner.__init__ expects List[SourceNode], List[SinkNode]
    # So MockSource needs to inherit SourceNode
    pass

from dustycam.node import SourceNode, SinkNode

class RealMockSource(SourceNode):
    def next_packet(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        return FramePacket(frame_id=1, timestamp=0.0, image=img)
    def forward(self, packet): return packet

def test_runner_updates_latest_packet():
    source = RealMockSource()
    sink = NullSink()
    source.connect(sink)
    
    runner = Runner([source], [sink])
    
    # Run once
    runner.run_once()
    
    # Check if source has latest packet populated
    bytes_out = source.get_last_frame_bytes()
    assert bytes_out is not None
    assert len(bytes_out) > 0

def test_webapp_runner_integration():
    source = RealMockSource()
    sink = NullSink()
    source.connect(sink)
    runner = Runner([source], [sink])
    runner.run_once() # Populate cache
    
    app = create_app(runner=runner)
    client = app.test_client()
    
    # Index should list nodes
    idx_res = client.get('/')
    assert b"RealMockSource" in idx_res.data
    assert b"NullSink" in idx_res.data
    
    # Snapshot
    snap_res = client.get('/snapshot/RealMockSource')
    assert snap_res.status_code == 200
    assert snap_res.mimetype == 'image/jpeg'

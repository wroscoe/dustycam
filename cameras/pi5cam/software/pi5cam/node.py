from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set
from abc import ABC, abstractmethod

from .frame import FramePacket


class Node(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._inputs: List[Node] = []
        self._outputs: List[Node] = []
        self._cache: Dict[int, FramePacket] = {}  # key: frame_id

    def connect(self, downstream: "Node") -> "Node":
        self._outputs.append(downstream)
        downstream._inputs.append(self)
        return downstream

    def clear_cache(self):
        self._cache.clear()

    def _get_from_cache(self, frame_id: int) -> Optional[FramePacket]:
        return self._cache.get(frame_id)

    def _set_cache(self, packet: FramePacket):
        self._cache[packet.frame_id] = packet

    def ready(self) -> bool:
        return all(inp is not None for inp in self._inputs)

    @abstractmethod
    def forward(self, packet: FramePacket) -> FramePacket:
        ...

    def get_properties(self) -> Dict:
        """Return a dictionary of editable properties."""
        return {}

    def set_properties(self, **kwargs):
        """Update properties from a dictionary."""
        pass


class SourceNode(Node):
    @abstractmethod
    def next_packet(self) -> Optional[FramePacket]:
        ...


class SinkNode(Node):
    @abstractmethod
    def consume(self, packet: FramePacket) -> None:
        ...

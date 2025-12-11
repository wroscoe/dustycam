from __future__ import annotations

from collections import deque
from typing import Iterable, List, Optional, Set, Dict

from .node import Node, SourceNode, SinkNode
from .frame import FramePacket


class Runner:
    def __init__(self, sources: List[SourceNode], sinks: List[SinkNode]):
        self.sources = sources
        self.sinks = sinks
        self.graph: List[Node] = self._collect_graph(sources)
        self.order: List[Node] = self._toposort()

    def _collect_graph(self, roots: List[Node]) -> List[Node]:
        seen: Set[Node] = set()
        stack: List[Node] = list(roots)
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(n._outputs)
        return list(seen)

    def _toposort(self) -> List[Node]:
        # Kahn's algorithm over directed acyclic graph
        indegree: Dict[Node, int] = {n: len(n._inputs) for n in self.graph}
        q = deque([n for n, d in indegree.items() if d == 0])
        order: List[Node] = []
        while q:
            n = q.popleft()
            order.append(n)
            for m in n._outputs:
                indegree[m] -= 1
                if indegree[m] == 0:
                    q.append(m)
        if len(order) != len(self.graph):
            raise ValueError("Graph contains cycles or disconnected nodes")
        return order

    def clear_caches(self):
        for n in self.graph:
            n.clear_cache()

    def run_once(self) -> bool:
        # Pull one packet from each source and push through sinks lazily
        progressed = False
        for src in self.sources:
            packet = src.next_packet()
            if packet is None:
                continue
            progressed = True
            if packet.drop_frame:
                continue
            # For each sink, request its output, triggering lazy execution
            for sink in self.sinks:
                out = self._eval_node_chain(sink, packet)
                if out is None:
                    continue
                if not out.drop_frame:
                    sink.consume(out)
        return progressed

    def _eval_node_chain(self, node: Node, packet: FramePacket) -> Optional[FramePacket]:
        # Memoize per node per frame_id
        cached = node._get_from_cache(packet.frame_id)
        if cached is not None:
            return cached
        # Evaluate dependencies first
        for inp in node._inputs:
            packet = self._eval_node_chain(inp, packet)  # type: ignore
            if packet is None:
                return None
            if packet.drop_frame:
                node._set_cache(packet)
                return packet
        # Now run this node
        result = node.forward(packet)
        node._set_cache(result)
        return result

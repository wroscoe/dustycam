import json
import os
import threading
import time
from typing import Callable, Tuple, List, Dict, Optional

from .node import Node, SourceNode, SinkNode
from .runner import Runner
from dustycam.nodes.sinks.web import GlobalWebServer

class PipelineManager:
    def __init__(self, factory_func: Callable[[Dict], Tuple[List[SourceNode], List[SinkNode]]], config_path: str = "pipeline_config.json"):
        """
        factory_func: Function(config_dict) -> (sources, sinks)
        config_path: File to load/save JSON config.
        """
        self.factory_func = factory_func
        self.config_path = config_path
        self.config = self._load_config()
        self.runner: Optional[Runner] = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def _load_config(self) -> Dict:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def save(self):
        with self.lock:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)

    def update_node_config(self, node_name: str, new_props: Dict):
        with self.lock:
            if "nodes" not in self.config:
                self.config["nodes"] = {}
            
            # Merge existing config
            current = self.config["nodes"].get(node_name, {})
            current.update(new_props)
            self.config["nodes"][node_name] = current
            
            print(f"Updated config for {node_name}: {current}")
            
    def start(self):
        if self.running:
            return
            
        self.running = True
        self._build_and_run()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            self.thread = None

    def reload(self):
        """Rebuilds pipeline with current config."""
        print("Reloading pipeline...")
        # Signal stop
        was_running = self.running
        self.running = False
        if self.thread:
            self.thread.join()
        
        # Restart
        if was_running:
            self.running = True
            self._build_and_run()
            
    def _build_and_run(self):
        # Build graph using factory
        try:
            sources, sinks = self.factory_func(self.config)
            self.runner = Runner(sources, sinks)
            
            # Update web dashboard graph
            # Note: WebSinks in the new graph need to register themselves, 
            # which they do in __init__. 
            # But we also need to tell GlobalWebServer the new full node list.
            GlobalWebServer.get_instance().set_graph(self.runner.order)
            
            # Apply saved config to nodes immediately (or factory should have done it)
            # The factory is responsible for using 'config' to init nodes.
            # But we can also double check or apply overrides.
            for node in self.runner.order:
                if "nodes" in self.config and node.name in self.config["nodes"]:
                     node.set_properties(**self.config["nodes"][node.name])

            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"Failed to build pipeline: {e}")
            self.running = False

    def _run_loop(self):
        print("Pipeline started.")
        while self.running and self.runner:
            if not self.runner.run_once():
                time.sleep(0.001)
        print("Pipeline stopped.")

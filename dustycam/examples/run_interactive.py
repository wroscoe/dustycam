import time
import logging
from typing import Dict, Tuple, List

from dustycam.nodes.sources import BouncingBallSource
from dustycam.nodes.detectors.yolo import YoloNode
from dustycam.nodes.processors.drawing import DrawDetectionsNode
from dustycam.nodes.sinks.web import WebSink, GlobalWebServer
from dustycam.pipeline import PipelineManager
from dustycam.node import SourceNode, SinkNode

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)

def build_pipeline(config: Dict) -> Tuple[List[SourceNode], List[SinkNode]]:
    """
    Factory function to build the graph based on configuration.
    """
    print(f"Building pipeline with config: {config}")
    
    # Defaults
    nodes_cfg = config.get("nodes", {})
    
    # 1. Source
    ball_cfg = nodes_cfg.get("BallSource", {})
    source = BouncingBallSource(
        radius=ball_cfg.get("radius", 30),
        color=tuple(ball_cfg.get("color", (0, 255, 255))),
        velocity=(ball_cfg.get("vx", 5), ball_cfg.get("vy", 5)),
        name="BallSource"
    )
    
    # 2. WebSink for proper "Raw" output
    sink_raw = WebSink(path="raw_input", name="WebSink_raw")
    source.connect(sink_raw)
    
    # 3. YoloNode
    yolo_cfg = nodes_cfg.get("YoloNode", {})
    yolo = YoloNode(model_name=yolo_cfg.get("model_name", "yolov8n"), name="YoloNode")
    source.connect(yolo)
    
    # 4. Draw Node
    draw = DrawDetectionsNode(name="DrawDetectionsNode")
    yolo.connect(draw)
    
    # 5. WebSink for processed output
    sink_processed = WebSink(path="yolo_output", name="WebSink_processed")
    draw.connect(sink_processed)
    
    return [source], [sink_raw, sink_processed]

def main():
    print("Starting Interactive Pipeline...")
    print("Go to http://localhost:8000 to configure.")
    
    # Initialize GlobalWebServer (singleton)
    web_server = GlobalWebServer.get_instance(port=8000)
    web_server.start()
    
    # Initialize PipelineManager
    manager = PipelineManager(factory_func=build_pipeline, config_path="pipeline_config.json")
    
    # Connect Manager to Web Server
    web_server.set_manager(manager)
    
    # Start the pipeline
    manager.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        manager.stop()

if __name__ == "__main__":
    main()

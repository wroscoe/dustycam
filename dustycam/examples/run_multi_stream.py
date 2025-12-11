
from dustycam.nodes.sources import BouncingBallSource
from dustycam.nodes.detectors.yolo import YoloNode
from dustycam.nodes.processors.drawing import DrawDetectionsNode
from dustycam.nodes.sinks.web import WebSink
from dustycam.runner import Runner

def main():
    print("Starting Multi-Stream Dashboard...")
    print("Go to http://localhost:8000 to view all streams.")
    
    # 1. Source: Bouncing Ball
    # We'll use one source and fan it out to two paths.
    source = BouncingBallSource(
        radius=30,
        color=(0, 255, 255),  # Yellow ball (BGR)
        size=(640, 480),
        velocity=(12, 12),
        name="BallSource"
    )
    
    # Path 1: Raw Stream
    sink_raw = WebSink(path="raw_input", port=8000)
    
    # Path 2: Processed Stream (YOLO)
    yolo = YoloNode(model_name="yolov8n.pt")
    drawer = DrawDetectionsNode()
    sink_processed = WebSink(path="yolo_output", port=8000)
    
    # Connect
    # Source feeds both paths
    source.connect(sink_raw) # Branch 1
    
    source.connect(yolo).connect(drawer).connect(sink_processed) # Branch 2
    
    # Run
    runner = Runner(sources=[source], sinks=[sink_raw, sink_processed])
    
    # Push the graph to the dashboard
    from dustycam.nodes.sinks.web import GlobalWebServer
    GlobalWebServer.get_instance().set_graph(runner.order)
    
    try:
        while True:
            if not runner.run_once():
                break
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()

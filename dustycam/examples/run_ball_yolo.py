
from dustycam.nodes.sources import BouncingBallSource
from dustycam.nodes.detectors.yolo import YoloNode
from dustycam.nodes.processors.drawing import DrawDetectionsNode
from dustycam.nodes.sinks.web import WebSink
from dustycam.runner import Runner

def main():
    print("Starting Bouncing Ball YOLO Demo...")
    print("Go to http://localhost:8000 to view the stream.")
    
    # 1. Source: Red Bouncing Ball
    # Radius=40 to be fairly large, Color=(0,0,255) is RED in BGR
    source = BouncingBallSource(
        radius=40,
        color=(0, 0, 255), 
        size=(640, 480),
        velocity=(10, 8)
    )
    
    # 2. Ops
    # Use yolov8n.pt. A red ball often gets detected as "sports ball" (class 32 in COCO)
    # or sometimes "apple" or "orange".
    yolo = YoloNode(model_name="yolov8n.pt")
    drawer = DrawDetectionsNode()
    
    # 3. Sink
    sink = WebSink(port=8000)
    
    # 4. Connect
    source.connect(yolo).connect(drawer).connect(sink)
    
    # 5. Run
    runner = Runner(sources=[source], sinks=[sink])
    
    try:
        while True:
            if not runner.run_once():
                break
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()

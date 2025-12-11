
import cv2
import io
import logging
import threading
from http import server
from threading import Condition
from typing import Optional, Dict, List

from dustycam.frame import FramePacket
from dustycam.node import SinkNode, Node

class StreamingOutput:
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, frame_bytes):
        with self.condition:
            self.frame = frame_bytes
            self.condition.notify_all()

class GlobalWebServer:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, port: int = 8000):
        self.port = port
        self.outputs: Dict[str, StreamingOutput] = {}
        self.nodes: List['Node'] = []
        self.manager = None
        self.server = None
        self.thread = None

    @classmethod
    def get_instance(cls, port: int = 8000):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(port)
            return cls._instance

    def register(self, path: str, output: StreamingOutput):
        # path should be something like "cam1"
        self.outputs[path] = output

    def set_graph(self, nodes: List['Node']):
        self.nodes = nodes
        
    def set_manager(self, manager):
        self.manager = manager

    def _find_downstream_path(self, start_node: 'Node') -> Optional[str]:
        # Implementation removed in previous step, but let's keep it clean
        return None

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        
        # Capture port to ensure we get the right instance (though it's singleton)
        port_captured = self.port
        
        class MultiStreamHandler(server.BaseHTTPRequestHandler):
            def do_POST(self):
                gws = GlobalWebServer.get_instance(port_captured)
                manager = gws.manager
                
                try:
                    content_length = int(self.headers.get('Content-Length', 0))
                    post_data = self.rfile.read(content_length)
                    import json
                    
                    if self.path == '/api/config':
                        data = json.loads(post_data)
                        if manager:
                            node_name = data.get('node_name')
                            props = data.get('props')
                            manager.update_node_config(node_name, props)
                            self.send_response(200)
                            self.end_headers()
                            self.wfile.write(b"OK")
                        else:
                            self.send_error(500, "No Manager")
                            
                    elif self.path == '/api/save':
                        if manager:
                            manager.save()
                            self.send_response(200)
                            self.end_headers()
                            self.wfile.write(b"Saved")
                    
                    elif self.path == '/api/restart':
                        if manager:
                            manager.reload()
                            self.send_response(200)
                            self.end_headers()
                            self.wfile.write(b"Restarted")
                    else:
                        self.send_error(404)
                        
                except Exception as e:
                    self.send_error(500, str(e))

            def do_GET(self):
                # Access the singleton dynamically to see updates
                gws = GlobalWebServer.get_instance(port_captured)
                outputs_ref = gws.outputs
                nodes_ref = gws.nodes

                if self.path == '/' or self.path == '/index.html':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b'<html><head><title>DustyCam</title>')
                    self.wfile.write(b'<style>body{font-family:sans-serif; background:#222; color:#fff;}')
                    self.wfile.write(b'.top-bar{padding:10px; background:#111; border-bottom:1px solid #444; margin-bottom:20px; display:flex; gap:10px;}')
                    self.wfile.write(b'button{padding:8px 16px; background:#4CAF50; color:white; border:none; border-radius:4px; cursor:pointer;}')
                    self.wfile.write(b'button:hover{background:#45a049;}')
                    self.wfile.write(b'.container{display:flex; flex-wrap:wrap; gap:20px;}')
                    self.wfile.write(b'.node-card{background:#333; padding:15px; border-radius:8px; border:1px solid #444; min-width:300px;}')
                    self.wfile.write(b'h3{margin-top:0; border-bottom:1px solid #555; padding-bottom:10px; display:flex; justify-content:space-between; align-items:center;}')
                    self.wfile.write(b'img{max-width:100%; height:auto; display:block; margin-top:10px;}')
                    self.wfile.write(b'.props{font-size:0.9em; color:#aaa; margin-top:10px;}')
                    self.wfile.write(b'.prop-row{margin-bottom:8px;} label{display:inline-block; width:80px;}')
                    self.wfile.write(b'input{background:#444; border:1px solid #555; color:white; padding:4px;}')
                    self.wfile.write(b'</style>')
                    
                    self.wfile.write(b'<script>')
                    self.wfile.write(b'function updateConfig(nodeName, key, value) {')
                    self.wfile.write(b'  fetch("/api/config", {method:"POST", body:JSON.stringify({node_name:nodeName, props:{[key]:value}})});')
                    self.wfile.write(b'}')
                    self.wfile.write(b'function save() { fetch("/api/save", {method:"POST"}).then(()=>alert("Saved!")); }')
                    self.wfile.write(b'function restart() { fetch("/api/restart", {method:"POST"}).then(()=>setTimeout(()=>location.reload(), 1000)); }')
                    self.wfile.write(b'</script>')

                    self.wfile.write(b'</head><body>')
                    
                    self.wfile.write(b'<div class="top-bar">')
                    self.wfile.write(b'<h1>DustyCam</h1>')
                    self.wfile.write(b'<button onclick="save()">Save Config</button>')
                    self.wfile.write(b'<button onclick="restart()">Restart Pipeline</button>')
                    self.wfile.write(b'</div>')
                    
                    self.wfile.write(b'<div class="container">')
                    
                    # Logic: If we have nodes set, iterate them. Otherwise fallback to just outputs (legacy/partial)
                    nodes_to_render = nodes_ref if nodes_ref else []
                    
                    for node in nodes_to_render:
                        name = node.name
                        self.wfile.write(f'<div class="node-card"><h3>{name}</h3>'.encode('utf-8'))
                        
                        # Render Properties Form
                        props = node.get_properties()
                        if props:
                            self.wfile.write(b'<div class="props">')
                            for k, v in props.items():
                                val_str = str(v)
                                # Simple heuristic for inputs
                                self.wfile.write(f'<div class="prop-row"><label>{k}:</label>'.encode('utf-8'))
                                # Handle lists (like color) as comma string for simplicity
                                if isinstance(v, list) or isinstance(v, tuple):
                                     val_str = ",".join(map(str, v))
                                     
                                self.wfile.write(f'<input value="{val_str}" onchange="updateConfig(\'{name}\', \'{k}\', this.value.split(\',\').length > 1 ? this.value.split(\',\').map(Number) : (isNaN(this.value) ? this.value : Number(this.value)))">'.encode('utf-8'))
                                self.wfile.write(b'</div>')
                            self.wfile.write(b'</div>')
                        
                        path = None
                        is_sink = False
                        
                        # Check if it's a WebSink
                        if hasattr(node, 'path') and getattr(node, 'path') in outputs_ref:
                             path = getattr(node, 'path')
                             is_sink = True
                        
                        if is_sink and path:
                             self.wfile.write(f'<img src="/{path}.mjpg" width="640" height="480" />'.encode('utf-8'))
                        elif not props:
                             # Just show text for non-sinks if no properties are rendered
                             self.wfile.write(b'<div class="props">Processing Node</div>')

                        self.wfile.write(b'</div>')

                    # If no nodes (e.g. set_graph not called), show connected sinks as fallback
                    if not nodes_to_render:
                        for path in outputs_ref.keys():
                            self.wfile.write(f'<div class="node-card"><h3>{path}</h3>'.encode('utf-8'))
                            self.wfile.write(f'<img src="/{path}.mjpg" width="640" height="480" /></div>'.encode('utf-8'))
                    
                    self.wfile.write(b'</div></body></html>')
                    return

                # Check if it's a stream request
                clean_path = self.path.lstrip('/')
                if clean_path.endswith('.mjpg'):
                    stream_name = clean_path[:-5] # remove .mjpg
                    if stream_name in outputs_ref:
                        output = outputs_ref[stream_name]
                        self.send_response(200)
                        self.send_header('Age', 0)
                        self.send_header('Cache-Control', 'no-cache, private')
                        self.send_header('Pragma', 'no-cache')
                        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
                        self.end_headers()
                        try:
                            while True:
                                with output.condition:
                                    output.condition.wait()
                                    frame = output.frame
                                self.wfile.write(b'--FRAME\r\n')
                                self.send_header('Content-Type', 'image/jpeg')
                                self.send_header('Content-Length', len(frame))
                                self.end_headers()
                                self.wfile.write(frame)
                                self.wfile.write(b'\r\n')
                        except Exception:
                            pass
                        return

                self.send_error(404)
                self.end_headers()

        # Use ThreadingHTTPServer to handle multiple concurrent streams
        self.server = server.ThreadingHTTPServer(('0.0.0.0', self.port), MultiStreamHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"GlobalWebServer serving at http://0.0.0.0:{self.port}")


class WebSink(SinkNode):
    def __init__(self, path: str = "stream", port: int = 8000, name: Optional[str] = None):
        """
        path: The identifier for this stream (e.g., 'raw', 'processed').
              Will be accessible at /{path}.mjpg
        """
        super().__init__(name=name or f"WebSink_{path}")
        self.path = path  # Save path for Logic/Debug
        self.output = StreamingOutput()
        
        # Register with global server
        web_server = GlobalWebServer.get_instance(port)
        web_server.register(path, self.output)
        web_server.start()

    def forward(self, packet: FramePacket) -> FramePacket:
        return packet

    def consume(self, packet: FramePacket) -> None:
        if packet.image is not None:
            # Encode image to JPEG
            ret, jpeg = cv2.imencode('.jpg', packet.image)
            if ret:
                self.output.write(jpeg.tobytes())
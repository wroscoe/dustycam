from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import asyncio
import base64
import json
import time
from typing import List, Dict, Any
import os

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # We must serialize to text for consistency with previous helper
        # Or just send JSON
        text = json.dumps(message)
        for connection in self.active_connections:
            try:
                await connection.send_text(text)
            except Exception:
                # Handle broken pipe
                pass

manager = ConnectionManager()

async def broadcast_loop(app):
    """Background task to push frames to connected clients."""
    print("Starting FastAPI broadcast loop...")
    import cv2
    
    while True:
        try:
            # Rate limit: 2 FPS
            await asyncio.sleep(0.5)
            
            pipeline = getattr(app.state, "pipeline", None)
            if not pipeline or not manager.active_connections:
                continue

            # Get previews
            previews = list(pipeline.previews.keys())
            payload = {}
            
            # Encode in thread to avoid blocking event loop
            def encode_frames():
                data = {}
                for name in previews:
                    frame = pipeline.get_preview(name)
                    if frame is not None:
                        ret, jpeg = cv2.imencode('.jpg', frame)
                        if ret:
                            b64 = base64.b64encode(jpeg).decode('utf-8')
                            data[name] = f"data:image/jpeg;base64,{b64}"
                return data

            payload = await asyncio.to_thread(encode_frames)
            
            if payload:
                meta = pipeline.state
                await manager.broadcast({'images': payload, 'meta': meta})
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Broadcast error: {e}")
            await asyncio.sleep(1)

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    previews = []
    if pipeline:
        previews = list(pipeline.previews.keys())
    return templates.TemplateResponse("index.html", {"request": request, "nodes": previews})

@router.get("/gallery", response_class=HTMLResponse)
async def gallery(request: Request):
    return templates.TemplateResponse("gallery.html", {"request": request})

@router.get("/api/recordings")
async def get_recordings():
    data_path = Path("/opt/dustycam/data")
    if not data_path.exists():
        return []
    
    runs = []
    # List all run_ directories
    for item in data_path.iterdir():
        if item.is_dir() and item.name.startswith("run_"):
            images = []
            # Find images in the run dir
            for img in item.glob("*.jpg"):
                # URL path: /files/{run_name}/{img_name}
                images.append(f"/files/{item.name}/{img.name}")
            
            # Sort images by name (timestamp usually)
            images.sort()
            
            runs.append({
                "name": item.name,
                "timestamp": item.stat().st_mtime,
                "images": images
            })
    
    # Sort runs by timestamp descending (newest first)
    runs.sort(key=lambda x: x['timestamp'], reverse=True)
    return runs

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, maybe receive commands later
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

@router.get("/api/settings")
async def get_settings(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if not pipeline:
        return JSONResponse({"error": "No pipeline"}, status_code=500)
    
    if pipeline.settings_model:
        return {
            "values": pipeline.settings.model_dump(),
            "schema": pipeline.settings_model.model_json_schema()
        }
    else:
        return {
            "values": pipeline.settings,
            "schema": None
        }

@router.post("/api/settings")
async def update_settings(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if not pipeline:
        return JSONResponse({"error": "No pipeline"}, status_code=500)
    
    new_settings = await request.json()
    
    # Check for restart requirement
    restart_needed = False
    if pipeline.settings_model:
        schema = pipeline.settings_model.model_json_schema()
        properties = schema.get('properties', {})
        for key in new_settings:
            field_props = properties.get(key, {})
            if field_props.get('restart_required'):
                restart_needed = True
                break
    
    pipeline.update_settings(new_settings, restart=restart_needed)
    return {"status": "ok", "restarted": restart_needed}

@router.get('/snapshot/{preview_name}')
async def snapshot(preview_name: str, request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if not pipeline:
         return JSONResponse("No Pipeline Configured", status_code=500)

    frame = pipeline.get_preview(preview_name)
    
    if frame is not None:
        import cv2
        # Run encode in thread
        def encode():
             ret, jpeg = cv2.imencode('.jpg', frame)
             return jpeg.tobytes() if ret else None
             
        data = await asyncio.to_thread(encode)
        if data:
             from fastapi.responses import Response
             return Response(content=data, media_type="image/jpeg")
             
    return JSONResponse("No frame", status_code=404)


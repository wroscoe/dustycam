from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import asyncio

from .routes import router, broadcast_loop

def create_app(pipeline=None):
    app = FastAPI(title="DustyCam")
    
    if pipeline:
        app.state.pipeline = pipeline
        
    static_path = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    # Mount data directory for recordings
    # Assuming data is at /opt/dustycam/data, relative to this file it is ../../../data if installed there, 
    # or we can use the absolute path /opt/dustycam/data as we know environment.
    # Let's use absolute path /opt/dustycam/data for safety as per user rules context.
    data_path = Path("/opt/dustycam/data")
    if data_path.exists():
        app.mount("/files", StaticFiles(directory=data_path), name="files")
    
    app.include_router(router)
    
    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(broadcast_loop(app))
    
    return app

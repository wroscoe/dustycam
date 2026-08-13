# server

Reserved for the base-station / ingest side of DustyCam — the thing cameras
report *to*, as opposed to the software that runs *on* a camera.

Nothing lives here yet. Intended scope:

- Receive detections (and optionally images) from one or more cameras
- Store and serve the resulting time series
- Provide a UI for reviewing detections across cameras

Note that the live MJPEG preview server (`GlobalWebServer` / `WebSink` in
`cameras/pi5cam/software/pi5cam/nodes/sinks/web.py`) is deliberately *not*
here — it runs on the camera as a pipeline sink, and is part of the camera's
runtime rather than a separate service.

Existing per-camera ingest endpoints that may inform this:
`cameras/esp32_s3_cam/software/server/pump_server.py` and
`cameras/esp32_s3_cam/software/tools/server.py`.

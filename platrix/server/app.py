"""FastAPI application factory and routes."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from platrix import __version__
from platrix.config import Settings, get_settings
from platrix.core.pipeline import annotate
from platrix.core.types import Frame
from platrix.logging_conf import configure_logging, get_logger
from platrix.server.streaming import StreamManager
from platrix.storage import EventStore

logger = get_logger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class StartRequest(BaseModel):
    source: str
    loop: bool = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    store = EventStore(settings)
    manager = StreamManager(settings, store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.default_source:
            try:
                manager.start(settings.default_source)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to start default source")
        yield
        manager.stop()

    app = FastAPI(
        title="Platrix",
        version=__version__,
        description="Real-time Iranian license plate recognition engine.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Meta ------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/api/status")
    def status() -> dict:
        return {**manager.status, "stats": store.stats()}

    # --- Live source control --------------------------------------------
    @app.post("/api/stream/start")
    def stream_start(req: StartRequest) -> dict:
        try:
            manager.start(req.source, loop=req.loop)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return manager.status

    @app.post("/api/stream/stop")
    def stream_stop() -> dict:
        manager.stop()
        return manager.status

    @app.get("/api/stream/mjpeg")
    def stream_mjpeg() -> StreamingResponse:
        return StreamingResponse(
            manager.mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # --- One-shot image recognition -------------------------------------
    @app.post("/api/recognize")
    async def recognize(file: UploadFile = File(...)) -> dict:
        raw = await file.read()
        array = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        frame = Frame(image=image, source=f"upload:{file.filename}")
        readings = manager.pipeline.process(frame)
        events = [store.record(r).to_dict() for r in readings]

        annotated = annotate(image, readings)
        ok, buf = cv2.imencode(".jpg", annotated)
        preview = buf.tobytes() if ok else b""
        import base64

        return {
            "count": len(events),
            "plates": events,
            "annotated_image": "data:image/jpeg;base64,"
            + base64.b64encode(preview).decode("ascii"),
        }

    # --- Event log -------------------------------------------------------
    @app.get("/api/events")
    def events(
        limit: int = Query(100, ge=1, le=1000),
        plate: str | None = Query(None),
    ) -> dict:
        return {"events": store.recent(limit=limit, plate=plate)}

    @app.get("/api/stats")
    def stats() -> dict:
        return store.stats()

    # --- Live event WebSocket -------------------------------------------
    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket) -> None:
        await ws.accept()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_event(event: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        manager.subscribe(on_event)
        # Replay recent events so a fresh client isn't empty.
        for ev in manager.recent_events[-10:]:
            await ws.send_json(ev)
        try:
            while True:
                event = await queue.get()
                await ws.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            manager.unsubscribe(on_event)

    # --- Static + dashboard ---------------------------------------------
    app.mount(
        "/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static"
    )
    # Serve saved snapshots for the log table.
    app.mount(
        "/snapshots",
        StaticFiles(directory=str(settings.snapshots_dir)),
        name="snapshots",
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB_DIR / "templates" / "index.html").read_text(encoding="utf-8")

    app.state.manager = manager
    app.state.store = store
    return app

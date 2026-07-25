"""FastAPI application factory and routes."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from platrix.server import auth

import json
import os
import subprocess
import sys

from platrix import __version__
from platrix.config import BASE_DIR, Settings, get_settings
from platrix.core.pipeline import annotate
from platrix.core.types import Frame
from platrix.logging_conf import configure_logging, get_logger
from platrix.server.multistream import ADHOC_ID, MultiStreamManager
from platrix.storage import EventStore

logger = get_logger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class StartRequest(BaseModel):
    source: str
    loop: bool = False
    direction: str = "unknown"  # "entry" | "exit" | "unknown"


class WatchRequest(BaseModel):
    plate: str
    name: str = ""
    list_type: str = "white"  # "white" | "black"
    note: str = ""


class CameraRequest(BaseModel):
    name: str = ""
    url: str
    direction: str = "unknown"


class CameraToggle(BaseModel):
    enabled: bool


class TestRequest(BaseModel):
    url: str


class AccessRequest(BaseModel):
    email: str


class LoginRequest(BaseModel):
    username: str
    password: str


class PasswordChange(BaseModel):
    current_password: str
    new_username: str
    new_password: str


class TokenRequest(BaseModel):
    name: str = "token"


class TrainRequest(BaseModel):
    epochs: int = 15
    device: str = "auto"        # "auto" | "cpu" | "gpu"
    install_cuda: bool = False  # opt-in: install a CUDA build if using the GPU


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    store = EventStore(settings)
    manager = MultiStreamManager(settings, store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Auto-start every camera marked "always-on" (surveillance mode).
        manager.start_enabled(store.list_cameras())
        if settings.default_source:
            try:
                manager.start_adhoc(settings.default_source)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to start default source")
        yield
        manager.stop_all()

    app = FastAPI(
        title="Platrix API",
        version=__version__,
        description=(
            "Real-time Iranian license plate recognition (ALPR) engine.\n\n"
            "**Authentication is token-based.** Create an API token in the "
            "dashboard (**Settings → API access tokens**), click **Authorize** "
            "above, and paste it. All requests then send "
            "`Authorization: Bearer <token>`.\n\n"
            "```bash\n"
            "curl -H 'Authorization: Bearer pltx_xxx' \\\n"
            "     -F 'file=@car.jpg' http://<host>/api/recognize\n"
            "```"
        ),
        lifespan=lifespan,
    )

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        from fastapi.openapi.utils import get_openapi

        schema = get_openapi(
            title=app.title, version=app.version,
            description=app.description, routes=app.routes,
        )
        schema.setdefault("components", {})["securitySchemes"] = {
            "BearerAuth": {
                "type": "http", "scheme": "bearer",
                "description": "An API token from Settings → API access tokens.",
            }
        }
        schema["security"] = [{"BearerAuth": []}]
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _valid_login(username: str, password: str) -> bool:
        db_user, db_hash = store.get_credentials()
        if db_user and db_hash:  # credentials changed via the dashboard
            return username == db_user and auth.verify_password(password, db_hash)
        return auth.check_credentials(settings, username, password)

    def _bearer(request: Request) -> str | None:
        header = request.headers.get("authorization", "")
        return header[7:].strip() if header.startswith("Bearer ") else None

    def _current_user(request: Request) -> str | None:
        """Username from a Bearer session token or the login cookie."""
        tok = _bearer(request)
        if tok:
            user = auth.verify_token(tok, settings.secret_key)
            if user:
                return user
        return auth.verify_token(request.cookies.get(auth.COOKIE_NAME), settings.secret_key)

    def _authorized(request: Request) -> bool:
        # Primary: a Bearer token in the header (dashboard session token OR an
        # API access token). This is what makes the API token-managed.
        tok = _bearer(request)
        if tok and (
            auth.verify_token(tok, settings.secret_key) is not None
            or store.verify_api_token(tok)
        ):
            return True
        # Browser <img> requests (MJPEG stream / snapshots) can't send headers,
        # so those two paths also accept the cookie or a ?token= query param.
        path = request.url.path
        if (path.startswith("/api/stream/mjpeg") or path.startswith("/snapshots")
                or path.startswith("/learn-media")):
            if auth.verify_token(request.cookies.get(auth.COOKIE_NAME), settings.secret_key):
                return True
            qt = request.query_params.get("token")
            if qt and (auth.verify_token(qt, settings.secret_key) or store.verify_api_token(qt)):
                return True
        return False

    @app.middleware("http")
    async def require_auth(request: Request, call_next):
        if settings.auth_enabled and not auth.is_public_path(request.url.path):
            if not _authorized(request):
                return JSONResponse({"detail": "Authentication required"}, status_code=401)
        return await call_next(request)

    # --- Authentication --------------------------------------------------
    @app.post("/api/login")
    def login(req: LoginRequest) -> JSONResponse:
        if not _valid_login(req.username, req.password):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        token = auth.make_token(req.username, settings.secret_key)
        # Return the token (the dashboard sends it as a Bearer header) and also
        # set a cookie (only used by the browser's <img> stream / snapshots).
        resp = JSONResponse({"ok": True, "username": req.username, "token": token})
        resp.set_cookie(
            auth.COOKIE_NAME, token, httponly=True, samesite="lax", max_age=7 * 24 * 3600
        )
        return resp

    @app.post("/api/logout")
    def logout() -> JSONResponse:
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(auth.COOKIE_NAME)
        return resp

    @app.get("/api/me")
    def me(request: Request) -> dict:
        if not settings.auth_enabled:
            return {"authenticated": True, "auth_enabled": False, "username": settings.auth_user}
        user = _current_user(request)
        return {"authenticated": user is not None, "auth_enabled": True, "username": user}

    @app.post("/api/account/password")
    def change_credentials(req: PasswordChange, request: Request) -> JSONResponse:
        current = _current_user(request) or store.get_credentials()[0] or settings.auth_user
        if not _valid_login(current, req.current_password):
            raise HTTPException(status_code=401, detail="Current password is incorrect")
        new_user = req.new_username.strip() or current
        if len(req.new_password) < 4:
            raise HTTPException(status_code=400, detail="Password must be at least 4 characters")
        store.set_credentials(new_user, req.new_password)
        token = auth.make_token(new_user, settings.secret_key)
        resp = JSONResponse({"ok": True, "username": new_user, "token": token})
        resp.set_cookie(
            auth.COOKIE_NAME, token, httponly=True, samesite="lax", max_age=7 * 24 * 3600
        )
        return resp

    # --- API tokens ------------------------------------------------------
    @app.get("/api/tokens")
    def tokens_list() -> dict:
        return {"tokens": store.list_api_tokens()}

    @app.post("/api/tokens")
    def tokens_create(req: TokenRequest) -> dict:
        raw, data = store.add_api_token(req.name)
        return {"token": raw, **data}  # raw token is returned only once

    @app.delete("/api/tokens/{token_id}")
    def tokens_delete(token_id: int) -> dict:
        if not store.delete_api_token(token_id):
            raise HTTPException(status_code=404, detail="Token not found")
        return {"deleted": token_id}

    # --- Meta ------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/api/status")
    def status() -> dict:
        return {**manager.overall_status(), "cameras": manager.statuses(), "stats": store.stats()}

    # --- Live source control (ad-hoc "view a URL") ----------------------
    @app.post("/api/stream/start")
    def stream_start(req: StartRequest) -> dict:
        try:
            manager.start_adhoc(req.source, direction=req.direction, loop=req.loop)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return manager.overall_status()

    @app.post("/api/stream/stop")
    def stream_stop() -> dict:
        manager.stop_camera(ADHOC_ID)
        return manager.overall_status()

    @app.get("/api/stream/mjpeg")
    def stream_mjpeg(camera: int = Query(ADHOC_ID)) -> StreamingResponse:
        return StreamingResponse(
            manager.mjpeg(camera),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # --- One-shot image recognition -------------------------------------
    @app.post("/api/recognize")
    async def recognize(
        file: UploadFile = File(...),
        direction: str = Query("unknown"),
    ) -> dict:
        raw = await file.read()
        array = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        frame = Frame(image=image, source=f"upload:{file.filename}")
        readings = manager.pipeline.process(frame)
        events = [store.record(r, direction=direction).to_dict() for r in readings]

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
        direction: str | None = Query(None),
        list_type: str | None = Query(None),
        date_from: str | None = Query(None),
        date_to: str | None = Query(None),
    ) -> dict:
        return {
            "events": store.recent(
                limit=limit,
                plate=plate,
                direction=direction,
                list_type=list_type,
                date_from=date_from,
                date_to=date_to,
            )
        }

    @app.get("/api/events/export")
    def events_export(
        plate: str | None = Query(None),
        direction: str | None = Query(None),
        list_type: str | None = Query(None),
        date_from: str | None = Query(None),
        date_to: str | None = Query(None),
        limit: int = Query(100000, ge=1, le=1000000),
    ) -> StreamingResponse:
        rows = store.recent(
            limit=limit, plate=plate, direction=direction,
            list_type=list_type, date_from=date_from, date_to=date_to,
        )

        def generate():
            import csv
            import io

            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow([
                "time", "plate", "plate_fa", "score", "direction",
                "matched_list", "matched_name", "source", "snapshot",
            ])
            yield buf.getvalue(); buf.seek(0); buf.truncate(0)
            for e in rows:
                writer.writerow([
                    e["created_at"], e["plate_text"], e["plate_text_fa"],
                    e["score"], e["direction"], e["matched_list"],
                    e["matched_name"], e["source"], e["snapshot_path"],
                ])
                yield buf.getvalue(); buf.seek(0); buf.truncate(0)

        return StreamingResponse(
            generate(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=platrix_detections.csv"},
        )

    @app.get("/api/stats")
    def stats() -> dict:
        return store.stats()

    # --- Watchlist (named plates, white / black list) --------------------
    @app.get("/api/watchlist")
    def watchlist_list(list_type: str | None = Query(None)) -> dict:
        return {"entries": store.list_watch(list_type=list_type)}

    @app.post("/api/watchlist")
    def watchlist_add(req: WatchRequest) -> dict:
        try:
            return store.add_watch(
                plate=req.plate,
                name=req.name,
                list_type=req.list_type,
                note=req.note,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/watchlist/{entry_id}")
    def watchlist_delete(entry_id: int) -> dict:
        if not store.delete_watch(entry_id):
            raise HTTPException(status_code=404, detail="Entry not found")
        return {"deleted": entry_id}

    # --- Cameras (saved video streams) ----------------------------------
    @app.get("/api/cameras")
    def cameras_list() -> dict:
        cams = store.list_cameras()
        live = manager.statuses()
        for c in cams:  # merge live connection status
            c["status"] = live.get(c["id"], {}).get("state", "off")
            c["live_fps"] = live.get(c["id"], {}).get("fps", 0)
        return {"cameras": cams}

    @app.post("/api/cameras")
    def cameras_add(req: CameraRequest) -> dict:
        try:
            return store.add_camera(name=req.name, url=req.url, direction=req.direction)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch("/api/cameras/{camera_id}")
    def cameras_toggle(camera_id: int, req: CameraToggle) -> dict:
        """Turn a camera's always-on monitoring on/off."""
        cam = store.set_camera_enabled(camera_id, req.enabled)
        if cam is None:
            raise HTTPException(status_code=404, detail="Camera not found")
        if req.enabled:
            manager.start_camera(cam)
        else:
            manager.stop_camera(camera_id)
        return cam

    @app.delete("/api/cameras/{camera_id}")
    def cameras_delete(camera_id: int) -> dict:
        manager.stop_camera(camera_id)
        if not store.delete_camera(camera_id):
            raise HTTPException(status_code=404, detail="Camera not found")
        return {"deleted": camera_id}

    @app.post("/api/cameras/test")
    def cameras_test(req: TestRequest) -> dict:
        from platrix.sources import test_source

        ok, message, frame = test_source(req.url)
        preview = None
        if ok and frame is not None:
            import base64

            h, w = frame.shape[:2]
            scale = 480 / max(w, 1)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            enc_ok, buf = cv2.imencode(".jpg", frame)
            if enc_ok:
                preview = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        return {"ok": ok, "message": message, "preview": preview}

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

    # --- Learn (train on user-labelled samples) -------------------------
    @app.get("/api/system/gpu")
    def system_gpu() -> dict:
        from platrix.server.system import gpu_info

        return gpu_info()

    @app.post("/api/learn/samples")
    async def learn_add(
        file: UploadFile = File(...),
        plate: str = Form(...),
        x: float = Form(...), y: float = Form(...),
        w: float = Form(...), h: float = Form(...),
    ) -> dict:
        raw = await file.read()
        return store.add_learn_sample(raw, plate, {"x": x, "y": y, "w": w, "h": h})

    @app.get("/api/learn/samples")
    def learn_list() -> dict:
        return {"samples": store.list_learn_samples()}

    @app.delete("/api/learn/samples/{sample_id}")
    def learn_delete(sample_id: int) -> dict:
        if not store.delete_learn_sample(sample_id):
            raise HTTPException(status_code=404, detail="Sample not found")
        return {"deleted": sample_id}

    def _job_file():
        return store.learn_dir() / "job.json"

    @app.get("/api/learn/status")
    def learn_status() -> dict:
        jf = _job_file()
        if jf.exists():
            try:
                return json.loads(jf.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
        return {"status": "idle"}

    @app.post("/api/learn/train")
    def learn_train(req: TrainRequest) -> dict:
        jf = _job_file()
        if jf.exists():
            try:
                if json.loads(jf.read_text()).get("status") == "running":
                    raise HTTPException(status_code=409, detail="Training already running")
            except HTTPException:
                raise
            except Exception:  # noqa: BLE001
                pass
        samples = store.list_learn_samples()
        if not samples:
            raise HTTPException(status_code=400, detail="Add at least one labelled sample first")

        images = store.learn_dir() / "images"
        dump = [
            {"image_path": str(images / s["image_file"]), "plate_text": s["plate_text"],
             "bbox": s["bbox"]}
            for s in samples
        ]
        (store.learn_dir() / "samples.json").write_text(
            json.dumps(dump, ensure_ascii=False), encoding="utf-8"
        )
        jf.write_text(json.dumps(
            {"status": "running", "step": "launching", "progress": 0, "log": []},
            ensure_ascii=False,
        ), encoding="utf-8")

        cmd = [
            sys.executable, "scripts/learn_train.py",
            "--job", str(jf), "--samples", str(store.learn_dir() / "samples.json"),
            "--out", str(settings.crnn_weights),
            "--epochs", str(max(1, min(req.epochs, 60))), "--device", req.device,
        ]
        synthetic = "/root/crnn_ds15k"
        if os.path.isdir(synthetic):
            cmd += ["--synthetic", synthetic, "--synthetic-limit", "1500"]
        if req.install_cuda:
            cmd += ["--install-cuda"]
        # Detached so it keeps running independently of the browser/session.
        subprocess.Popen(
            cmd, cwd=str(BASE_DIR), start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return {"ok": True, "started": True}

    @app.post("/api/learn/apply")
    def learn_apply() -> dict:
        manager.reload_models()
        return {"ok": True}

    # --- Access gate (email capture) ------------------------------------
    @app.post("/api/access")
    def access(req: AccessRequest, request: Request) -> dict:
        ua = request.headers.get("user-agent", "")
        try:
            return store.record_email(req.email, user_agent=ua)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/access")
    def access_list() -> dict:
        return {"emails": store.list_emails()}

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
    # Serve Learn sample images (thumbnails in the annotation UI).
    learn_images = settings.data_dir / "learn" / "images"
    learn_images.mkdir(parents=True, exist_ok=True)
    app.mount("/learn-media", StaticFiles(directory=str(learn_images)), name="learn-media")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB_DIR / "templates" / "index.html").read_text(encoding="utf-8")

    app.state.manager = manager
    app.state.store = store
    return app

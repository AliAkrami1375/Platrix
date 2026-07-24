import cv2
import numpy as np
from fastapi.testclient import TestClient

from platrix.config import Settings
from platrix.server.app import create_app


def _client(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        snapshots_dir=tmp_path / "data" / "snapshots",
        ocr="none",
        detector="contour",
    )
    return TestClient(create_app(settings))


def test_health(tmp_path):
    with _client(tmp_path) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


def test_recognize_endpoint(tmp_path):
    img = np.full((300, 400, 3), 40, dtype=np.uint8)
    img[130:160, 150:250] = 235
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    with _client(tmp_path) as client:
        res = client.post(
            "/api/recognize",
            files={"file": ("car.jpg", buf.tobytes(), "image/jpeg")},
        )
        assert res.status_code == 200
        body = res.json()
        assert "plates" in body
        assert body["annotated_image"].startswith("data:image/jpeg;base64,")


def test_events_empty(tmp_path):
    with _client(tmp_path) as client:
        res = client.get("/api/events")
        assert res.status_code == 200
        assert res.json()["events"] == []

from fastapi.testclient import TestClient

from platrix.config import Settings
from platrix.server.app import create_app
from platrix.server import auth


def _app(tmp_path, **kw):
    s = Settings(
        data_dir=tmp_path / "d", snapshots_dir=tmp_path / "d" / "s",
        ocr="none", detector="contour",
        auth_enabled=True, auth_user="admin", auth_password="secret",
        secret_key="k", **kw,
    )
    return TestClient(create_app(s))


def test_token_roundtrip():
    tok = auth.make_token("admin", "k")
    assert auth.verify_token(tok, "k") == "admin"
    assert auth.verify_token(tok, "other") is None
    assert auth.verify_token("garbage", "k") is None


def test_protected_without_login(tmp_path):
    with _app(tmp_path) as c:
        assert c.get("/api/health").status_code == 200  # public
        assert c.get("/api/events").status_code == 401   # protected


def test_login_then_access(tmp_path):
    with _app(tmp_path) as c:
        assert c.post("/api/login", json={"username": "admin", "password": "wrong"}).status_code == 401
        r = c.post("/api/login", json={"username": "admin", "password": "secret"})
        assert r.status_code == 200
        # cookie is now set on the client; protected route works
        assert c.get("/api/events").status_code == 200
        assert c.get("/api/me").json()["authenticated"] is True


def test_csv_export(tmp_path):
    with _app(tmp_path) as c:
        c.post("/api/login", json={"username": "admin", "password": "secret"})
        r = c.get("/api/events/export")
        assert r.status_code == 200
        assert "text/csv" in r.headers["content-type"]
        assert r.text.splitlines()[0].startswith("time,plate")

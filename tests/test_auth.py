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


def test_password_hash_roundtrip():
    h = auth.hash_password("hunter2")
    assert auth.verify_password("hunter2", h)
    assert not auth.verify_password("wrong", h)


def test_change_credentials_then_login(tmp_path):
    with _app(tmp_path) as c:
        c.post("/api/login", json={"username": "admin", "password": "secret"})
        r = c.post("/api/account/password", json={
            "current_password": "secret", "new_username": "ali", "new_password": "newpass"})
        assert r.status_code == 200
        c.post("/api/logout")
        # old creds rejected, new creds accepted
        assert c.post("/api/login", json={"username": "admin", "password": "secret"}).status_code == 401
        assert c.post("/api/login", json={"username": "ali", "password": "newpass"}).status_code == 200


def test_api_token_access(tmp_path):
    with _app(tmp_path) as c:
        c.post("/api/login", json={"username": "admin", "password": "secret"})
        tok = c.post("/api/tokens", json={"name": "ci"}).json()["token"]
        assert tok.startswith("pltx_")
        assert len(c.get("/api/tokens").json()["tokens"]) == 1
        c.post("/api/logout")
        # cookie gone → 401, but Bearer token works
        assert c.get("/api/events").status_code == 401
        r = c.get("/api/events", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200


def test_docs_public(tmp_path):
    with _app(tmp_path) as c:
        assert c.get("/openapi.json").status_code == 200  # reachable without login

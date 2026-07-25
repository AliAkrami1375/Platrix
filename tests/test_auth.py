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


def _login(c, user="admin", pw="secret"):
    r = c.post("/api/login", json={"username": user, "password": pw})
    return r


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_token_roundtrip():
    tok = auth.make_token("admin", "k")
    assert auth.verify_token(tok, "k") == "admin"
    assert auth.verify_token(tok, "other") is None
    assert auth.verify_token("garbage", "k") is None


def test_password_hash_roundtrip():
    h = auth.hash_password("hunter2")
    assert auth.verify_password("hunter2", h)
    assert not auth.verify_password("wrong", h)


def test_protected_requires_bearer(tmp_path):
    with _app(tmp_path) as c:
        assert c.get("/api/health").status_code == 200          # public
        assert c.get("/api/events").status_code == 401           # no token
        r = _login(c)                                            # cookie is set...
        assert r.status_code == 200
        # ...but the API is token-based: cookie alone is NOT accepted for data routes
        assert c.get("/api/events").status_code == 401
        token = r.json()["token"]
        assert c.get("/api/events", headers=_bearer(token)).status_code == 200


def test_login_invalid(tmp_path):
    with _app(tmp_path) as c:
        assert c.post("/api/login", json={"username": "admin", "password": "no"}).status_code == 401


def test_change_credentials_then_login(tmp_path):
    with _app(tmp_path) as c:
        token = _login(c).json()["token"]
        r = c.post("/api/account/password", headers=_bearer(token), json={
            "current_password": "secret", "new_username": "ali", "new_password": "newpass"})
        assert r.status_code == 200
        assert c.post("/api/login", json={"username": "admin", "password": "secret"}).status_code == 401
        assert c.post("/api/login", json={"username": "ali", "password": "newpass"}).status_code == 200


def test_api_token_access(tmp_path):
    with _app(tmp_path) as c:
        token = _login(c).json()["token"]
        tok = c.post("/api/tokens", headers=_bearer(token), json={"name": "ci"}).json()["token"]
        assert tok.startswith("pltx_")
        # an API token authorizes the API on its own
        assert c.get("/api/events", headers=_bearer(tok)).status_code == 200
        assert len(c.get("/api/tokens", headers=_bearer(tok)).json()["tokens"]) == 1


def test_csv_export(tmp_path):
    with _app(tmp_path) as c:
        token = _login(c).json()["token"]
        r = c.get("/api/events/export", headers=_bearer(token))
        assert r.status_code == 200
        assert "text/csv" in r.headers["content-type"]
        assert r.text.splitlines()[0].startswith("time,plate")


def test_docs_public(tmp_path):
    with _app(tmp_path) as c:
        assert c.get("/openapi.json").status_code == 200
        schema = c.get("/openapi.json").json()
        assert "BearerAuth" in schema["components"]["securitySchemes"]

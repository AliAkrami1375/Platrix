"""Lightweight signed-cookie authentication (no external dependencies).

A successful login issues an HMAC-signed token stored in a cookie; protected
routes verify the signature. This keeps the dashboard and API private on a
self-hosted deployment without pulling in a session library.
"""

from __future__ import annotations

import hashlib
import hmac

from platrix.config import Settings

COOKIE_NAME = "platrix_auth"

# Paths reachable without authentication (login page assets + auth endpoints).
_PUBLIC_PREFIXES = ("/static/", "/favicon")
_PUBLIC_EXACT = {"/", "/api/health", "/api/login", "/api/logout", "/api/me"}


def _sign(value: str, secret: str) -> str:
    sig = hmac.new(secret.encode(), value.encode(), hashlib.sha256).hexdigest()
    return f"{value}.{sig}"


def make_token(username: str, secret: str) -> str:
    return _sign(username, secret)


def verify_token(token: str | None, secret: str) -> str | None:
    """Return the username if the token's signature is valid, else None."""
    if not token or "." not in token:
        return None
    value, sig = token.rsplit(".", 1)
    expected = hmac.new(secret.encode(), value.encode(), hashlib.sha256).hexdigest()
    return value if hmac.compare_digest(sig, expected) else None


def check_credentials(settings: Settings, username: str, password: str) -> bool:
    return hmac.compare_digest(username, settings.auth_user) and hmac.compare_digest(
        password, settings.auth_password
    )


def is_public_path(path: str) -> bool:
    if path in _PUBLIC_EXACT:
        return True
    return any(path.startswith(p) for p in _PUBLIC_PREFIXES)

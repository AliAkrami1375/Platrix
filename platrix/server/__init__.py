"""HTTP/WebSocket server package.

``create_app`` is exposed lazily so that importing a leaf module such as
``platrix.server.auth`` does not pull in the whole application (which would
create an import cycle with the storage layer).
"""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str):
    if name == "create_app":
        from platrix.server.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

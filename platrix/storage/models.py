"""SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DetectionEvent(Base):
    """One recognized plate, persisted with time and snapshot reference."""

    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plate_text: Mapped[str] = mapped_column(String(32), index=True, default="")
    plate_text_fa: Mapped[str] = mapped_column(String(64), default="")
    detection_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    ocr_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    source: Mapped[str] = mapped_column(String(255), index=True, default="")

    bbox_x: Mapped[int] = mapped_column(Integer, default=0)
    bbox_y: Mapped[int] = mapped_column(Integer, default=0)
    bbox_w: Mapped[int] = mapped_column(Integer, default=0)
    bbox_h: Mapped[int] = mapped_column(Integer, default=0)

    snapshot_path: Mapped[str] = mapped_column(String(512), default="")

    # Direction of travel for gate/lane scenarios: "entry" | "exit" | "unknown".
    direction: Mapped[str] = mapped_column(String(16), default="unknown", index=True)
    # Watchlist match (empty when the plate is not on a list).
    matched_name: Mapped[str] = mapped_column(String(64), default="")
    matched_list: Mapped[str] = mapped_column(String(16), default="", index=True)  # white|black

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "plate_text": self.plate_text,
            "plate_text_fa": self.plate_text_fa,
            "detection_confidence": self.detection_confidence,
            "ocr_confidence": self.ocr_confidence,
            "score": self.score,
            "source": self.source,
            "direction": self.direction,
            "matched_name": self.matched_name,
            "matched_list": self.matched_list,
            "bbox": {
                "x": self.bbox_x,
                "y": self.bbox_y,
                "w": self.bbox_w,
                "h": self.bbox_h,
            },
            "snapshot_path": self.snapshot_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class WatchlistEntry(Base):
    """A named, tracked plate on the white or black list."""

    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plate_text: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(64), default="")
    list_type: Mapped[str] = mapped_column(String(16), default="white", index=True)  # white|black
    note: Mapped[str] = mapped_column(String(255), default="")
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "plate_text": self.plate_text,
            "name": self.name,
            "list_type": self.list_type,
            "note": self.note,
            "active": self.active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Camera(Base):
    """A saved video source (RTSP/HTTP stream, file, or webcam index)."""

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(80), default="")
    url: Mapped[str] = mapped_column(String(512))
    direction: Mapped[str] = mapped_column(String(16), default="unknown")  # entry|exit|unknown
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "direction": self.direction,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AccessEmail(Base):
    """An email captured at the access gate."""

    __tablename__ = "access_emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), index=True)
    user_agent: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "user_agent": self.user_agent,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

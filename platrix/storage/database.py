"""Event store: writes snapshots and persists detection events to SQLite."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
from sqlalchemy import create_engine, desc, func, select
from sqlalchemy.orm import Session, sessionmaker

from platrix.config import Settings
from platrix.core.types import PlateReading
from platrix.logging_conf import get_logger
from platrix.storage.models import Base, DetectionEvent

logger = get_logger(__name__)


class EventStore:
    """Thread-safe facade over the detection-event database and snapshots."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_dirs()
        self._engine = create_engine(
            settings.resolved_database_url(),
            connect_args={"check_same_thread": False},
            future=True,
        )
        Base.metadata.create_all(self._engine)
        self._Session: sessionmaker[Session] = sessionmaker(
            bind=self._engine, expire_on_commit=False, future=True
        )
        self._snap_lock = threading.Lock()

    # -- snapshots --------------------------------------------------------
    def _write_snapshot(self, reading: PlateReading) -> Optional[str]:
        crop = reading.detection.crop
        if crop is None or crop.size == 0:
            return None
        ts = reading.timestamp.astimezone(timezone.utc)
        day_dir = self.settings.snapshots_dir / ts.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        plate = reading.text or "unknown"
        safe_plate = "".join(c for c in plate if c.isalnum()) or "unknown"
        fname = f"{ts.strftime('%H%M%S_%f')}_{safe_plate}.jpg"
        path = day_dir / fname
        with self._snap_lock:
            cv2.imwrite(str(path), crop)
        return str(path.relative_to(self.settings.data_dir.parent))

    # -- writes -----------------------------------------------------------
    def record(self, reading: PlateReading, save_snapshot: bool = True) -> DetectionEvent:
        """Persist a reading (and its snapshot) and return the stored row."""
        snapshot_path = self._write_snapshot(reading) if save_snapshot else None
        reading.snapshot_path = snapshot_path
        x, y, w, h = reading.detection.bbox
        event = DetectionEvent(
            plate_text=reading.text,
            plate_text_fa=reading.text_fa,
            detection_confidence=reading.detection.confidence,
            ocr_confidence=reading.ocr_confidence,
            score=reading.score,
            source=reading.source,
            bbox_x=x,
            bbox_y=y,
            bbox_w=w,
            bbox_h=h,
            snapshot_path=snapshot_path or "",
            created_at=reading.timestamp,
        )
        with self._Session() as session:
            session.add(event)
            session.commit()
            session.refresh(event)
        logger.info(
            "LOGGED plate=%r score=%.2f source=%s snapshot=%s",
            reading.text or "-",
            reading.score,
            reading.source,
            snapshot_path or "-",
        )
        return event

    # -- reads ------------------------------------------------------------
    def recent(self, limit: int = 100, plate: str | None = None) -> list[dict]:
        stmt = select(DetectionEvent).order_by(desc(DetectionEvent.created_at)).limit(limit)
        if plate:
            stmt = (
                select(DetectionEvent)
                .where(DetectionEvent.plate_text.like(f"%{plate}%"))
                .order_by(desc(DetectionEvent.created_at))
                .limit(limit)
            )
        with self._Session() as session:
            return [row.to_dict() for row in session.scalars(stmt)]

    def stats(self) -> dict:
        with self._Session() as session:
            total = session.scalar(select(func.count(DetectionEvent.id))) or 0
            distinct = (
                session.scalar(
                    select(func.count(func.distinct(DetectionEvent.plate_text))).where(
                        DetectionEvent.plate_text != ""
                    )
                )
                or 0
            )
            last: DetectionEvent | None = session.scalars(
                select(DetectionEvent).order_by(desc(DetectionEvent.created_at)).limit(1)
            ).first()
        return {
            "total_events": int(total),
            "distinct_plates": int(distinct),
            "last_seen": last.created_at.isoformat() if last and last.created_at else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

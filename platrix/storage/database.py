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
from platrix.ocr.persian import to_english_digits
from platrix.storage.models import Base, DetectionEvent, WatchlistEntry

logger = get_logger(__name__)


def normalize_plate(text: str) -> str:
    """Canonical form for matching: ASCII digits, upper-case, no separators."""
    text = to_english_digits(text or "")
    return "".join(ch for ch in text if ch.isalnum()).upper()


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
        # In-memory watchlist index: normalized_plate -> (name, list_type).
        self._watch_lock = threading.Lock()
        self._watch_index: dict[str, tuple[str, str]] = {}
        self._reload_watch_index()

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

    # -- watchlist --------------------------------------------------------
    def _reload_watch_index(self) -> None:
        with self._Session() as session:
            rows = session.scalars(
                select(WatchlistEntry).where(WatchlistEntry.active.is_(True))
            ).all()
        index = {normalize_plate(r.plate_text): (r.name, r.list_type) for r in rows}
        with self._watch_lock:
            self._watch_index = index

    def match_plate(self, text: str) -> tuple[str, str]:
        """Return ``(name, list_type)`` if *text* is watchlisted, else ``("", "")``."""
        if not text:
            return "", ""
        with self._watch_lock:
            return self._watch_index.get(normalize_plate(text), ("", ""))

    def add_watch(
        self, plate: str, name: str = "", list_type: str = "white", note: str = ""
    ) -> dict:
        plate_norm = normalize_plate(plate)
        if not plate_norm:
            raise ValueError("Plate is empty")
        if list_type not in ("white", "black"):
            raise ValueError("list_type must be 'white' or 'black'")
        with self._Session() as session:
            entry = session.scalars(
                select(WatchlistEntry).where(WatchlistEntry.plate_text == plate_norm)
            ).first()
            if entry is None:
                entry = WatchlistEntry(plate_text=plate_norm)
                session.add(entry)
            entry.name = name
            entry.list_type = list_type
            entry.note = note
            entry.active = True
            session.commit()
            session.refresh(entry)
            data = entry.to_dict()
        self._reload_watch_index()
        logger.info("WATCHLIST +%s %r name=%r", list_type, plate_norm, name)
        return data

    def list_watch(self, list_type: str | None = None) -> list[dict]:
        stmt = select(WatchlistEntry).order_by(desc(WatchlistEntry.created_at))
        if list_type in ("white", "black"):
            stmt = stmt.where(WatchlistEntry.list_type == list_type)
        with self._Session() as session:
            return [r.to_dict() for r in session.scalars(stmt)]

    def delete_watch(self, entry_id: int) -> bool:
        with self._Session() as session:
            entry = session.get(WatchlistEntry, entry_id)
            if entry is None:
                return False
            session.delete(entry)
            session.commit()
        self._reload_watch_index()
        return True

    # -- writes -----------------------------------------------------------
    def record(
        self,
        reading: PlateReading,
        save_snapshot: bool = True,
        direction: str = "unknown",
    ) -> DetectionEvent:
        """Persist a reading (and its snapshot) and return the stored row."""
        snapshot_path = self._write_snapshot(reading) if save_snapshot else None
        reading.snapshot_path = snapshot_path
        matched_name, matched_list = self.match_plate(reading.text)
        x, y, w, h = reading.detection.bbox
        event = DetectionEvent(
            plate_text=reading.text,
            plate_text_fa=reading.text_fa,
            detection_confidence=reading.detection.confidence,
            ocr_confidence=reading.ocr_confidence,
            score=reading.score,
            source=reading.source,
            direction=direction if direction in ("entry", "exit") else "unknown",
            matched_name=matched_name,
            matched_list=matched_list,
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
        flag = f" [{matched_list.upper()}:{matched_name}]" if matched_list else ""
        logger.info(
            "LOGGED plate=%r score=%.2f dir=%s source=%s%s",
            reading.text or "-",
            reading.score,
            event.direction,
            reading.source,
            flag,
        )
        return event

    # -- reads ------------------------------------------------------------
    def recent(
        self,
        limit: int = 100,
        plate: str | None = None,
        direction: str | None = None,
        list_type: str | None = None,
    ) -> list[dict]:
        stmt = select(DetectionEvent)
        if plate:
            needle = normalize_plate(plate)
            stmt = stmt.where(DetectionEvent.plate_text.like(f"%{needle}%"))
        if direction in ("entry", "exit", "unknown"):
            stmt = stmt.where(DetectionEvent.direction == direction)
        if list_type in ("white", "black"):
            stmt = stmt.where(DetectionEvent.matched_list == list_type)
        stmt = stmt.order_by(desc(DetectionEvent.created_at)).limit(limit)
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

            def _count(**filters) -> int:
                stmt = select(func.count(DetectionEvent.id))
                for col, val in filters.items():
                    stmt = stmt.where(getattr(DetectionEvent, col) == val)
                return int(session.scalar(stmt) or 0)

            entries = _count(direction="entry")
            exits = _count(direction="exit")
            white_hits = _count(matched_list="white")
            black_hits = _count(matched_list="black")
            watch_white = int(
                session.scalar(
                    select(func.count(WatchlistEntry.id)).where(
                        WatchlistEntry.list_type == "white"
                    )
                )
                or 0
            )
            watch_black = int(
                session.scalar(
                    select(func.count(WatchlistEntry.id)).where(
                        WatchlistEntry.list_type == "black"
                    )
                )
                or 0
            )
        return {
            "total_events": int(total),
            "distinct_plates": int(distinct),
            "entries": entries,
            "exits": exits,
            "whitelist_hits": white_hits,
            "blacklist_hits": black_hits,
            "watchlist_white": watch_white,
            "watchlist_black": watch_black,
            "last_seen": last.created_at.isoformat() if last and last.created_at else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

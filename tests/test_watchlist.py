import numpy as np

from platrix.config import Settings
from platrix.core.types import PlateDetection, PlateReading
from platrix.storage import EventStore
from platrix.storage.database import normalize_plate


def _store(tmp_path) -> EventStore:
    return EventStore(
        Settings(data_dir=tmp_path / "d", snapshots_dir=tmp_path / "d" / "snap")
    )


def _reading(text: str) -> PlateReading:
    det = PlateDetection(0, 0, 40, 12, 0.9, np.zeros((12, 40, 3), np.uint8))
    return PlateReading(detection=det, text=text, text_fa=text, ocr_confidence=0.8)


def test_normalize_plate():
    assert normalize_plate("12 ب 34567") == "12" + "ب".upper() + "34567"
    assert normalize_plate("12-b-345") == "12B345"
    assert normalize_plate("۱۲۳") == "123"


def test_add_and_match_watchlist(tmp_path):
    store = _store(tmp_path)
    store.add_watch("12B34567", name="Ali", list_type="black")
    name, list_type = store.match_plate("12B34567")
    assert name == "Ali"
    assert list_type == "black"
    # Unknown plate does not match.
    assert store.match_plate("99Z00000") == ("", "")


def test_record_tags_match_and_direction(tmp_path):
    store = _store(tmp_path)
    store.add_watch("12B34567", name="Staff", list_type="white")
    event = store.record(_reading("12B34567"), save_snapshot=False, direction="entry")
    assert event.matched_list == "white"
    assert event.matched_name == "Staff"
    assert event.direction == "entry"


def test_search_filters(tmp_path):
    store = _store(tmp_path)
    store.add_watch("11A11111", name="X", list_type="black")
    store.record(_reading("11A11111"), save_snapshot=False, direction="exit")
    store.record(_reading("22B22222"), save_snapshot=False, direction="entry")

    assert len(store.recent(list_type="black")) == 1
    assert len(store.recent(direction="entry")) == 1
    assert len(store.recent(plate="11A")) == 1


def test_delete_watch(tmp_path):
    store = _store(tmp_path)
    entry = store.add_watch("33C33333", list_type="white")
    assert store.delete_watch(entry["id"]) is True
    assert store.match_plate("33C33333") == ("", "")

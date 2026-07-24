from platrix.config import Settings
from platrix.storage import EventStore


def _store(tmp_path) -> EventStore:
    return EventStore(
        Settings(data_dir=tmp_path / "d", snapshots_dir=tmp_path / "d" / "snap")
    )


def test_camera_crud(tmp_path):
    store = _store(tmp_path)
    cam = store.add_camera("Front Gate", "rtsp://cam/stream", "entry")
    assert cam["name"] == "Front Gate"
    assert cam["direction"] == "entry"
    assert len(store.list_cameras()) == 1
    assert store.delete_camera(cam["id"]) is True
    assert store.list_cameras() == []


def test_camera_defaults(tmp_path):
    store = _store(tmp_path)
    cam = store.add_camera("", "0", "sideways")
    assert cam["name"] == "0"  # falls back to url
    assert cam["direction"] == "unknown"  # invalid dir normalized


def test_date_range_filter(tmp_path):
    import numpy as np

    from platrix.core.types import PlateDetection, PlateReading

    store = _store(tmp_path)
    det = PlateDetection(0, 0, 40, 12, 0.9, np.zeros((12, 40, 3), np.uint8))
    store.record(PlateReading(det, "11A", "11A", 0.5), save_snapshot=False)

    assert len(store.recent(date_from="2000-01-01")) == 1
    assert len(store.recent(date_from="2099-01-01")) == 0
    assert len(store.recent(date_to="2000-01-01")) == 0

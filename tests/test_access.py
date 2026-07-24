import pytest

from platrix.config import Settings
from platrix.storage import EventStore


def _store(tmp_path) -> EventStore:
    return EventStore(
        Settings(data_dir=tmp_path / "d", snapshots_dir=tmp_path / "d" / "snap")
    )


def test_record_valid_email(tmp_path):
    store = _store(tmp_path)
    row = store.record_email("User@Example.com", user_agent="pytest")
    assert row["email"] == "user@example.com"  # normalized lower-case
    assert len(store.list_emails()) == 1


@pytest.mark.parametrize("bad", ["", "nope", "a@b", "a@b.", "@x.com"])
def test_reject_invalid_email(tmp_path, bad):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.record_email(bad)

"""Regression checks for persistent guest ID and conversation associations."""

import json

from robo_cup_robot.task_manager import main as receptionist


def test_guest_details_update_existing_logical_identity(tmp_path, monkeypatch):
    records = tmp_path / "guest_records.json"
    monkeypatch.setattr(receptionist, "GUEST_RECORDS_FILE", records)

    guest_id = receptionist.next_guest_id()
    assert guest_id == "guest_1"
    receptionist.save_guest_record(guest_id, "Alice")
    receptionist.save_guest_record(guest_id, "Alice", "gardening", "tea")

    saved = json.loads(records.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["guest_id"] == "guest_1"
    assert saved[0]["name"] == "Alice"
    assert saved[0]["interest"] == "gardening"
    assert saved[0]["favorite_drink"] == "tea"
    assert receptionist.next_guest_id() == "guest_2"

"""Receptionist monitor preserves camera and guest identity status."""

import json
from types import SimpleNamespace

from robo_cup_robot.vision.receptionist_monitor import ReceptionistMonitor


def test_identity_readiness_requires_fresh_camera_and_loaded_model():
    monitor = ReceptionistMonitor()
    monitor._on_vision_status(SimpleNamespace(data=json.dumps({
        "camera_ready": True,
        "identity_model_ready": True,
        "vlm_status": "ready",
    })))
    assert monitor.wait_for_identity_ready(timeout=0.01)


def test_current_guest_identity_uses_logical_id():
    monitor = ReceptionistMonitor()
    monitor._on_message(SimpleNamespace(data=json.dumps({
        "people": [{
            "id": "person_track_42",
            "tracker_id": 42,
            "guest_id": "guest_1",
            "identity_state": "REID_RECOVERED",
        }],
    })))
    result = monitor.wait_for_guest_identities(("guest_1",), timeout=0.01)
    assert result["guest_1"][0]["identity_state"] == "REID_RECOVERED"

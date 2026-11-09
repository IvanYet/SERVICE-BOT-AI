"""Offline behavior checks for conservative multi-guest ReID association."""

import numpy as np

from robo_cup_robot.vision.guest_identity import GuestIdentityManager


def vector(index):
    value = np.zeros(512, dtype=np.float32)
    value[index] = 1.0
    return value


def enrollment_samples(index):
    base = vector(index)
    # Small deterministic variations model a frozen multi-view gallery.
    samples = []
    for j in range(10):
        item = base.copy()
        item[(index + 20 + j) % len(item)] = 0.05
        samples.append(item)
    return samples


def update(manager, track_id, embedding, now):
    return manager.observe([{"track_id": track_id, "embedding": embedding}], now)[0]


def test_single_guest_enrollment_and_stable_identity():
    manager = GuestIdentityManager()
    manager.enroll("guest_1", 7, enrollment_samples(1), 0.0)
    result = update(manager, 7, vector(1), 1.0)
    assert result["guest_id"] == "guest_1"
    assert result["state"] == "TRACKING"


def test_multiple_guest_enrollment_and_distinct_gallery_binding():
    manager = GuestIdentityManager()
    manager.enroll("guest_1", 7, enrollment_samples(1), 0.0)
    manager.enroll("guest_2", 8, enrollment_samples(2), 0.0)
    assert manager.track_to_guest == {7: "guest_1", 8: "guest_2"}


def test_enroll_new_guest_while_known_guest_remains_visible():
    manager = GuestIdentityManager()
    candidate = manager.select_enrollment_candidate(
        [7, 22],
        [
            {"track_id": 7, "state": "TRACKING", "guest_id": "guest_1"},
            {"track_id": 22, "state": "NEW_PERSON", "guest_id": None},
        ],
    )
    assert candidate == 22


def test_enrollment_refuses_multiple_unknown_or_unresolved_people():
    select = GuestIdentityManager.select_enrollment_candidate
    assert select(
        [22, 23],
        [{"track_id": 22, "state": "NEW_PERSON"}, {"track_id": 23, "state": "NEW_PERSON"}],
    ) is None
    assert select(
        [7, 22],
        [{"track_id": 7, "state": "REID_CANDIDATE"}, {"track_id": 22, "state": "NEW_PERSON"}],
    ) is None
    # A previously visible guest track cannot be relabeled as the new arrival
    # just because appearance matching is temporarily weak.
    assert select(
        [7], [{"track_id": 7, "state": "NEW_PERSON"}], excluded_track_ids={7}
    ) is None
    assert select(
        [7, 22],
        [{"track_id": 7, "state": "IDENTITY_AMBIGUOUS"}, {"track_id": 22, "state": "NEW_PERSON"}],
        excluded_track_ids={7},
    ) == 22


def test_temporary_track_id_change_requires_repeated_reid():
    manager = GuestIdentityManager()
    manager.enroll("guest_1", 7, enrollment_samples(1), 0.0)
    assert manager.observe([], 1.0) == []
    states = [update(manager, 31, vector(1), t)["state"] for t in (1.1, 1.6, 2.1)]
    assert states == ["REID_CANDIDATE", "REID_CANDIDATE", "REID_RECOVERED"]
    assert manager.track_to_guest == {31: "guest_1"}


def test_short_occlusion_keeps_logical_guest():
    manager = GuestIdentityManager()
    manager.enroll("guest_1", 7, enrollment_samples(1), 0.0)
    manager.observe([], 0.3)
    result = update(manager, 7, vector(1), 0.5)
    assert result["guest_id"] == "guest_1"
    assert result["state"] == "TRACKING"


def test_long_disappearance_can_recover_without_reusing_old_tracker_id():
    manager = GuestIdentityManager()
    manager.enroll("guest_1", 7, enrollment_samples(1), 0.0)
    manager.observe([], 70.0)
    assert manager.guests["guest_1"]["state"] == "LOST"
    states = [update(manager, 31, vector(1), t)["state"] for t in (70.1, 70.6, 71.1)]
    assert states[-1] == "REID_RECOVERED"


def test_two_guests_cross_and_recover_to_their_own_logical_ids():
    manager = GuestIdentityManager()
    manager.enroll("guest_1", 7, enrollment_samples(1), 0.0)
    manager.enroll("guest_2", 8, enrollment_samples(2), 0.0)
    # Both originals disappear; detector order is deliberately reversed.
    for t in (1.0, 1.5, 2.0):
        manager.observe([
            {"track_id": 22, "embedding": vector(2)},
            {"track_id": 11, "embedding": vector(1)},
        ], t)
    assert manager.track_to_guest == {22: "guest_2", 11: "guest_1"}


def test_similar_looking_guests_are_reported_ambiguous():
    manager = GuestIdentityManager(recover_threshold=0.80, margin=0.06)
    common = vector(1)
    manager.enroll("guest_1", 7, [common] * 10, 0.0)
    manager.enroll("guest_2", 8, [common] * 10, 0.0)
    manager.observe([], 1.0)
    result = update(manager, 20, common, 1.1)
    assert result["state"] == "IDENTITY_AMBIGUOUS"
    assert result["guest_id"] is None


def test_unknown_person_is_not_assigned_a_guest_id():
    manager = GuestIdentityManager()
    manager.enroll("guest_1", 7, enrollment_samples(1), 0.0)
    manager.observe([], 1.0)
    result = update(manager, 20, vector(40), 1.1)
    assert result["state"] == "NEW_PERSON"
    assert result["guest_id"] is None


def test_low_confidence_reid_does_not_recover():
    manager = GuestIdentityManager(recover_threshold=0.86)
    manager.enroll("guest_1", 7, enrollment_samples(1), 0.0)
    manager.observe([], 1.0)
    result = update(manager, 20, vector(40), 1.1)
    assert result["state"] == "NEW_PERSON"
    assert 20 not in manager.track_to_guest


def test_frame_interruption_and_duplicate_id_prevention():
    manager = GuestIdentityManager()
    manager.enroll("guest_1", 7, enrollment_samples(1), 0.0)
    manager.observe([], 1.0)  # fresh camera frame with no detections
    results = manager.observe([
        {"track_id": 7, "embedding": vector(1)},
        {"track_id": 20, "embedding": vector(1)},
    ], 1.2)
    assert sum(item["guest_id"] == "guest_1" for item in results) == 1
    assert manager.track_to_guest.get(20) is None


def test_contradictory_reused_tracker_id_stays_quarantined():
    manager = GuestIdentityManager()
    manager.enroll("guest_1", 7, enrollment_samples(1), 0.0)
    mismatch = update(manager, 7, vector(40), 1.0)
    assert mismatch["state"] == "IDENTITY_AMBIGUOUS"
    repeated = update(manager, 7, vector(40), 1.5)
    assert repeated["state"] == "IDENTITY_AMBIGUOUS"
    assert repeated["guest_id"] is None
    assert manager.track_to_guest.get(7) is None


def test_guest_logical_id_survives_reid_recovery():
    manager = GuestIdentityManager()
    manager.enroll("guest_1", 7, enrollment_samples(1), 0.0)
    manager.observe([], 1.0)
    for t in (1.1, 1.6, 2.1):
        result = update(manager, 31, vector(1), t)
    assert result["guest_id"] == "guest_1"
    assert result["state"] == "REID_RECOVERED"

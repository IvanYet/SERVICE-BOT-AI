"""Conservative multi-guest identity association from person appearance only.

The manager consumes temporary person tracker IDs and normalized appearance
embeddings. A gallery is changed only during explicit enrollment. ReID is a
hypothesis that needs a score margin and repeated observations; it is never a
real-world name verifier.
"""

import math

import numpy as np


def _unit(vector):
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.size < 64 or not np.isfinite(value).all():
        raise ValueError("invalid appearance embedding")
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise ValueError("zero appearance embedding")
    return value / norm


class GuestIdentityManager:
    """Bind logical guest IDs to transient tracker IDs using frozen galleries."""

    def __init__(self, gallery_size=10, recover_threshold=0.60,
                 track_threshold=0.60, margin=0.06, min_hits=3, max_missing_s=60.0):
        if (gallery_size < 2 or not 0 < track_threshold < 1
                or not 0 < recover_threshold < 1 or margin < 0 or min_hits < 2):
            raise ValueError("invalid identity-manager configuration")
        self.gallery_size = int(gallery_size)
        self.recover_threshold = float(recover_threshold)
        self.track_threshold = float(track_threshold)
        self.margin = float(margin)
        self.min_hits = int(min_hits)
        self.max_missing_s = float(max_missing_s)
        self.guests = {}             # logical ID -> frozen embedding gallery + state
        self.track_to_guest = {}     # current tracker ID -> logical guest ID
        self.pending = {}            # tracker ID -> (logical ID, hit count, last time)
        self.quarantined_tracks = {} # tracker ID -> expiry time after a contradiction
        self.embedding_size = None

    def enroll(self, guest_id, track_id, embeddings, now):
        """Explicitly enroll a guest from multiple stable views of one person."""
        if not isinstance(guest_id, str) or not guest_id.strip():
            raise ValueError("guest_id is required")
        guest_id = guest_id.strip()
        if guest_id in self.guests:
            raise ValueError(f"guest already enrolled: {guest_id}")
        if track_id in self.track_to_guest:
            raise ValueError("tracker ID already belongs to an enrolled guest")
        gallery = [_unit(item) for item in embeddings]
        if len(gallery) < 2 or len(gallery) > self.gallery_size:
            raise ValueError("enrollment requires 2..gallery_size embeddings")
        dims = {len(item) for item in gallery}
        if len(dims) != 1 or (self.embedding_size is not None and len(gallery[0]) != self.embedding_size):
            raise ValueError("appearance embedding dimensions do not match")
        self.embedding_size = len(gallery[0])
        self.guests[guest_id] = {
            "gallery": tuple(item.copy() for item in gallery),
            "track_id": track_id,
            "state": "TRACKING",
            "last_seen": float(now),
            "confidence": 1.0,
        }
        self.track_to_guest[track_id] = guest_id
        return guest_id

    def _score(self, gallery, embedding):
        vector = _unit(embedding)
        if len(vector) != self.embedding_size:
            raise ValueError("appearance embedding dimensions do not match")
        similarities = sorted(float(np.dot(sample, vector)) for sample in gallery)
        count = max(2, len(similarities) // 3)
        return sum(similarities[-count:]) / count

    @staticmethod
    def select_enrollment_candidate(person_track_ids, identity_results,
                                    excluded_track_ids=(), distance_by_track=None,
                                    preferred_track_id=None, locked_track_id=None):
        """Select the nearest unknown track when multiple people are visible.

        Existing guests and explicitly excluded tracks are not enrolled again.
        Multiple new people require valid depth for each candidate. Once chosen,
        the candidate stays selected while visible to avoid switching mid-enrollment.
        """
        track_ids = list(person_track_ids)
        excluded = set(excluded_track_ids)
        if not track_ids or any(track_id is None for track_id in track_ids):
            return None
        results = {item.get("track_id"): item for item in identity_results}
        if any(track_id not in results for track_id in track_ids):
            return None
        candidates = [
            track_id for track_id in track_ids
            if track_id not in excluded and results[track_id].get("state") == "NEW_PERSON"
        ]
        if locked_track_id is not None:
            return locked_track_id if locked_track_id in candidates else None
        if not candidates:
            return None
        if any(
            results[track_id].get("state") not in (
                "TRACKING", "REID_RECOVERED", "NEW_PERSON"
            )
            for track_id in track_ids if track_id not in candidates and track_id not in excluded
        ):
            return None
        if preferred_track_id in candidates:
            return preferred_track_id
        if len(candidates) == 1:
            return candidates[0]

        distances = distance_by_track or {}
        if any(
            track_id not in distances
            or not math.isfinite(float(distances[track_id]))
            or float(distances[track_id]) <= 0.0
            for track_id in candidates
        ):
            return None
        return min(candidates, key=lambda track_id: float(distances[track_id]))

    def observe(self, observations, now):
        """Update states from fresh detections; return per-track identity results.

        Each observation has ``track_id`` and optionally ``embedding``. Only
        guests currently missing can be recovered onto an unbound track. A
        guest already visible on another track cannot be claimed a second time.
        """
        now = float(now)
        seen_tracks = set()
        results = []
        # Mark existing logical identities missing before applying this frame.
        for record in self.guests.values():
            if now - record["last_seen"] > self.max_missing_s:
                record["state"] = "LOST"
            elif record["state"] == "TRACKING":
                record["state"] = "TEMPORARILY_MISSING"

        # Reserve every identity whose existing tracker is visible before
        # evaluating unknown tracks, independent of detector ordering.
        active_guest_ids = set()
        ambiguous_tracks = set()
        # Check already-bound tracks first. If a reused tracker ID has a
        # contradictory appearance, quarantine and unbind that track so a
        # separately visible matching guest can be considered for recovery.
        for obs in observations:
            track_id = obs.get("track_id")
            guest_id = self.track_to_guest.get(track_id)
            if guest_id is None:
                continue
            record = self.guests[guest_id]
            embedding = obs.get("embedding")
            score = None
            if embedding is not None:
                try:
                    score = self._score(record["gallery"], embedding)
                except ValueError:
                    pass
            if score is not None and score < self.track_threshold:
                self.track_to_guest.pop(track_id, None)
                self.quarantined_tracks[track_id] = now + self.max_missing_s
                record["state"] = "IDENTITY_AMBIGUOUS"
                record["confidence"] = score
                ambiguous_tracks.add(track_id)
                continue
            active_guest_ids.add(guest_id)
        for obs in observations:
            track_id = obs.get("track_id")
            if track_id is None or track_id in seen_tracks:
                continue
            seen_tracks.add(track_id)
            if track_id in ambiguous_tracks:
                results.append(self._result(track_id, None, "IDENTITY_AMBIGUOUS", 0.0))
                continue
            if track_id in self.quarantined_tracks:
                if now < self.quarantined_tracks[track_id]:
                    results.append(self._result(track_id, None, "IDENTITY_AMBIGUOUS", 0.0))
                    continue
                del self.quarantined_tracks[track_id]
            guest_id = self.track_to_guest.get(track_id)
            embedding = obs.get("embedding")
            if guest_id is not None:
                record = self.guests[guest_id]
                record["last_seen"] = now
                record["state"] = "TRACKING"
                record["confidence"] = (
                    self._score(record["gallery"], embedding)
                    if embedding is not None else 1.0
                )
                results.append(self._result(track_id, guest_id, "TRACKING", record["confidence"]))
                continue

            if embedding is None:
                results.append(self._result(track_id, None, "NEW_PERSON", 0.0))
                continue

            candidates = []
            for candidate_id, record in self.guests.items():
                if candidate_id in active_guest_ids:
                    continue
                try:
                    candidates.append((candidate_id, self._score(record["gallery"], embedding)))
                except ValueError:
                    continue
            candidates.sort(key=lambda item: item[1], reverse=True)
            if not candidates:
                self.pending.pop(track_id, None)
                results.append(self._result(track_id, None, "NEW_PERSON", 0.0))
                continue

            best_id, best_score = candidates[0]
            second_score = candidates[1][1] if len(candidates) > 1 else -1.0
            if best_score < self.recover_threshold:
                self.pending.pop(track_id, None)
                results.append(self._result(track_id, None, "NEW_PERSON", best_score))
            elif best_score - second_score < self.margin:
                self.pending.pop(track_id, None)
                results.append(self._result(track_id, None, "IDENTITY_AMBIGUOUS", best_score))
            else:
                old = self.pending.get(track_id)
                hits = old[1] + 1 if old and old[0] == best_id and now - old[2] <= 1.5 else 1
                self.pending[track_id] = (best_id, hits, now)
                if hits >= self.min_hits and best_id not in active_guest_ids:
                    previous_track = self.guests[best_id]["track_id"]
                    if previous_track != track_id and self.track_to_guest.get(previous_track) == best_id:
                        del self.track_to_guest[previous_track]
                    self.track_to_guest[track_id] = best_id
                    self.pending.pop(track_id, None)
                    record = self.guests[best_id]
                    record["track_id"] = track_id
                    record["last_seen"] = now
                    record["state"] = "REID_RECOVERED"
                    record["confidence"] = best_score
                    active_guest_ids.add(best_id)
                    results.append(self._result(track_id, best_id, "REID_RECOVERED", best_score))
                else:
                    results.append(self._result(track_id, None, "REID_CANDIDATE",
                                                best_score, candidate=best_id,
                                                hits=hits))

        # A temporarily absent identity remains associated with its last track
        # only until it ages out; on expiration free that tracker label.
        for track_id, guest_id in list(self.track_to_guest.items()):
            record = self.guests[guest_id]
            if record["state"] == "LOST" and track_id not in seen_tracks:
                del self.track_to_guest[track_id]
        for track_id in list(self.pending):
            if track_id not in seen_tracks and now - self.pending[track_id][2] > 1.5:
                del self.pending[track_id]
        return results

    @staticmethod
    def _result(track_id, guest_id, state, confidence, **extra):
        result = {
            "track_id": track_id,
            "guest_id": guest_id,
            "state": state,
            "confidence": round(float(confidence), 4) if math.isfinite(float(confidence)) else 0.0,
        }
        result.update(extra)
        return result

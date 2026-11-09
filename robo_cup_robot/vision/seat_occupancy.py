"""Stable chair/couch observations for the Receptionist vision node."""


def _overlap_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))


def _area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _box_iou(box_a, box_b):
    overlap = _overlap_area(box_a, box_b)
    union = _area(box_a) + _area(box_b) - overlap
    return overlap / union if union else 0.0


class SeatOccupancy:
    """Keep physical seat IDs and debounce changes on distinct camera frames."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.last_frame_at = None
        self.tuning = {
            "seat_confirm_s": 1.0,
            "seat_hold_s": 1.5,
            "seat_keep_s": 4.0,
            "seat_overlap_min": 0.20,
            "seat_fallback_overlap": 0.55,
        }

    def configure(self, values):
        # Swap a whole dict so the worker sees a consistent set per frame.
        self.tuning = {
            key: values[key]
            for key in self.tuning
        }

    def update(self, detections, suppressed_ids, nested_boxes, people, poses, frame_shape, now):
        tuning = self.tuning
        if self.last_frame_at is not None and now - self.last_frame_at > tuning["seat_hold_s"]:
            for track in self.tracks.values():
                track.update(sitting_since=None, clear_since=None, status="uncertain seat")
        self.last_frame_at = now

        for seat_id, track in list(self.tracks.items()):
            if (track["detector_id"] in suppressed_ids
                    or now - max(track["last_detection"], track["last_sitting"]) > tuning["seat_keep_s"]):
                del self.tracks[seat_id]

        # The detector's numeric ID is only a hint. Spatial matches preserve
        # the ID Task Manager is waiting on even if BoT-SORT reallocates it.
        matches = []
        for index, (*bbox, detector_id, class_id) in enumerate(detections):
            for seat_id, track in self.tracks.items():
                same_detector = detector_id is not None and detector_id == track["detector_id"]
                overlap = _box_iou(bbox, track["bbox"])
                same_class = track["class_id"] == class_id
                if (same_class and overlap >= 0.25) or (same_detector and overlap >= 0.65):
                    matches.append((overlap + 0.10 * same_detector, index, seat_id))

        assigned = {}
        visible = set()
        for _, index, seat_id in sorted(matches, reverse=True):
            if index not in assigned and seat_id not in visible:
                assigned[index] = seat_id
                visible.add(seat_id)

        for index, (*bbox, detector_id, class_id) in enumerate(detections):
            seat_id = assigned.get(index)
            if seat_id is None:
                seat_id = f"seat_{self.next_id}"
                self.next_id += 1
                self.tracks[seat_id] = {
                    "class_id": class_id, "last_sitting": float("-inf"), "sitting_since": None,
                    "clear_since": None, "status": "uncertain seat",
                }
            track = self.tracks[seat_id]
            track.update(bbox=tuple(bbox), detector_id=detector_id, last_detection=now)
            visible.add(seat_id)
            track["nested"] = tuple(bbox) in nested_boxes

        sitting = {}
        uncertain = set()
        height, width = frame_shape[:2]
        for index, (x1, y1, x2, y2, _) in enumerate(people):
            posture, confidence = poses.get(index, ("unknown", 0.0))
            if posture == "standing":
                continue
            bbox = (x1, y1, x2, y2)
            # The earlier Receptionist detector's aspect/coverage check is a
            # fallback when seat occlusion hides the ankles or pose is absent.
            # A crop cut off at the image edge is insufficient for this test.
            compact = (
                x1 > 1 and y1 > 1 and x2 < width - 1 and y2 < height - 1
                and x2 > x1 and 0 < y2 - y1 <= 1.85 * (x2 - x1)
            )
            best_seat, best_rank = None, (-1, 0.0)
            for seat_id, track in self.tracks.items():
                sx1, _, sx2, _ = track["bbox"]
                if not sx1 <= (x1 + x2) / 2 <= sx2:
                    continue
                intersection = _overlap_area(bbox, track["bbox"])
                ratio = intersection / max(1, _area(track["bbox"]))
                couch_overlap = track["class_id"] == 57 and intersection >= 0.35 * max(1, _area(bbox))
                if ratio < tuning["seat_overlap_min"] and not couch_overlap:
                    continue
                # A missed chair may still be behind a sitting person. Require
                # stronger overlap to extend it beyond a brief detector gap.
                if seat_id not in visible and ratio <= tuning["seat_fallback_overlap"] and now - track["last_detection"] > tuning["seat_hold_s"]:
                    continue
                rank = (ratio + 0.08 * (seat_id in visible), ratio)
                if rank > best_rank:
                    best_seat, best_rank = seat_id, rank
            if best_seat is None:
                continue
            if posture == "sitting" or (compact and best_rank[1] > tuning["seat_fallback_overlap"]):
                sitting[best_seat] = max(sitting.get(best_seat, 0.0), confidence if posture == "sitting" else 0.0)
            else:
                uncertain.add(best_seat)

        for seat_id, track in self.tracks.items():
            if seat_id in sitting:
                if track["sitting_since"] is None or now - track["last_sitting"] > tuning["seat_hold_s"]:
                    track["sitting_since"] = now
                    track["status"] = "uncertain seat"
                track["last_sitting"] = now
                track["clear_since"] = None
                if now - track["sitting_since"] >= tuning["seat_confirm_s"]:
                    track["status"] = "occupied seat"
                elif track["status"] != "occupied seat":
                    track["status"] = "uncertain seat"
            else:
                clear = seat_id in visible and seat_id not in uncertain and not track["nested"]
                if clear and track["clear_since"] is None:
                    track["clear_since"] = now
                elif not clear:
                    track["clear_since"] = None
                if now - track["last_sitting"] > tuning["seat_hold_s"]:
                    track["sitting_since"] = None
                    if clear and (track["status"] == "empty seat"
                                  or now - track["clear_since"] >= tuning["seat_confirm_s"]):
                        track["status"] = "empty seat"
                    else:
                        track["status"] = "uncertain seat"
            track["pose_confidence"] = sitting.get(seat_id, 0.0) if track["status"] == "occupied seat" else 0.0
        return self.tracks

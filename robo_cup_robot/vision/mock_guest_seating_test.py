#!/usr/bin/env python3
"""Mock two-guest registration and seat-identity test; does not start navigation."""

import argparse
import hashlib
import os
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

try:
    import ollama
except ImportError:
    ollama = None

from robo_cup_robot.speech import speech_config
from robo_cup_robot.speech.mouth import Mouth
from robo_cup_robot.vision.guest_identity import GuestIdentityManager
from robo_cup_robot.vision.guest_reid_lab import (
    _arm_return_test,
    _load_gallery,
    _save_gallery,
)
from robo_cup_robot.vision.image_utils import image_msg_to_numpy
from robo_cup_robot.vision.osnet_encoder import OsnetEncoder
from robo_cup_robot.vision.receptionist_vision_node import (
    _box_iou,
    _suppress_nested_seat_boxes,
    classify_sitting_pose,
)
from robo_cup_robot.vision.seat_occupancy import SeatOccupancy


ROOT = Path(__file__).resolve().parents[2]
PERSON_MODEL = ROOT / "yolov8n.pt"
POSE_MODEL = ROOT / "robo_cup_robot/vision/models/yolo11n-pose.pt"
OSNET_MODEL = ROOT / "robo_cup_robot/vision/models/osnet_x0_25_msmt17.onnx"
DEFAULT_GALLERY = ROOT / "guest_reid_lab_gallery.json"
IMAGE_TOPIC = "/camera/camera/color/image_raw"
NO_PERSON_RESET_S = 2.0
ENROLL_SAMPLE_COUNT = 10
ENROLL_SAMPLE_INTERVAL_S = 0.55
SEAT_ID_CONFIRM_S = 0.75


class CameraSource(Node):
    def __init__(self):
        super().__init__("mock_guest_seating_test")
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, 5)

    def _on_image(self, message):
        try:
            frame = image_msg_to_numpy(message, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Could not decode camera frame: {exc}")
            return
        with self.lock:
            self.frame = frame
            self.frame_id += 1

    def latest(self):
        with self.lock:
            if self.frame is None:
                return None, self.frame_id
            return self.frame.copy(), self.frame_id


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _crop(frame, box):
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 - x1 < 35 or y2 - y1 < 65:
        return None
    return frame[y1:y2, x1:x2].copy()


def _pose_by_person(frame, people, pose_model):
    """Associate YOLO pose outputs with person boxes, as in the main vision node."""
    poses = {}
    if people:
        try:
            result = pose_model(frame, verbose=False, conf=0.35)[0]
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            points = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []
            confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else []
            used = set()
            for index, person in enumerate(people):
                best_index, best_iou = None, 0.0
                for pose_index, pose_box in enumerate(boxes):
                    if pose_index in used:
                        continue
                    score = _box_iou(person["box"], tuple(map(int, pose_box)))
                    if score > best_iou:
                        best_index, best_iou = pose_index, score
                if best_index is not None and best_iou >= 0.25:
                    used.add(best_index)
                    pose_conf = confidences[best_index] if len(confidences) else None
                    poses[index] = classify_sitting_pose(points[best_index], pose_conf)
        except Exception as exc:
            print(f"[POSE] Inference failed: {exc}")
    for index in range(len(people)):
        poses.setdefault(index, ("unknown", 0.0))
    return poses


def _seat_for_person(person_box, seat_tracks):
    """Return the best seat overlapping a person, using the seat monitor geometry."""
    x1, y1, x2, y2 = person_box
    person_area = max(1, (x2 - x1) * (y2 - y1))
    center_x = (x1 + x2) / 2.0
    best = None
    best_ratio = 0.0
    for seat_id, track in seat_tracks.items():
        sx1, sy1, sx2, sy2 = track["bbox"]
        if not sx1 <= center_x <= sx2:
            continue
        overlap = max(0, min(x2, sx2) - max(x1, sx1)) * max(
            0, min(y2, sy2) - max(y1, sy1)
        )
        ratio = overlap / max(1, (sx2 - sx1) * (sy2 - sy1))
        couch_overlap = track["class_id"] == 57 and overlap >= 0.35 * person_area
        if ratio < 0.20 and not couch_overlap:
            continue
        if ratio > best_ratio:
            best, best_ratio = seat_id, ratio
    return best


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--guest1-label", default="Guest 1")
    parser.add_argument("--guest2-label", default="Guest 2")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--margin", type=float, default=0.06)
    parser.add_argument("--llm-model", default=os.getenv(
        "RECEPTIONIST_LLM_MODEL", speech_config.OLLAMA_MODEL
    ))
    parser.add_argument("--no-speech", action="store_true",
                        help="Print the LLM announcement without Piper audio")
    return parser.parse_args()


def _llm_worker(events, model, mouth):
    while True:
        event = events.get()
        if event is None:
            return
        guest_label, seat_number = event
        fallback = f"{guest_label} is seated in chair {seat_number}."
        phrase = fallback
        if ollama is not None:
            try:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Give one short, polite spoken confirmation. Use only the "
                                "verified guest label and chair number supplied by the "
                                "program. Do not infer anything else."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Verified event: guest_label={guest_label!r}; "
                                f"chair_number={seat_number}."
                            ),
                        },
                    ],
                    options={"temperature": 0.1, "num_predict": 48},
                    keep_alive="15m",
                )
                candidate = response.get("message", {}).get("content", "").strip()
                if candidate:
                    phrase = candidate
            except Exception as exc:
                print(f"[LLM] Could not generate announcement; using fixed phrase: {exc}")
        print(f"[ANNOUNCEMENT] {phrase}")
        if mouth is not None:
            try:
                mouth.speak(phrase)
            except Exception as exc:
                print(f"[SPEECH] Could not speak announcement: {exc}")
        events.task_done()


def main():
    args = _parse_args()
    if not 0.3 <= args.threshold < 1.0 or not 0.0 <= args.margin < 0.5:
        raise SystemExit("Choose --threshold in [0.3, 1.0) and --margin in [0, 0.5).")
    for path in (PERSON_MODEL, POSE_MODEL, OSNET_MODEL):
        if not path.is_file():
            raise SystemExit(f"Missing model file: {path}")

    manager = GuestIdentityManager(
        gallery_size=ENROLL_SAMPLE_COUNT,
        recover_threshold=args.threshold,
        track_threshold=args.threshold,
        margin=args.margin,
        min_hits=3,
        max_missing_s=3600.0,
    )
    model_hash = _sha256(OSNET_MODEL)
    try:
        label_by_id = _load_gallery(args.gallery.expanduser().resolve(), manager, model_hash)
    except Exception as exc:
        raise SystemExit(f"Could not load guest gallery: {exc}") from exc
    label_by_id.setdefault("guest_1", args.guest1_label)
    label_by_id.setdefault("guest_2", args.guest2_label)
    seat_tracker = SeatOccupancy()

    print(f"Loading YOLO person/chair model: {PERSON_MODEL}")
    detector = YOLO(str(PERSON_MODEL))
    print(f"Loading YOLO pose model: {POSE_MODEL}")
    pose_model = YOLO(str(POSE_MODEL))
    print(f"Loading OSNet appearance encoder: {OSNET_MODEL}")
    encoder = OsnetEncoder(OSNET_MODEL, threads=int(os.getenv("RECEPTIONIST_REID_THREADS", "2")))

    mouth = None
    if not args.no_speech:
        try:
            mouth = Mouth()
        except Exception as exc:
            print(f"[SPEECH] Piper audio unavailable; announcements will print only: {exc}")
    events = queue.Queue()
    threading.Thread(
        target=_llm_worker, args=(events, args.llm_model, mouth), daemon=True
    ).start()

    rclpy.init()
    camera = CameraSource()
    print(f"Waiting for RealSense color frames on {IMAGE_TOPIC}.")
    print("Camera should stay fixed on both chairs; this test sends no navigation goals.")
    print("Press 1 to enroll Guest 1, 2 to enroll Guest 2, and Q to quit.")
    print("Loaded galleries:", ", ".join(
        f"{guest_id}={label_by_id.get(guest_id, guest_id)}"
        for guest_id in manager.guests
    ) or "none")
    print("Existing lab gallery is reused; new enrollments are saved to that same file.")

    enrollment = None
    enrollment_message = "Ready"
    enrollment_timeout_s = 45.0
    last_frame_id = -1
    last_identity_at = 0.0
    last_identity_by_track = {}
    latest_identity_results = {}
    no_people_since = None
    return_test_armed = False
    candidate_by_seat = {}
    announced = set()

    try:
        while rclpy.ok():
            rclpy.spin_once(camera, timeout_sec=0.005)
            frame, frame_id = camera.latest()
            if frame is None:
                time.sleep(0.01)
                continue
            if frame_id == last_frame_id:
                cv2.waitKey(1)
                continue
            last_frame_id = frame_id
            now = time.monotonic()

            detections = detector.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[0, 56, 57],
                conf=0.45,
                verbose=False,
            )[0]
            people = []
            seat_candidates = []
            if detections.boxes is not None:
                for box in detections.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    track_id = int(box.id[0]) if box.id is not None else None
                    if cls_id == 0:
                        crop = _crop(frame, (x1, y1, x2, y2))
                        people.append({
                            "box": (x1, y1, x2, y2), "track_id": track_id,
                            "confidence": confidence, "crop": crop,
                        })
                    elif cls_id in (56, 57) and confidence >= 0.75:
                        width, height = x2 - x1, y2 - y1
                        aspect = width / float(max(1, height))
                        if width >= 50 and height >= 55 and aspect <= 2.2:
                            seat_candidates.append((x1, y1, x2, y2, track_id, cls_id))

            raw_seats, nested_ids, nested_boxes = _suppress_nested_seat_boxes(seat_candidates)
            poses = _pose_by_person(frame, people, pose_model)
            seat_tracks = seat_tracker.update(
                raw_seats, nested_ids, nested_boxes,
                [( *person["box"], person["track_id"]) for person in people],
                poses, frame.shape, now,
            )

            embeddings_by_track = {}
            valid_people = [
                person for person in people
                if person["confidence"] >= 0.45 and person["track_id"] is not None
                and person["crop"] is not None
            ]
            should_encode = (
                (manager.guests or enrollment is not None)
                and now - last_identity_at >= 0.4
            )
            if should_encode:
                last_identity_at = now
                try:
                    vectors = encoder.encode_batch([person["crop"] for person in valid_people])
                    embeddings_by_track = {
                        person["track_id"]: vector
                        for person, vector in zip(valid_people, vectors)
                    }
                except Exception as exc:
                    print(f"[OSNET] Embedding failed: {exc}")

                observations = [
                    {"track_id": track_id, "embedding": vector}
                    for track_id, vector in embeddings_by_track.items()
                ]
                identities = manager.observe(observations, now) if manager.guests else []
                identity_results = {item["track_id"]: item for item in identities}
                latest_identity_results = identity_results
                for person in valid_people:
                    if person["track_id"] not in identity_results:
                        identity_results[person["track_id"]] = {
                            "track_id": person["track_id"], "guest_id": None,
                            "state": "NEW_PERSON", "confidence": 0.0,
                        }
                last_identity_by_track = dict(identity_results)

                if enrollment is not None:
                    target_people = [
                        person for person in valid_people
                        if identity_results.get(person["track_id"], {}).get("state") == "NEW_PERSON"
                    ]
                    if len(people) != 1 or len(target_people) != 1:
                        enrollment["track_id"] = None
                        enrollment["vectors"].clear()
                        enrollment["last_sample_at"] = 0.0
                        enrollment_message = "Show only the guest being enrolled; samples reset while ambiguous"
                    else:
                        person = target_people[0]
                        track_id = person["track_id"]
                        if enrollment["track_id"] not in (None, track_id):
                            enrollment["track_id"] = track_id
                            enrollment["vectors"].clear()
                            enrollment["last_sample_at"] = 0.0
                        vector = embeddings_by_track.get(track_id)
                        if vector is not None and now - enrollment["last_sample_at"] >= ENROLL_SAMPLE_INTERVAL_S:
                            duplicate = bool(enrollment["vectors"]) and max(
                                float(np.dot(vector, old)) for old in enrollment["vectors"]
                            ) >= 0.995
                            enrollment["track_id"] = track_id
                            enrollment["last_sample_at"] = now
                            if duplicate:
                                enrollment_message = (
                                    f"{enrollment['guest_id']}: same view; turn a little "
                                    f"({len(enrollment['vectors'])}/{ENROLL_SAMPLE_COUNT})"
                                )
                            else:
                                enrollment["vectors"].append(vector.copy())
                                enrollment_message = (
                                    f"Registering {enrollment['guest_id']}: "
                                    f"{len(enrollment['vectors'])}/{ENROLL_SAMPLE_COUNT}; "
                                    "turn slowly"
                                )
                            if len(enrollment["vectors"]) >= ENROLL_SAMPLE_COUNT:
                                guest_id = enrollment["guest_id"]
                                manager.enroll(guest_id, track_id, enrollment["vectors"], now)
                                _save_gallery(
                                    args.gallery.expanduser().resolve(), manager,
                                    label_by_id, model_hash,
                                )
                                last_identity_by_track[track_id] = {
                                    "track_id": track_id, "guest_id": guest_id,
                                    "state": "TRACKING", "confidence": 1.0,
                                }
                                enrollment_message = f"Saved {label_by_id[guest_id]} ({guest_id})"
                                print(f"[REGISTERED] {guest_id} saved in {args.gallery}")
                                enrollment = None

            # Use the most recent confirmed identity for tracks between OSNet runs.
            current_track_ids = {person["track_id"] for person in people if person["track_id"] is not None}
            identity_by_track = {
                track_id: identity for track_id, identity in last_identity_by_track.items()
                if track_id in current_track_ids
            }
            for track_id in current_track_ids:
                identity_by_track.setdefault(track_id, {
                    "guest_id": None, "state": "UNREGISTERED", "confidence": 0.0,
                })
            last_identity_by_track = dict(identity_by_track)

            if people:
                no_people_since = None
                return_test_armed = False
            elif manager.guests and enrollment is None:
                if no_people_since is None:
                    no_people_since = now
                elif not return_test_armed and now - no_people_since >= NO_PERSON_RESET_S:
                    _arm_return_test(manager, last_identity_by_track, latest_identity_results, now)
                    enrollment_message = "No people detected; return test armed"
                    return_test_armed = True
                    print("[RETURN TEST] No people detected; saved guest galleries retained.")
            else:
                no_people_since = None

            sitting_by_seat = {}
            for index, person in enumerate(people):
                posture, _pose_confidence = poses.get(index, ("unknown", 0.0))
                identity = identity_by_track.get(person["track_id"], {})
                guest_id = identity.get("guest_id")
                if posture != "sitting" or guest_id is None or identity.get("state") not in (
                    "TRACKING", "REID_RECOVERED"
                ):
                    continue
                seat_id = _seat_for_person(person["box"], seat_tracks)
                if seat_id is not None:
                    sitting_by_seat.setdefault(seat_id, []).append(guest_id)

            for seat_id, seat in seat_tracks.items():
                occupants = set(sitting_by_seat.get(seat_id, []))
                if seat.get("status") == "occupied seat" and len(occupants) == 1:
                    guest_id = next(iter(occupants))
                    previous = candidate_by_seat.get(seat_id)
                    if previous is None or previous["guest_id"] != guest_id:
                        candidate_by_seat[seat_id] = {"guest_id": guest_id, "since": now}
                    elif now - previous["since"] >= SEAT_ID_CONFIRM_S:
                        event = (guest_id, seat_id)
                        if event not in announced:
                            label = label_by_id.get(guest_id, guest_id)
                            seat_number = seat_id.removeprefix("seat_")
                            events.put((label, seat_number))
                            announced.add(event)
                            print(f"[SEATED] {label} confirmed in {seat_id}")
                else:
                    candidate_by_seat.pop(seat_id, None)

            display = frame.copy()
            for seat_id, seat in seat_tracks.items():
                x1, y1, x2, y2 = seat["bbox"]
                color = (0, 0, 255) if seat["status"] == "occupied seat" else (0, 255, 0)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                occupant = next(iter(set(sitting_by_seat.get(seat_id, []))), None)
                label = label_by_id.get(occupant, occupant) if occupant else seat["status"]
                cv2.putText(display, f"{seat_id}: {label}", (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            for person in people:
                x1, y1, x2, y2 = person["box"]
                identity = identity_by_track.get(person["track_id"], {})
                guest_id = identity.get("guest_id")
                state = identity.get("state", "UNREGISTERED")
                label = label_by_id.get(guest_id, guest_id) if guest_id else state
                color = (0, 200, 0) if guest_id else (0, 0, 220)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{label} | {person['confidence']:.2f}",
                            (x1, min(display.shape[0] - 5, y2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.rectangle(display, (0, 0), (display.shape[1], 38), (24, 24, 24), -1)
            cv2.putText(display, enrollment_message[:120], (8, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("Mock Guest Seating Identity Test", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("1"), ord("2")):
                guest_id = "guest_1" if key == ord("1") else "guest_2"
                label = label_by_id.get(guest_id, args.guest1_label if guest_id == "guest_1" else args.guest2_label)
                if guest_id in manager.guests:
                    enrollment_message = f"{label} ({guest_id}) already loaded"
                elif enrollment is not None:
                    enrollment_message = "Finish or cancel the current enrollment first"
                else:
                    label_by_id[guest_id] = label
                    enrollment = {
                        "guest_id": guest_id, "track_id": None,
                        "vectors": [], "last_sample_at": 0.0,
                        "started_at": now,
                    }
                    enrollment_message = f"Show {label} alone and turn slowly"
                    print(f"[REGISTER] Enrolling {label} as {guest_id}.")
    finally:
        events.put(None)
        cv2.destroyAllWindows()
        camera.destroy_node()
        rclpy.shutdown()
        if mouth is not None:
            mouth.close()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Isolated guest registration and leave/return ReID lab.

This tool uses the project's COCO person detector, BoT-SORT temporary tracks,
and OSNet appearance embeddings. It does not start or alter the receptionist
task, chair detector, chair status logic, or VLM.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

from robo_cup_robot.vision.guest_identity import GuestIdentityManager
from robo_cup_robot.vision.image_utils import image_msg_to_numpy
from robo_cup_robot.vision.osnet_encoder import OsnetEncoder


ROOT = Path(__file__).resolve().parents[2]
PERSON_MODEL = ROOT / "yolov8n.pt"
OSNET_MODEL = Path(__file__).resolve().parent / "models" / "osnet_x0_25_msmt17.onnx"
IMAGE_TOPIC = "/camera/camera/color/image_raw"
NO_PERSON_RESET_S = 2.0


class CameraSource(Node):
    def __init__(self):
        super().__init__("guest_reid_lab")
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
            return (None, self.frame_id) if self.frame is None else (self.frame.copy(), self.frame_id)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gallery_path(path):
    return Path(path).expanduser().resolve()


def _save_gallery(path, manager, label_by_id, model_hash):
    payload = {
        "format": 1,
        "encoder": "osnet_x0_25_msmt17.onnx",
        "encoder_sha256": model_hash,
        "guests": [],
    }
    for guest_id, record in manager.guests.items():
        payload["guests"].append({
            "guest_id": guest_id,
            "label": label_by_id.get(guest_id, guest_id),
            "embeddings": [vector.astype(float).tolist() for vector in record["gallery"]],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _arm_return_test(manager, cached_identity, last_log, now):
    """Clear transient tracking state while retaining enrolled guest galleries."""
    manager.track_to_guest.clear()
    manager.pending.clear()
    manager.quarantined_tracks.clear()
    for record in manager.guests.values():
        record["state"] = "LOST"
        record["track_id"] = None
        record["last_seen"] = now - manager.max_missing_s - 1.0
    cached_identity.clear()
    last_log.clear()


def _load_gallery(path, manager, model_hash):
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != 1 or payload.get("encoder_sha256") != model_hash:
        raise RuntimeError(
            f"Saved gallery {path} was made with a different OSNet model; "
            "refusing to compare incompatible embeddings."
        )
    labels = {}
    for guest in payload.get("guests", []):
        guest_id = str(guest["guest_id"])
        vectors = [np.asarray(item, dtype=np.float32).reshape(-1) for item in guest["embeddings"]]
        if not 2 <= len(vectors) <= manager.gallery_size:
            raise RuntimeError(f"Saved gallery for {guest_id} has an invalid sample count")
        if any(item.size != 512 or not np.isfinite(item).all() for item in vectors):
            raise RuntimeError(f"Saved gallery for {guest_id} has invalid OSNet vectors")
        vectors = [item / max(float(np.linalg.norm(item)), 1e-8) for item in vectors]
        manager.embedding_size = 512
        # Loaded galleries deliberately have no live detector-track binding.
        manager.guests[guest_id] = {
            "gallery": tuple(item.copy() for item in vectors),
            "track_id": None,
            "state": "LOST",
            "last_seen": time.monotonic() - manager.max_missing_s - 1.0,
            "confidence": 0.0,
        }
        labels[guest_id] = str(guest.get("label", guest_id))
    return labels


def _person_crop(frame, box):
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 - x1 < 35 or y2 - y1 < 65:
        return None
    return frame[y1:y2, x1:x2].copy()


def _good_registration_view(person, frame_shape):
    x1, y1, x2, y2 = person["box"]
    height, width = frame_shape[:2]
    return (
        person["confidence"] >= 0.45
        and person["crop"] is not None
        and person["track_id"] is not None
        and y2 - y1 >= 100
        and x1 > 2 and y1 > 2 and x2 < width - 2 and y2 < height - 2
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guest-id", default="guest_1", help="Stable logical ID to register")
    parser.add_argument("--label", help="Display name; prompted if omitted")
    parser.add_argument(
        "--gallery", default=str(ROOT / "guest_reid_lab_gallery.json"),
        help="Local persistent embedding gallery (no camera images are saved)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace this guest's saved gallery")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="Minimum cosine similarity for OSNet recovery (default: 0.60)")
    parser.add_argument("--margin", type=float, default=0.06,
                        help="Required lead over the next guest (default: 0.06)")
    return parser.parse_args()


def main():
    args = _parse_args()
    label = args.label or input("Guest display label (for example, Alice): ").strip()
    if not label:
        raise SystemExit("A guest display label is required.")
    if not 0.3 <= args.threshold < 1.0 or not 0.0 <= args.margin < 0.5:
        raise SystemExit("Choose a threshold in [0.3, 1.0) and margin in [0, 0.5).")
    if not PERSON_MODEL.is_file() or not OSNET_MODEL.is_file():
        raise SystemExit(f"Missing model file: {PERSON_MODEL if not PERSON_MODEL.is_file() else OSNET_MODEL}")

    gallery_file = _gallery_path(args.gallery)
    model_hash = _sha256(OSNET_MODEL)
    manager = GuestIdentityManager(
        gallery_size=10,
        recover_threshold=args.threshold,
        track_threshold=args.threshold,
        margin=args.margin,
        min_hits=3,
        max_missing_s=3600.0,
    )
    label_by_id = _load_gallery(gallery_file, manager, model_hash)
    if args.overwrite and args.guest_id in manager.guests:
        del manager.guests[args.guest_id]
        label_by_id.pop(args.guest_id, None)

    print(f"Loading person detector: {PERSON_MODEL}")
    detector = YOLO(str(PERSON_MODEL))
    print(f"Loading OSNet appearance encoder: {OSNET_MODEL}")
    encoder = OsnetEncoder(OSNET_MODEL, threads=int(os.getenv("RECEPTIONIST_REID_THREADS", "2")))

    rclpy.init()
    camera = CameraSource()
    print(f"Waiting for color frames on {IMAGE_TOPIC} …")
    print("Keys: E enroll this guest; L manually arm a leave/return challenge; Q quit.")
    print("Enrollment requires only one clearly visible person. Stand full-body in view and turn slowly.")
    print(f"After registration, the return test arms automatically after {NO_PERSON_RESET_S:.0f} seconds with no people detected.")
    print("The lab will only assign the saved ID after 3 consistent OSNet matches; ambiguous stays unknown.")

    last_frame_id = -1
    last_reid_at = 0.0
    last_log = {}
    cached_identity = {}
    enrollment = None
    enrollment_status = "Loaded" if manager.guests else "Not registered"
    enrollment_timeout_s = 40.0
    no_people_since = None
    return_test_armed = False
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
            result = detector.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[0],
                conf=0.45,
                verbose=False,
            )[0]

            people = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else None
                    crop = _person_crop(frame, (x1, y1, x2, y2))
                    people.append({
                        "box": (x1, y1, x2, y2), "confidence": confidence,
                        "track_id": track_id, "crop": crop,
                    })

            if people:
                no_people_since = None
                return_test_armed = False
            elif manager.guests and enrollment is None:
                if no_people_since is None:
                    no_people_since = now
                elif (not return_test_armed
                      and now - no_people_since >= NO_PERSON_RESET_S):
                    _arm_return_test(manager, cached_identity, last_log, now)
                    enrollment_status = "No people detected; return test armed"
                    return_test_armed = True
                    print("[RETURN TEST] No people detected for "
                          f"{NO_PERSON_RESET_S:.1f}s; cleared live IDs, saved gallery retained.")
            else:
                no_people_since = None

            embeddings_by_track = {}
            quality_people = [
                person for person in people
                if person["confidence"] >= 0.45 and person["track_id"] is not None
                and person["crop"] is not None
            ]
            should_encode = (
                (manager.guests or enrollment is not None)
                and now - last_reid_at >= 0.4
            )
            if should_encode:
                last_reid_at = now
                try:
                    vectors = encoder.encode_batch([person["crop"] for person in quality_people])
                    embeddings_by_track = {
                        person["track_id"]: vector
                        for person, vector in zip(quality_people, vectors)
                    }
                except Exception as exc:
                    print(f"[OSNET] embedding failed: {exc}")

                observations = [
                    {"track_id": track_id, "embedding": vector}
                    for track_id, vector in embeddings_by_track.items()
                ]
                if manager.guests:
                    verified_bound_track = False
                    for person in quality_people:
                        bound_guest = manager.track_to_guest.get(person["track_id"])
                        if bound_guest is None:
                            continue
                        try:
                            score = manager._score(
                                manager.guests[bound_guest]["gallery"],
                                embeddings_by_track[person["track_id"]],
                            )
                        except (KeyError, ValueError):
                            continue
                        if score >= manager.track_threshold:
                            verified_bound_track = True
                            break

                    # If several people are visible and none is already confirmed
                    # as this guest, do not let the single-guest gallery claim one
                    # of them by chance. Ask for a one-person return view instead.
                    if len(people) > 1 and not verified_bound_track:
                        manager.pending.clear()
                        for person in people:
                            track_id = person["track_id"]
                            if track_id is not None:
                                cached_identity[track_id] = {
                                    "track_id": track_id, "guest_id": None,
                                    "state": "IDENTITY_AMBIGUOUS", "confidence": 0.0,
                                }
                        if enrollment_status != "Multiple people visible; show one person to verify identity":
                            print("[ID] Multiple people visible with no confirmed guest track; refusing to guess.")
                        enrollment_status = "Multiple people visible; show one person to verify identity"
                    else:
                        for identity in manager.observe(observations, now):
                            cached_identity[identity["track_id"]] = identity
                            signature = (
                                identity["guest_id"], identity["state"],
                                identity.get("candidate"), identity.get("hits"),
                            )
                            if last_log.get(identity["track_id"]) != signature:
                                print(
                                    f"[ID] tracker={identity['track_id']} "
                                    f"guest={identity['guest_id'] or '-'} "
                                    f"state={identity['state']} "
                                    f"score={identity['confidence']:.3f} "
                                    f"candidate={identity.get('candidate', '-')} "
                                    f"hits={identity.get('hits', '-') }"
                                )
                                last_log[identity["track_id"]] = signature
                    current_tracks = {person["track_id"] for person in people
                                      if person["track_id"] is not None}
                    quality_tracks = set(embeddings_by_track)
                    for track_id in current_tracks - quality_tracks:
                        cached_identity.pop(track_id, None)
                    for track_id in list(cached_identity):
                        if track_id not in current_tracks:
                            cached_identity.pop(track_id, None)

            if enrollment is not None:
                if now - enrollment["started_at"] > enrollment_timeout_s:
                    enrollment = None
                    enrollment_status = "Registration timed out; press E to retry"
                elif len(people) != 1:
                    enrollment_status = "Show exactly one person; keeping collected samples"
                else:
                    person = people[0]
                    track_id = person["track_id"]
                    vector = embeddings_by_track.get(track_id)
                    if not _good_registration_view(person, frame.shape):
                        enrollment_status = "Need a clear, uncropped person box at least 100 px tall; keep one person in view"
                    elif enrollment["track_id"] not in (None, track_id):
                        enrollment["track_id"] = track_id
                        enrollment["vectors"].clear()
                        enrollment["last_sample_at"] = 0.0
                        enrollment_status = "Tracker changed; restarting samples for one person"
                    elif vector is not None and now - enrollment["last_sample_at"] >= 0.55:
                        enrollment["track_id"] = track_id
                        duplicate_view = bool(enrollment["vectors"]) and max(
                            float(np.dot(vector, old)) for old in enrollment["vectors"]
                        ) >= 0.995
                        enrollment["last_sample_at"] = now
                        if duplicate_view:
                            enrollment_status = (
                                f"View too similar: {len(enrollment['vectors'])}/10; "
                                "turn your body a little before the next sample"
                            )
                        else:
                            enrollment["vectors"].append(vector.copy())
                            enrollment_status = (
                                f"Registering {label}: {len(enrollment['vectors'])}/10; "
                                "turn slowly so samples cover different views"
                            )
                        if len(enrollment["vectors"]) >= 10:
                            try:
                                manager.enroll(args.guest_id, track_id,
                                               enrollment["vectors"][:10], now)
                            except Exception as exc:
                                enrollment_status = f"Registration failed: {exc}"
                            else:
                                label_by_id[args.guest_id] = label
                                _save_gallery(gallery_file, manager, label_by_id, model_hash)
                                enrollment_status = f"Saved {label} ({args.guest_id}); leave view to test return"
                                print(f"[REGISTERED] {label} as {args.guest_id}; saved gallery to {gallery_file}")
                            enrollment = None

            display = frame.copy()
            visible_tracks = set()
            for person in people:
                x1, y1, x2, y2 = person["box"]
                track_id = person["track_id"]
                if track_id is not None:
                    visible_tracks.add(track_id)
                identity = cached_identity.get(track_id, {}) if track_id is not None else {}
                guest_id = identity.get("guest_id")
                candidate = identity.get("candidate")
                state = identity.get("state", "UNREGISTERED" if not manager.guests else "NEW_PERSON")
                shown_id = label_by_id.get(guest_id, guest_id) if guest_id else (
                    f"possible {label_by_id.get(candidate, candidate)}" if candidate else state
                )
                color = (0, 200, 0) if guest_id else (0, 180, 255) if candidate else (0, 0, 220)
                text = f"{shown_id} | track {track_id} | {person['confidence']:.2f}"
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, text, (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            banner = enrollment_status
            cv2.rectangle(display, (0, 0), (display.shape[1], 34), (24, 24, 24), -1)
            cv2.putText(display, banner[:120], (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)
            cv2.imshow("Guest ReID Registration Lab", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("e"):
                if args.guest_id in manager.guests and not args.overwrite:
                    enrollment_status = f"{args.guest_id} already registered; use --overwrite to replace"
                else:
                    enrollment = {
                        "started_at": now, "track_id": None, "vectors": [],
                        "last_sample_at": 0.0,
                    }
                    enrollment_status = f"Starting registration for {label}; show one person clearly"
                    print(f"[REGISTER] Waiting to capture 10 samples for {label}.")
            if key == ord("l"):
                if people:
                    enrollment_status = "Clear live IDs only after everyone leaves the camera view"
                    print("[RETURN TEST] Refused to clear IDs while people remain visible.")
                elif not manager.guests:
                    enrollment_status = "Register a guest before the return test"
                else:
                    _arm_return_test(manager, cached_identity, last_log, now)
                    enrollment_status = "Return test armed; registered guest can re-enter"
                    return_test_armed = True
                    print("[RETURN TEST] Cleared transient track bindings; saved gallery retained.")

    finally:
        cv2.destroyAllWindows()
        camera.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

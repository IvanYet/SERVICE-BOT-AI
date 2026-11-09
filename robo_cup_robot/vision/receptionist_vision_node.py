#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
import json
import threading
import time
import os
from pathlib import Path
import queue
import ollama
from ultralytics import YOLO
from robo_cup_robot.vision.image_utils import image_msg_to_numpy
from robo_cup_robot.vision.scene_features import DEFAULT_VLM_MODEL, describe_person_features
from robo_cup_robot.vision.guest_identity import GuestIdentityManager
from robo_cup_robot.vision.osnet_encoder import OsnetEncoder
from robo_cup_robot.vision.seat_occupancy import SeatOccupancy
from robo_cup_robot.vision.tuning_server import VisionTuningServer


def _keypoint_angle(a, b, c):
    """Return the angle ABC in degrees, or None for degenerate points."""
    ba = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    bc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return None
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def classify_sitting_pose(keypoints, confidences=None, confidence_threshold=0.35):
    """Classify sitting from COCO shoulder, hip, knee, and ankle keypoints.

    Sitting needs a bent knee and bent hip on one visible side. Standing needs
    near-straight knee and hip evidence on at least one visible side. Partial or
    ambiguous poses return ``unknown`` instead of guessing.
    """
    points = np.asarray(keypoints, dtype=np.float32)
    if points.shape[0] < 17 or points.shape[1] < 2:
        return "unknown", 0.0
    if confidences is None:
        scores = np.ones(17, dtype=np.float32)
    else:
        scores = np.asarray(confidences, dtype=np.float32).reshape(-1)
        if scores.size < 17:
            return "unknown", 0.0

    side_angles = []
    for shoulder, hip, knee, ankle in ((5, 11, 13, 15), (6, 12, 14, 16)):
        indices = (shoulder, hip, knee, ankle)
        if not all(scores[i] >= confidence_threshold for i in indices):
            continue
        knee_angle = _keypoint_angle(points[hip], points[knee], points[ankle])
        hip_angle = _keypoint_angle(points[shoulder], points[hip], points[knee])
        if knee_angle is not None and hip_angle is not None:
            side_angles.append((knee_angle, hip_angle, min(float(scores[i]) for i in indices)))

    if not side_angles:
        return "unknown", 0.0
    sitting_sides = [s for s in side_angles if s[0] < 145.0 and s[1] < 145.0]
    if sitting_sides:
        return "sitting", max(s[2] for s in sitting_sides)
    standing_sides = [s for s in side_angles if s[0] > 155.0 and s[1] > 150.0]
    if standing_sides:
        return "standing", max(s[2] for s in standing_sides)
    return "unknown", max(s[2] for s in side_angles)


def _box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _suppress_nested_seat_boxes(detections, containment_threshold=0.90):
    """Treat a same-class seat box nested inside another as one seat."""
    ordered = sorted(
        detections,
        key=lambda detection: max(0, detection[2] - detection[0])
        * max(0, detection[3] - detection[1]),
        reverse=True,
    )
    kept = []
    suppressed_track_ids = set()
    nested_container_boxes = set()
    for detection in ordered:
        x1, y1, x2, y2, _track_id, class_id = detection
        area = max(0, x2 - x1) * max(0, y2 - y1)
        nested = False
        for outer in kept:
            ox1, oy1, ox2, oy2, _outer_track_id, outer_class_id = outer
            if class_id != outer_class_id:
                continue
            inner_area = min(area, max(0, ox2 - ox1) * max(0, oy2 - oy1))
            if inner_area == 0:
                continue
            intersection = max(0, min(x2, ox2) - max(x1, ox1)) * max(
                0, min(y2, oy2) - max(y1, oy1)
            )
            if intersection / float(inner_area) >= containment_threshold:
                nested = True
                nested_container_boxes.add(tuple(outer[:4]))
                break
        if not nested:
            kept.append(detection)
        elif detection[4] is not None:
            suppressed_track_ids.add(detection[4])
    return kept, suppressed_track_ids, nested_container_boxes


class ReceptionistVisionNode(Node):
    def __init__(self):
        super().__init__('receptionist_vision_node')

        # Camera Subscribers
        self.sub_color = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.sub_depth = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        # Publisher for tracked person and guest identity data.
        self.receptionist_pub = self.create_publisher(String, '/robot/vision/receptionist_data', 10)
        self.description_pub = self.create_publisher(String, '/robot/vision/description', 10)
        self.status_pub = self.create_publisher(String, '/robot/vision/status', 10)
        self.description_sub = self.create_subscription(
            String, '/robot/vision/describe_request', self.request_scene_description, 10)
        self.identity_command_sub = self.create_subscription(
            String, '/robot/vision/identity_command', self.identity_command, 10)
        self.identity_status_pub = self.create_publisher(
            String, '/robot/vision/identity_status', 10)

        # Frame Cache & Thread Locks
        self.latest_frame = None
        self.latest_frame_id = 0
        self.latest_frame_received_at = 0.0
        self.latest_depth = None
        self.lock = threading.Lock()

        # Visual Smoothing Box Cache
        self.smoothed_boxes = {}
        self.alpha = 0.35

        # Track detections across distinct camera frames.
        self.topic_filter_registry = {}
        self.STABILITY_THRESHOLD = 5
        self.seat_occupancy = SeatOccupancy()

        self.identity_manager = GuestIdentityManager(
            gallery_size=10,
            recover_threshold=float(os.getenv("RECEPTIONIST_REID_THRESHOLD", "0.60")),
            track_threshold=float(os.getenv("RECEPTIONIST_TRACK_THRESHOLD", "0.60")),
            margin=float(os.getenv("RECEPTIONIST_REID_MARGIN", "0.06")),
            min_hits=int(os.getenv("RECEPTIONIST_REID_HITS", "3")),
        )
        self.tuning_lock = threading.Lock()
        self.identity_lock = threading.Lock()
        self.vision_tuning = {}
        self.latest_tuning_status = {"seats": [], "people": []}
        try:
            tuning_path = Path(__file__).resolve().parents[2] / "receptionist_vision_tuning.json"
            self.tuning_server = VisionTuningServer(
                tuning_path,
                on_update=self.apply_tuning,
                get_status=self.get_tuning_status,
                logger=self.get_logger(),
                port=int(os.getenv("RECEPTIONIST_TUNING_PORT", "8765")),
            )
            self.apply_tuning(self.tuning_server.values)
            tuning_port = int(os.getenv("RECEPTIONIST_TUNING_PORT", "8765"))
            self.get_logger().info(f"Local vision tuning UI: http://127.0.0.1:{tuning_port}")
        except Exception as exc:
            self.tuning_server = None
            self.get_logger().warning(f"Local vision tuning server unavailable: {exc}")
        self.identity_enrollment = None
        self.latest_person_track_ids = set()
        self.last_identity_inference_at = 0.0
        self.identity_encoder = None
        self.identity_ready = False
        self.last_identity_by_track = {}

        # Task 1 needs the VLM for guest appearance capture. Set
        # RECEPTIONIST_VLM=0 to disable it explicitly.
        self.vlm_enabled = os.getenv("RECEPTIONIST_VLM", "1").lower() in ("1", "true", "yes")
        self.vlm_status = "starting" if self.vlm_enabled else "disabled"
        self.vlm_queue = queue.Queue(maxsize=4)
        if self.vlm_enabled:
            threading.Thread(target=self.vlm_worker, daemon=True).start()

        # Throttle Timer (Publish once every 3.0 seconds)
        self.last_pub_time = 0.0
        self.status_timer = self.create_timer(1.0, self.publish_vision_status)

        # Background Inference Worker
        self.worker_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.worker_thread.start()

        self.get_logger().info("Receptionist person and guest identity vision node online.")

    def apply_tuning(self, values):
        with self.tuning_lock:
            self.vision_tuning = dict(values)
        self.seat_occupancy.configure(values)
        with self.identity_lock:
            self.identity_manager.recover_threshold = float(values["reid_recover_threshold"])
            self.identity_manager.track_threshold = float(values["reid_track_threshold"])
            self.identity_manager.margin = float(values["reid_margin"])
            self.identity_manager.min_hits = int(values["reid_min_hits"])

    def get_tuning_status(self):
        return dict(self.latest_tuning_status)

    def identity_command(self, message):
        """Start an explicit appearance enrollment for one named guest."""
        try:
            request = json.loads(message.data)
            if request.get("action") != "enroll" or not request.get("guest_id"):
                raise ValueError("expected action=enroll and a guest_id")
            guest_id = str(request["guest_id"]).strip()
            target_track_id = request.get("tracker_id")
            if target_track_id is not None:
                target_track_id = int(target_track_id)
            if not self.identity_ready:
                raise RuntimeError("OSNet ReID is unavailable; guest identity cannot be enrolled")
            with self.identity_lock:
                if self.identity_enrollment is not None:
                    raise ValueError("another guest enrollment is already in progress")
                if guest_id in self.identity_manager.guests:
                    raise ValueError(f"guest is already enrolled: {guest_id}")
                known_tracks = set(self.identity_manager.track_to_guest)
                self.identity_enrollment = {
                    "guest_id": guest_id,
                    "track_id": None,
                    "target_track_id": target_track_id,
                    "embeddings": [],
                    "last_sample_at": 0.0,
                    "last_seen_at": time.monotonic(),
                    "started_at": time.monotonic(),
                    "excluded_track_ids": known_tracks.intersection(self.latest_person_track_ids),
                }
            self.publish_identity_status(guest_id, "enrolling", "Show one guest clearly to the camera")
        except Exception as exc:
            self.publish_identity_status(str(locals().get("guest_id", "")), "failed", str(exc))

    def publish_identity_status(self, guest_id, status, detail=""):
        message = String()
        message.data = json.dumps({"guest_id": guest_id, "status": status, "detail": detail})
        self.identity_status_pub.publish(message)

    def color_callback(self, msg):
        try:
            cv_image = image_msg_to_numpy(msg, desired_encoding='bgr8')
            with self.lock:
                self.latest_frame = cv_image
                self.latest_frame_id += 1
                self.latest_frame_received_at = time.monotonic()
        except Exception: 
            pass

    def depth_callback(self, msg):
        try:
            depth_image = image_msg_to_numpy(msg, desired_encoding='16UC1')
            with self.lock: 
                self.latest_depth = depth_image
        except Exception: 
            pass

    def publish_description(self, text):
        message = String()
        message.data = text
        self.description_pub.publish(message)

    def publish_vision_status(self):
        with self.lock:
            camera_ready = (
                self.latest_frame is not None
                and time.monotonic() - self.latest_frame_received_at <= 2.0
            )
        message = String()
        message.data = json.dumps({
            "camera_ready": camera_ready,
            "vlm_status": self.vlm_status,
            "identity_model_ready": self.identity_ready,
        })
        self.status_pub.publish(message)

    def request_scene_description(self, _msg):
        """Queue one copy of the latest camera frame for spoken scene description."""
        if not self.vlm_enabled:
            if self.vlm_status == "unavailable":
                text = (
                    "The vision model is unavailable. Check that Ollama is running "
                    "and the configured model is installed."
                )
            else:
                text = "Appearance features are unavailable because the local vision model is disabled."
            self.publish_description(text)
            return

        with self.lock:
            frame_is_fresh = (
                self.latest_frame is not None
                and time.monotonic() - self.latest_frame_received_at <= 2.0
            )
            frame = self.latest_frame.copy() if frame_is_fresh else None
        if frame is None:
            self.publish_description(
                "The RealSense camera is not providing a fresh image. Please check that it is running."
            )
            return

        try:
            self.vlm_queue.put_nowait(("describe", None, frame))
        except queue.Full:
            self.publish_description("The vision system is busy. Please ask me again in a moment.")

    def vlm_worker(self):
        """Describe requested camera frames away from the processing thread."""
        try:
            model_name = os.getenv("RECEPTIONIST_VLM_MODEL", DEFAULT_VLM_MODEL)
            ollama.Client(
                timeout=float(os.getenv("RECEPTIONIST_VLM_TIMEOUT", "120"))
            ).show(model_name)
            self.vlm_status = "ready"
            self.get_logger().info(f"Ollama VLM ready: {model_name}")
        except Exception as exc:
            self.vlm_enabled = False
            self.vlm_status = "unavailable"
            self.get_logger().error(f"Ollama VLM unavailable; using YOLO only: {exc}")
            while True:
                try:
                    job_type, _, _ = self.vlm_queue.get_nowait()
                except queue.Empty:
                    break
                if job_type == "describe":
                    self.publish_description(
                        "The vision model is unavailable. Check that Ollama is running and the configured model is installed."
                    )
                self.vlm_queue.task_done()
            return

        while rclpy.ok():
            try:
                job_type, job_id, crop = self.vlm_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                encoded_ok, encoded_image = cv2.imencode(".jpg", crop)
                if not encoded_ok:
                    raise RuntimeError("Could not encode the camera image for Ollama")
                if job_type == "describe":
                    answer = describe_person_features(encoded_image.tobytes(), model_name)
                    self.publish_description(answer or "I could not identify the scene clearly.")
            except Exception as exc:
                self.get_logger().warning(f"Ollama VLM inference failed: {exc}")
                if job_type == "describe":
                    self.publish_description(
                        "I could not describe the camera view. Please try again."
                    )
            finally:
                self.vlm_queue.task_done()

    def calculate_accurate_distance(self, depth_frame, x1, y1, x2, y2):
        """Calculates ultra-accurate depth distance by sampling only the tight center ROI 

        and applying percentile filtering to eliminate background wall/floor pixels.

        """
        try:
            if depth_frame is None or depth_frame.size == 0:
                return 0.0

            h_img, w_img = depth_frame.shape
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))

            w, h = x2 - x1, y2 - y1
            if w < 10 or h < 10:
                return 0.0

            # 1. Take ONLY the central 30% core of the box (ignores edges/background)
            crop_x1 = x1 + int(w * 0.35)
            crop_x2 = x1 + int(w * 0.65)
            crop_y1 = y1 + int(h * 0.35)
            crop_y2 = y1 + int(h * 0.65)

            depth_roi = depth_frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # 2. Filter out invalid zero pixels (sensor shadow) and unrealistic range (> 5m)
            valid_pixels = depth_roi[(depth_roi > 200) & (depth_roi < 5000)]

            if len(valid_pixels) == 0:
                return 0.0

            # 3. Percentile Filtering: Keep 15th to 65th percentile (removes background wall depth)
            p15 = np.percentile(valid_pixels, 15)
            p65 = np.percentile(valid_pixels, 65)
            filtered_depth = valid_pixels[(valid_pixels >= p15) & (valid_pixels <= p65)]

            if len(filtered_depth) == 0:
                return round(float(np.median(valid_pixels)) / 1000.0, 2)

            # Return distance in meters rounded to 2 decimal places
            return round(float(np.median(filtered_depth)) / 1000.0, 2)
        except Exception:
            return 0.0

    def smooth_bbox(self, box_id, target_coords, label, color):
        if box_id not in self.smoothed_boxes:
            self.smoothed_boxes[box_id] = {
                "coords": list(target_coords),
                "label": label,
                "color": color,
                "seen": 5
            }
        else:
            curr = self.smoothed_boxes[box_id]["coords"]
            smooth_x1 = int(self.alpha * target_coords[0] + (1 - self.alpha) * curr[0])
            smooth_y1 = int(self.alpha * target_coords[1] + (1 - self.alpha) * curr[1])
            smooth_x2 = int(self.alpha * target_coords[2] + (1 - self.alpha) * curr[2])
            smooth_y2 = int(self.alpha * target_coords[3] + (1 - self.alpha) * curr[3])
            
            self.smoothed_boxes[box_id] = {
                "coords": [smooth_x1, smooth_y1, smooth_x2, smooth_y2],
                "label": label,
                "color": color,
                "seen": 5
            }

    def processing_worker(self):
        print("[WORKER] Loading YOLO Engine...")
        detector_model_path = Path(__file__).resolve().parents[2] / "yolov8n.pt"
        if not detector_model_path.is_file():
            raise FileNotFoundError(
                f"Expected COCO person/chair detector at {detector_model_path}"
            )
        model = YOLO(str(detector_model_path))
        self.get_logger().info(
            f"Using COCO person/chair detector: {detector_model_path}"
        )
        pose_model_path = Path(__file__).resolve().parent / "models" / "yolo11n-pose.pt"
        pose_model = None
        try:
            pose_model = YOLO(str(pose_model_path))
        except Exception as exc:
            self.get_logger().warning(
                f"Sitting pose model unavailable; seat overlap fallback will be used: {exc}"
            )
        reid_model_path = Path(__file__).resolve().parent / "models" / "osnet_x0_25_msmt17.onnx"
        try:
            self.identity_encoder = OsnetEncoder(
                reid_model_path,
                threads=int(os.getenv("RECEPTIONIST_REID_THREADS", "2")),
            )
            self.identity_ready = True
            self.get_logger().info("OSNet appearance encoder ready for guest identity")
        except Exception as exc:
            self.get_logger().warning(f"Guest ReID unavailable; identity will remain uncertain: {exc}")
        print("[WORKER] Vision engine loaded!")

        last_processed_frame_id = -1
        while rclpy.ok():
            frame_to_process, depth_to_process = None, None
            with self.lock:
                frame_id = self.latest_frame_id
                captured_at = self.latest_frame_received_at
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
                if self.latest_depth is not None:
                    depth_to_process = self.latest_depth.copy()

            # Only update persistence timers once for each new camera frame.
            if frame_to_process is None or frame_id == last_processed_frame_id:
                time.sleep(0.01)
                continue
            last_processed_frame_id = frame_id

            try:
                with self.tuning_lock:
                    tuning = dict(self.vision_tuning)
                inference_args = {
                    "verbose": False,
                    "conf": 0.45,
                    "classes": [0, 56, 57],
                }
                # The laptop's 4 GB GPU cannot hold both models. Keep YOLO on
                # CPU when Moondream is enabled so its GPU has the available VRAM.
                if self.vlm_enabled:
                    inference_args["device"] = "cpu"
                tracking_args = dict(inference_args)
                tracking_args["persist"] = True
                tracking_args["tracker"] = "botsort.yaml"
                results = model.track(frame_to_process, **tracking_args)[0]
                raw_people = []
                chair_candidates = []
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0]) if box.id is not None else None
                    if cls_id == 0:
                        raw_people.append((x1, y1, x2, y2, track_id))
                    elif cls_id in (56, 57):
                        # Use strong seat detections for both seat and nesting
                        # checks; person/ReID detections keep their 0.45 threshold.
                        if confidence < tuning.get("seat_confidence", 0.75) or x2 <= x1 or y2 <= y1:
                            continue
                        chair_candidates.append((x1, y1, x2, y2, track_id, cls_id))

                # A chair visible inside another chair (for example, nested or
                # stacked chairs) must not be counted as a second available seat.
                (
                    chair_candidates,
                    nested_track_ids,
                    nested_container_boxes,
                ) = _suppress_nested_seat_boxes(chair_candidates)

                # Match the reference workspace's seat geometry filters. Apply
                # after nesting detection so a small inner chair still makes the
                # containing stack unavailable rather than silently disappearing.
                chair_detections = []
                for x1, y1, x2, y2, track_id, cls_id in chair_candidates:
                    box_width = x2 - x1
                    box_height = y2 - y1
                    aspect_ratio = box_width / float(box_height)
                    if (
                        box_width < tuning.get("seat_min_width", 70)
                        or box_height < tuning.get("seat_min_height", 90)
                        or box_width * box_height < tuning.get("seat_min_area", 7000)
                        or aspect_ratio < tuning.get("seat_aspect_min", 0.45)
                        or aspect_ratio > tuning.get("seat_aspect_max", 1.8)
                    ):
                        continue
                    chair_detections.append((x1, y1, x2, y2, track_id, cls_id))

                # Associate pose detections with tracked person boxes by IoU.
                pose_by_person = {}
                if raw_people and pose_model is not None:
                    try:
                        pose_args = {"verbose": False, "conf": 0.35}
                        if self.vlm_enabled:
                            pose_args["device"] = "cpu"
                        pose_results = pose_model(frame_to_process, **pose_args)[0]
                        used_pose_indices = set()
                        pose_boxes = pose_results.boxes.xyxy.cpu().numpy() if pose_results.boxes is not None else []
                        pose_xy = pose_results.keypoints.xy.cpu().numpy() if pose_results.keypoints is not None else []
                        pose_conf = pose_results.keypoints.conf.cpu().numpy() if pose_results.keypoints is not None else []
                        for person_index, person in enumerate(raw_people):
                            person_box = person[:4]
                            best_index, best_iou = None, 0.0
                            for pose_index, pose_box in enumerate(pose_boxes):
                                if pose_index in used_pose_indices:
                                    continue
                                overlap = _box_iou(person_box, pose_box)
                                if overlap > best_iou:
                                    best_index, best_iou = pose_index, overlap
                            if best_index is not None and best_iou >= 0.25:
                                used_pose_indices.add(best_index)
                                confidences = pose_conf[best_index] if len(pose_conf) else None
                                pose_by_person[person_index] = classify_sitting_pose(
                                    pose_xy[best_index], confidences
                                )
                    except Exception as exc:
                        self.get_logger().warning(f"Sitting pose inference failed: {exc}")
                for person_index in range(len(raw_people)):
                    pose_by_person.setdefault(person_index, ("unknown", 0.0))

                with self.identity_lock:
                    self.latest_person_track_ids = {
                        person[4] for person in raw_people if person[4] is not None
                    }

                now = time.monotonic()
                identity_by_track = {}
                enrollment_failed = None
                should_encode = (
                    self.identity_ready
                    and (self.identity_manager.guests or self.identity_enrollment is not None)
                    and now - self.last_identity_inference_at >= 0.45
                )
                if should_encode:
                    self.last_identity_inference_at = now
                    crops = [
                        frame_to_process[max(0, py1):max(0, py2), max(0, px1):max(0, px2)].copy()
                        for px1, py1, px2, py2, _ in raw_people
                    ]
                    try:
                        embeddings = self.identity_encoder.encode_batch(crops)
                    except Exception as exc:
                        embeddings = []
                        self.get_logger().warning(f"OSNet inference failed: {exc}")
                    observations = []
                    for person, embedding in zip(raw_people, embeddings):
                        if person[4] is not None:
                            observations.append({"track_id": person[4], "embedding": embedding})

                    with self.identity_lock:
                        identity_observations = self.identity_manager.observe(observations, now)
                    for item in identity_observations:
                        identity_by_track[item["track_id"]] = item
                    self.last_identity_by_track = dict(identity_by_track)

                    with self.identity_lock:
                        enrollment = self.identity_enrollment
                        if enrollment is not None:
                            distance_by_track = {}
                            for person in raw_people:
                                track_id = person[4]
                                if track_id is None:
                                    continue
                                distance = self.calculate_accurate_distance(
                                    depth_to_process, *person[:4]
                                )
                                if distance > 0.0:
                                    distance_by_track[track_id] = distance
                            candidate_track = self.identity_manager.select_enrollment_candidate(
                                [person[4] for person in raw_people],
                                identity_observations,
                                enrollment["excluded_track_ids"],
                                distance_by_track=distance_by_track,
                                preferred_track_id=enrollment.get("track_id"),
                                locked_track_id=enrollment.get("target_track_id"),
                            )
                            if candidate_track is not None:
                                track_id = candidate_track
                                candidate_index = next(
                                    idx for idx, person in enumerate(raw_people)
                                    if person[4] == candidate_track
                                )
                                if candidate_index >= len(embeddings):
                                    enrollment["embeddings"].clear()
                                    enrollment["track_id"] = None
                                    enrollment["last_seen_at"] = now
                                else:
                                    if enrollment["track_id"] != track_id:
                                        enrollment["track_id"] = track_id
                                        enrollment["embeddings"].clear()
                                        self.get_logger().info(
                                            "Enrolling nearest visible new person "
                                            f"track {track_id} "
                                            f"({distance_by_track.get(track_id, 0.0):.2f} m)"
                                        )
                                    enrollment["last_seen_at"] = now
                                    if now - enrollment["last_sample_at"] >= 0.4:
                                        enrollment["embeddings"].append(embeddings[candidate_index])
                                        enrollment["last_sample_at"] = now
                                    if len(enrollment["embeddings"]) >= self.identity_manager.gallery_size:
                                        guest_id = enrollment["guest_id"]
                                        try:
                                            self.identity_manager.enroll(
                                                guest_id, track_id, enrollment["embeddings"], now
                                            )
                                        except Exception as exc:
                                            enrollment_failed = (guest_id, str(exc))
                                        else:
                                            self.publish_identity_status(
                                                guest_id, "enrolled",
                                                f"Captured {len(enrollment['embeddings'])} appearance samples",
                                            )
                                            identity_by_track[track_id] = {
                                                "track_id": track_id,
                                                "guest_id": guest_id,
                                                "state": "TRACKING",
                                                "confidence": 1.0,
                                            }
                                            self.last_identity_by_track = dict(identity_by_track)
                                        self.identity_enrollment = None
                            elif not raw_people:
                                if now - enrollment["last_seen_at"] > 1.25:
                                    enrollment["track_id"] = None
                                    enrollment["embeddings"].clear()
                            else:
                                # A crowd, unresolved old guest, or multiple
                                # unknown people makes the enrollment target unclear.
                                enrollment["track_id"] = None
                                enrollment["embeddings"].clear()
                                enrollment["last_seen_at"] = now

                if not should_encode:
                    identity_by_track = {
                        track_id: self.last_identity_by_track[track_id]
                        for *_, track_id in raw_people
                        if track_id is not None and track_id in self.last_identity_by_track
                    }

                with self.identity_lock:
                    enrollment = self.identity_enrollment
                    if enrollment is not None and now - enrollment["started_at"] > 30.0:
                        guest_id = enrollment["guest_id"]
                        self.identity_enrollment = None
                        enrollment_failed = (guest_id, "Timed out before collecting stable appearance samples")

                if enrollment_failed:
                    self.publish_identity_status(*enrollment_failed[:1], "failed", enrollment_failed[1])

                current_frame_candidates = {}
                active_ids = []
                now = time.monotonic()

                # Stable spatial IDs and debounced seat status survive temporary
                # detector ID changes and brief person/seat occlusion.
                seat_tracks = self.seat_occupancy.update(
                    chair_detections,
                    nested_track_ids,
                    nested_container_boxes,
                    raw_people,
                    pose_by_person,
                    frame_to_process.shape,
                    captured_at or now,
                )
                for seat_id, track in seat_tracks.items():
                    x1, y1, x2, y2 = track["bbox"]
                    dist_m = self.calculate_accurate_distance(
                        depth_to_process, x1, y1, x2, y2
                    )
                    status = track["status"]
                    color = {
                        "occupied seat": (0, 0, 255),
                        "uncertain seat": (0, 191, 255),
                        "empty seat": (0, 255, 0),
                    }[status]
                    place_type = "chair" if track["class_id"] == 56 else "couch"
                    self.smooth_bbox(
                        seat_id, (x1, y1, x2, y2),
                        f"{place_type}: {status} ({dist_m}m)", color
                    )
                    active_ids.append(seat_id)
                    current_frame_candidates[seat_id] = {
                        "category": "seat",
                        "data": {
                            "id": seat_id,
                            "type": "chair" if track["class_id"] == 56 else "couch",
                            "status": status,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "distance_meters": dist_m,
                            "center_pixel_x": int((x1 + x2) / 2),
                            "image_width": int(frame_to_process.shape[1]),
                            "pose_confidence": track["pose_confidence"],
                        },
                    }

                for idx, (px1, py1, px2, py2, track_id) in enumerate(raw_people):
                    identity = identity_by_track.get(track_id, {})
                    guest_id = identity.get("guest_id")
                    state = identity.get("state", "UNENROLLED")
                    p_id = f"person_track_{track_id}" if track_id is not None else f"person_untracked_{idx + 1}"
                    active_ids.append(p_id)
                    dist_m = self.calculate_accurate_distance(
                        depth_to_process, px1, py1, px2, py2
                    )
                    posture, pose_confidence = pose_by_person.get(idx, ("unknown", 0.0))
                    self.smooth_bbox(
                        p_id, (px1, py1, px2, py2),
                        f"{guest_id or state} ({posture}, {dist_m}m)", (255, 255, 0)
                    )
                    current_frame_candidates[p_id] = {
                        "category": "person",
                        "data": {
                            "id": p_id,
                            "tracker_id": track_id,
                            "bbox": [int(px1), int(py1), int(px2), int(py2)],
                            "guest_id": guest_id,
                            "identity_state": state,
                            "identity_confidence": identity.get("confidence", 0.0),
                            "identity_candidate": identity.get("candidate"),
                            "identity_hits": identity.get("hits", 0),
                            "posture": posture,
                            "pose_confidence": pose_confidence,
                            "distance_meters": dist_m,
                            "center_pixel_x": int((px1 + px2) / 2),
                        },
                    }

                # Decay UI bounding boxes
                with self.lock:
                    for k in list(self.smoothed_boxes.keys()):
                        if k not in active_ids:
                            self.smoothed_boxes[k]["seen"] -= 1
                            if self.smoothed_boxes[k]["seen"] <= 0:
                                del self.smoothed_boxes[k]

                # =========================================================
                # TOPIC-STAGE FILTERING (DEBOUNCE FOR LLM STABILITY)
                # =========================================================
                for obj_id, obj_info in current_frame_candidates.items():
                    if obj_id not in self.topic_filter_registry:
                        self.topic_filter_registry[obj_id] = {
                            "hits": 1,
                            "category": obj_info["category"],
                            "data": obj_info["data"]
                        }
                    else:
                        self.topic_filter_registry[obj_id]["hits"] += 1
                        self.topic_filter_registry[obj_id]["data"] = obj_info["data"]

                for reg_id in list(self.topic_filter_registry.keys()):
                    if reg_id not in current_frame_candidates:
                        self.topic_filter_registry[reg_id]["hits"] -= 1
                        if self.topic_filter_registry[reg_id]["hits"] <= 0:
                            del self.topic_filter_registry[reg_id]

                verified_payload = {"seats": [], "people": []}
                for reg_id, reg_info in self.topic_filter_registry.items():
                    if reg_info["hits"] >= self.STABILITY_THRESHOLD:
                        if reg_info["category"] == "seat":
                            verified_payload["seats"].append(reg_info["data"])
                        elif reg_info["category"] == "person":
                            verified_payload["people"].append(reg_info["data"])
                self.latest_tuning_status = verified_payload

                # Publish to ROS 2 topic once every 3.0 seconds
                current_time = time.time()
                if (current_time - self.last_pub_time) >= 3.0:
                    msg = String()
                    msg.data = json.dumps(verified_payload)
                    self.receptionist_pub.publish(msg)
                    self.last_pub_time = current_time

            except Exception:
                pass

            time.sleep(0.01)

def main(args=None):
    rclpy.init(args=args)
    node = ReceptionistVisionNode()
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    last_preview_update = 0.0

    while rclpy.ok():
        display_target = None
        boxes_dict = {}
        
        with node.lock:
            if node.latest_frame is not None: 
                display_target = node.latest_frame.copy()
            boxes_dict = dict(node.smoothed_boxes)

        if display_target is not None:
            for b_id, b_info in boxes_dict.items():
                x1, y1, x2, y2 = b_info["coords"]
                label_text = b_info["label"]
                color = b_info["color"]

                cv2.rectangle(display_target, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_target, label_text, (x1, max(20, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Task 3: Precise Receptionist Stream", display_target)
            if (node.tuning_server is not None
                    and time.monotonic() - last_preview_update >= 0.2):
                node.tuning_server.update_preview(display_target)
                last_preview_update = time.monotonic()
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        else:
            time.sleep(0.01)

    cv2.destroyAllWindows()
    if node.tuning_server is not None:
        node.tuning_server.close()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

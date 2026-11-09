"""Start the RealSense camera and search its color stream with best.pt."""

import signal
import re
import statistics
import subprocess
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

from robo_cup_robot.vision.image_utils import image_msg_to_numpy


WEIGHTS = "/home/eyec/Downloads/best.pt"
IMAGE_TOPIC = "/camera/camera/color/image_raw"
DRINK_CLASSES = {"coffee", "green tea", "mineral water"}


def _normalize_label(label):
    text = str(label).lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def _canonical_drink(value):
    """Map natural spoken variants onto the drink labels supported by best.pt."""
    text = _normalize_label(value)
    aliases = {
        "coffee": ("coffee", "espresso", "latte", "cappuccino"),
        "green tea": ("green tea", "tea"),
        "mineral water": ("mineral water", "bottled water", "water"),
    }
    padded_text = f" {text} "
    for canonical, variants in aliases.items():
        if any(f" {variant} " in padded_text for variant in variants):
            return canonical
    return text


class _DrinkSearch(Node):
    def __init__(self, favorite):
        super().__init__("receptionist_drink_search")
        self.favorite = _canonical_drink(favorite)
        self.model = YOLO(WEIGHTS)
        self.finished = threading.Event()
        self.frame_received = threading.Event()
        self.found = False
        self.center_samples = {}
        self.observed = {}
        self.favorite_position = None
        self.favorite_seen_at = None
        self.image_width = None
        self.subscription = self.create_subscription(
            Image, IMAGE_TOPIC, self._on_image, 10
        )
        self.get_logger().info(
            f"Loaded {WEIGHTS}; searching for {self.favorite!r} on {IMAGE_TOPIC}"
        )

    def _on_image(self, message):
        self.frame_received.set()
        try:
            frame = image_msg_to_numpy(message, "bgr8")
            result = self.model.predict(frame, device="cpu", verbose=False)[0]
            self.image_width = frame.shape[1]
            frame_detections = {}
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = self.model.names[class_id]
                normalized = _normalize_label(label)
                if normalized not in DRINK_CLASSES:
                    continue
                confidence = float(box.conf[0].item())
                x1, _, x2, _ = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2.0
                previous = frame_detections.get(normalized)
                if previous is None or confidence > previous["confidence"]:
                    frame_detections[normalized] = {
                        "confidence": confidence,
                        "center_x": center_x,
                    }
                if normalized == self.favorite:
                    self.found = True
                    if self.favorite_seen_at is None:
                        self.favorite_seen_at = time.monotonic()

            for label, detection in frame_detections.items():
                self.center_samples.setdefault(label, []).append(detection["center_x"])

            if self.favorite_seen_at is not None:
                have_all_drinks = DRINK_CLASSES.issubset(self.center_samples)
                collection_finished = time.monotonic() - self.favorite_seen_at >= 1.5
                if have_all_drinks or collection_finished:
                    self._assign_positions()
                    self.finished.set()
        except Exception as exc:
            self.get_logger().error(f"Could not process camera frame: {exc}")

    def _assign_positions(self):
        """Rank detected drinks across the table instead of dividing the image."""
        centers = {
            label: statistics.median(samples)
            for label, samples in self.center_samples.items()
            if samples
        }
        ordered = sorted(centers, key=centers.get)
        self.observed = {
            label: (
                "left" if index == 0
                else "right" if index == len(ordered) - 1
                else "center"
            )
            for index, label in enumerate(ordered)
        }
        if len(ordered) == 1 and self.image_width:
            label = ordered[0]
            relative_x = centers[label] / self.image_width
            self.observed[label] = (
                "left" if relative_x < (1.0 / 3.0)
                else "right" if relative_x >= (2.0 / 3.0)
                else "center"
            )
        self.favorite_position = self.observed.get(self.favorite)


def find_favorite_drink(favorite, timeout=30.0):
    """Return (found, observed drink labels) after searching live camera frames."""
    camera_process = None
    node = None
    executor = None
    spin_thread = None
    initialized_here = not rclpy.ok()

    try:
        if initialized_here:
            rclpy.init(args=None)

        node = _DrinkSearch(favorite)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        deadline = time.monotonic() + timeout
        # Reuse a camera stream if one is already running; otherwise start it here.
        node.frame_received.wait(timeout=3.0)
        if not node.frame_received.is_set():
            camera_process = subprocess.Popen(
                [
                    "ros2", "launch", "realsense2_camera", "rs_launch.py",
                    "align_depth.enable:=true",
                ]
            )

        while time.monotonic() < deadline and not node.finished.is_set():
            if camera_process is not None and camera_process.poll() is not None:
                node.get_logger().error("RealSense launch exited before a match was found")
                break
            node.finished.wait(timeout=0.2)

        observed = dict(node.observed)
        return node.found, observed, node.favorite_position
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if node is not None:
            node.destroy_node()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        if camera_process is not None and camera_process.poll() is None:
            camera_process.send_signal(signal.SIGINT)
            try:
                camera_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                camera_process.terminate()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()

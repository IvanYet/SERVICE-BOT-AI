#!/usr/bin/env python3
"""Capture a RealSense color frame after a short delay and describe it with Ollama."""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from robo_cup_robot.vision.image_utils import image_msg_to_numpy
from robo_cup_robot.vision.scene_features import (
    DEFAULT_PARAPHRASE_MODEL,
    DEFAULT_VLM_MODEL,
    describe_person_features,
    paraphrase_person_features,
)


class CameraSnapshot(Node):
    def __init__(self):
        super().__init__("camera_scene_description")
        self.frame = None
        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.on_image,
            10,
        )

    def on_image(self, message):
        try:
            self.frame = image_msg_to_numpy(message, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Could not convert camera image: {exc}")


def main():
    delay_seconds = 5.0
    model = os.getenv("RECEPTIONIST_VLM_MODEL", DEFAULT_VLM_MODEL)
    rclpy.init()
    node = CameraSnapshot()
    window_name = "RealSense live camera (press q to close)"

    try:
        deadline = time.monotonic() + delay_seconds
        print(f"Waiting {delay_seconds:g} seconds before capturing the camera view...", flush=True)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.frame is not None:
                cv2.imshow(window_name, node.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Preview closed before capture.")
                    return 0

        # Give the subscription a brief chance to receive its first image if it started late.
        frame_deadline = time.monotonic() + 3.0
        while rclpy.ok() and node.frame is None and time.monotonic() < frame_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

        if node.frame is None:
            print("No camera frame received on /camera/camera/color/image_raw.", file=sys.stderr)
            return 1

        frame = node.frame.copy()
        snapshot_path = Path.home() / "Desktop" / "robocupedu_camera_frame.jpg"
        if not cv2.imwrite(str(snapshot_path), frame):
            print(f"Could not save camera frame to {snapshot_path}.", file=sys.stderr)
            return 1
        print(f"Saved camera frame to {snapshot_path}", flush=True)
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            print("Could not encode the camera frame.", file=sys.stderr)
            return 1

        print(
            f"Captured {frame.shape[1]}x{frame.shape[0]} frame. "
            f"Asking VLM {model}, then LLM {DEFAULT_PARAPHRASE_MODEL}...",
            flush=True,
        )

        def analyze_appearance(image_bytes):
            features = describe_person_features(image_bytes, model)
            paragraph = paraphrase_person_features(features)
            return features, paragraph

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(analyze_appearance, encoded.tobytes())
            while not future.done() and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
                if node.frame is not None:
                    cv2.imshow(window_name, node.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Closing preview; waiting for the description...", flush=True)
                    break
            features, paragraph = future.result()

        print("\nStructured VLM attributes:\n" + features)
        print("\nLLM paragraph:\n" + paragraph)
        return 0
    except Exception as exc:
        print(f"Camera description failed: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

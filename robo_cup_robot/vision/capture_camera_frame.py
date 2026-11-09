#!/usr/bin/env python3
"""Save one current RealSense color frame for quick visual inspection."""

import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from robo_cup_robot.vision.image_utils import image_msg_to_numpy


class FrameCapture(Node):
    def __init__(self):
        super().__init__("receptionist_frame_capture")
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
    output_path = Path.home() / "Desktop" / "robocupedu_camera_frame.jpg"
    rclpy.init()
    node = FrameCapture()
    deadline = time.monotonic() + 10.0
    try:
        while rclpy.ok() and node.frame is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.frame is None:
            print("No image received from /camera/camera/color/image_raw.")
            return 1
        if not cv2.imwrite(str(output_path), node.frame):
            print(f"Could not save camera frame to {output_path}.")
            return 1
        print(f"Saved {node.frame.shape[1]}x{node.frame.shape[0]} camera frame to {output_path}")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

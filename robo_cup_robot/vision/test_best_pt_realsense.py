"""Start RealSense and display its live image with standalone best.pt detections."""

import signal
import subprocess
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

from robo_cup_robot.vision.image_utils import image_msg_to_numpy


WEIGHTS = "/home/eyec/Downloads/best.pt"
IMAGE_TOPIC = "/camera/camera/color/image_raw"


class BestPtCameraViewer(Node):
    def __init__(self):
        super().__init__("best_pt_realsense_viewer")
        self.model = YOLO(WEIGHTS)
        self.subscription = self.create_subscription(
            Image, IMAGE_TOPIC, self.on_image, 10
        )
        self.get_logger().info(f"Loaded {WEIGHTS}; classes: {self.model.names}")
        self.get_logger().info(f"Waiting for images on {IMAGE_TOPIC}")

    def on_image(self, message):
        try:
            frame = image_msg_to_numpy(message, "bgr8")
            result = self.model.predict(frame, device="cpu", verbose=False)[0]
            cv2.imshow("RealSense camera", frame)
            cv2.imshow("best.pt detections", result.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                rclpy.shutdown()
        except Exception as exc:
            self.get_logger().error(f"Could not process camera frame: {exc}")


def main():
    camera_process = subprocess.Popen(
        [
            "ros2",
            "launch",
            "realsense2_camera",
            "rs_launch.py",
            "align_depth.enable:=true",
        ]
    )
    node = None
    try:
        # Give the camera driver a moment to start while model loading begins.
        time.sleep(2.0)
        rclpy.init()
        node = BestPtCameraViewer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()
        if camera_process.poll() is None:
            camera_process.send_signal(signal.SIGINT)
            try:
                camera_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                camera_process.terminate()


if __name__ == "__main__":
    main()

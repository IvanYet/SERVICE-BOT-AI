#!/usr/bin/env python3
"""
ir_capture: records camera images (IR by default) for training a YOLO model.

    ros2 run bag_picking ir_capture --ros-args -p out_dir:=$HOME/bag_ir_dataset -p interval:=1.0

A preview window opens. Images are saved automatically every `interval` seconds
(set interval:=0.0 to save only when you press SPACE in the window). Press Q in the window to quit.

Images are converted exactly like bag_locator does at run time (bag_picking.ir_utils.to_bgr8),
so the model is trained on what it will see.
"""
import os
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from bag_picking.ir_utils import to_bgr8


class IrCapture(Node):
    def __init__(self):
        super().__init__('ir_capture')
        self.declare_parameter('image_topic', '/camera/ir/image_raw')
        self.declare_parameter('out_dir', os.path.expanduser('~/bag_ir_dataset'))
        self.declare_parameter('interval', 1.0)
        self.declare_parameter('prefix', 'ir')
        self.out_dir = self.get_parameter('out_dir').value
        self.interval = float(self.get_parameter('interval').value)
        self.prefix = self.get_parameter('prefix').value
        os.makedirs(self.out_dir, exist_ok=True)
        self.count = len([f for f in os.listdir(self.out_dir) if f.endswith('.jpg')])
        self.last_save = 0.0
        self.bridge = CvBridge()
        self.create_subscription(Image, self.get_parameter('image_topic').value, self.on_image,
                                 qos_profile_sensor_data)
        mode = f'every {self.interval:.1f} s' if self.interval > 0 else 'when you press SPACE'
        self.get_logger().info(f'Saving to {self.out_dir} {mode}. Press Q in the window to quit.')

    def save(self, img):
        name = os.path.join(self.out_dir, f'{self.prefix}_{int(time.time()*1000)}.jpg')
        cv2.imwrite(name, img)
        self.count += 1
        self.get_logger().info(f'saved {name}  ({self.count} images)')

    def on_image(self, msg):
        img = to_bgr8(self.bridge.imgmsg_to_cv2(msg, 'passthrough'))
        now = time.monotonic()
        if self.interval > 0 and now - self.last_save >= self.interval:
            self.last_save = now
            self.save(img)
        view = img.copy()
        cv2.putText(view, f'{self.count} saved', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('ir_capture (SPACE = save, Q = quit)', view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            self.save(img)
        elif key in (ord('q'), ord('Q')):
            raise KeyboardInterrupt


def main():
    rclpy.init()
    node = IrCapture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

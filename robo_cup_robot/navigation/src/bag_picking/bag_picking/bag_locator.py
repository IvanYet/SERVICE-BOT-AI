#!/usr/bin/env python3
"""
bag_locator: finds the paper bag with YOLO v11 and works out where to grab it, in arm coordinates.

  camera image (IR or colour) + depth image + camera_info  ->  YOLO  ->  grasp pixel  ->  3D point
  -> transformed into base_link  ->  published on /bag/grasp_point (geometry_msgs/PointStamped)

Also publishes
  /bag/debug_image   the camera image with the detection and grasp point drawn on it (view it in RViz)
  /bag/marker        a sphere at the grasp point (view it in RViz)

The depth image must line up pixel-for-pixel with the detection image. On the Dabai DW the IR
image and the depth image come from the same sensor, so they line up by construction.
(With a colour camera, use depth-to-colour alignment, e.g. depth_registration:=true.)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

import cv2
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped with tf2)

from bag_picking.ir_utils import to_bgr8


class BagLocator(Node):
    def __init__(self):
        super().__init__('bag_locator')

        p = self.declare_parameter
        p('model_path', '')                    # your trained YOLO v11 weights, e.g. /home/you/models/best.pt
        p('target_class', 'paperbag')          # class name inside the model (printed at startup)
        p('confidence', 0.5)
        p('device', 'cuda:0')                  # 'cpu' if no GPU in WSL
        p('image_topic', '/camera/ir/image_raw')          # Dabai DW: IR. Colour cameras: /camera/color/image_raw
        p('depth_topic', '/camera/depth/image_raw')
        p('camera_info_topic', '/camera/depth/camera_info')  # intrinsics of the depth image the 3D point is read from
        p('target_frame', 'base_link')
        p('depth_scale', 0.001)                # 16-bit depth in millimetres -> metres
        p('min_depth', 0.12)                   # metres; closer than this the camera cannot see
        p('max_depth', 1.50)
        p('grasp_v_fraction', 0.25)            # grasp height: 0 = top of the box, 1 = bottom
        p('patch_size', 15)                    # pixels; median depth over this square
        p('max_rate', 10.0)                    # detections per second at most

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        model_path = gp('model_path')
        if not model_path:
            raise RuntimeError("Set the 'model_path' parameter to your YOLO .pt file")

        from ultralytics import YOLO   # imported here so a missing install gives a clear error
        self.model = YOLO(model_path)
        names = self.model.names        # {id: name}
        self.get_logger().info(f'YOLO classes in {model_path}: {names}')
        wanted = gp('target_class')
        ids = [i for i, n in names.items() if n == wanted]
        if ids:
            self.class_id = ids[0]
        elif len(names) == 1:
            self.class_id = next(iter(names))
            self.get_logger().warn(f"No class called '{wanted}', using the model's only class "
                                   f"'{names[self.class_id]}'. Set target_class to silence this.")
        else:
            raise RuntimeError(f"No class '{wanted}' in the model. Classes are: {names}")

        self.conf = gp('confidence')
        self.device = gp('device')
        self.target_frame = gp('target_frame')
        self.depth_scale = gp('depth_scale')
        self.min_d, self.max_d = gp('min_depth'), gp('max_depth')
        self.v_frac = gp('grasp_v_fraction')
        self.patch = int(gp('patch_size'))
        self.min_period = 1.0 / max(gp('max_rate'), 0.1)
        self.last_run = 0.0

        self.bridge = CvBridge()
        self.K = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, gp('camera_info_topic'), self.on_info, qos_profile_sensor_data)
        color_sub = message_filters.Subscriber(self, Image, gp('image_topic'), qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, gp('depth_topic'), qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.05)
        self.sync.registerCallback(self.on_images)

        self.pub_point = self.create_publisher(PointStamped, '/bag/grasp_point', 10)
        self.pub_marker = self.create_publisher(Marker, '/bag/marker', 10)
        self.pub_debug = self.create_publisher(Image, '/bag/debug_image', 2)
        self.get_logger().info('bag_locator ready, waiting for camera images...')

    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def on_images(self, color_msg: Image, depth_msg: Image):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_run < self.min_period:
            return
        self.last_run = now
        if self.K is None:
            self.get_logger().warn('No camera_info yet', throttle_duration_sec=5.0)
            return

        color = to_bgr8(self.bridge.imgmsg_to_cv2(color_msg, 'passthrough'))
        depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        if depth.shape[:2] != color.shape[:2]:
            self.get_logger().error(
                f'Depth {depth.shape[:2]} and image {color.shape[:2]} sizes differ: set the IR and depth '
                'streams to the same resolution (or align depth to colour)', throttle_duration_sec=5.0)
            return

        res = self.model.predict(color, conf=self.conf, classes=[self.class_id],
                                 device=self.device, verbose=False)[0]
        debug = color.copy()
        if len(res.boxes) == 0:
            self.publish_debug(debug, color_msg, 'no bag')
            return

        best = int(np.argmax(res.boxes.conf.cpu().numpy()))
        x1, y1, x2, y2 = res.boxes.xyxy[best].cpu().numpy().astype(int)
        score = float(res.boxes.conf[best])
        u = int((x1 + x2) / 2)
        v = int(y1 + self.v_frac * (y2 - y1))

        # median of the valid depths in a small square around the grasp pixel
        h = self.patch // 2
        roi = depth[max(v - h, 0):v + h + 1, max(u - h, 0):u + h + 1].astype(np.float64)
        if depth.dtype == np.uint16:
            roi *= self.depth_scale
        valid = roi[(roi > self.min_d) & (roi < self.max_d) & np.isfinite(roi)]
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(debug, (u, v), 6, (0, 0, 255), -1)
        if valid.size < 5:
            self.publish_debug(debug, color_msg, f'bag {score:.2f}: no depth at grasp point')
            return
        z = float(np.median(valid))

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        pt = PointStamped()
        pt.header.stamp = color_msg.header.stamp
        pt.header.frame_id = depth_msg.header.frame_id        # camera_depth_optical_frame
        pt.point.x = (u - cx) * z / fx
        pt.point.y = (v - cy) * z / fy
        pt.point.z = z

        try:
            out = self.tf_buffer.transform(pt, self.target_frame, timeout=Duration(seconds=0.1))
        except Exception:
            # the image stamp can be slightly ahead of TF; fall back to the latest transform
            try:
                pt.header.stamp = Time().to_msg()
                out = self.tf_buffer.transform(pt, self.target_frame, timeout=Duration(seconds=0.1))
            except Exception as e:
                self.get_logger().warn(f'TF {pt.header.frame_id} -> {self.target_frame} failed: {e}',
                                       throttle_duration_sec=5.0)
                return
        out.header.stamp = self.get_clock().now().to_msg()
        self.pub_point.publish(out)
        self.publish_marker(out)
        self.publish_debug(debug, color_msg,
                           f'bag {score:.2f}  {z:.2f} m  base ({out.point.x:.3f}, {out.point.y:.3f}, {out.point.z:.3f})')

    def publish_marker(self, p: PointStamped):
        m = Marker()
        m.header = p.header
        m.ns, m.id, m.type, m.action = 'bag', 0, Marker.SPHERE, Marker.ADD
        m.pose.position = p.point
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.03
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0
        m.lifetime = Duration(seconds=1.0).to_msg()
        self.pub_marker.publish(m)

    def publish_debug(self, img, src_msg, text):
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        msg = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        msg.header = src_msg.header
        self.pub_debug.publish(msg)


def main():
    rclpy.init()
    node = BagLocator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

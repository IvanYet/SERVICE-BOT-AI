#!/usr/bin/env python3
"""
bag_picker: picks up the paper bag that bag_locator found.

Call the service to start one pick:
    ros2 service call /pick_bag std_srvs/srv/Trigger

Sequence
  1. open the gripper
  2. go to the "look" pose (joint angles from the parameters) and wait for the arm to settle
  3. average the next few /bag/grasp_point detections
  4. move above the grasp point (gripper pointing straight down, fingers closing along the
     line from the arm's base to the bag, so they pinch the bag's front wall)
  5. go straight down, close the gripper on the bag's wall
  6. lift straight up

Why top-down: the fingers open at most ~8 cm, so they cannot wrap around a bag's body. Pinching the
front wall of the body from above fits the gripper and is an orientation this 5-joint arm can reach
anywhere in its workspace (straight down, any base angle).
"""
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_srvs.srv import Trigger
from pymoveit2 import MoveIt2

ARM_JOINTS = ['base_joint', 'joint_1', 'joint_2', 'joint_3', 'joint_4']


def quat_from_matrix(R):
    """3x3 rotation matrix -> quaternion [x, y, z, w]."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return [x, y, z, w]


def top_down_quat(px, py):
    """Gripper pointing straight down (its z = world -z), fingers sliding (its y) along the radial
    line from the base to (px, py). Matches the gripper_base / link_5 frame of my_arm.urdf."""
    psi = math.atan2(py, px)
    c, s = math.cos(psi), math.sin(psi)
    z_t = np.array([0.0, 0.0, -1.0])
    y_t = np.array([c, s, 0.0])
    x_t = np.cross(y_t, z_t)             # = (-s, c, 0)
    return quat_from_matrix(np.column_stack([x_t, y_t, z_t]))


class BagPicker(Node):
    def __init__(self):
        super().__init__('bag_picker')
        p = self.declare_parameter
        p('arm_group', 'arm')
        p('gripper_group', 'gripper')
        p('base_link', 'base_link')
        p('tool_link', 'link_5')              # tip link of the arm group (gripper_base sits at the same spot)
        p('tcp_offset', 0.085)                # metres from tool_link origin to the middle of the fingertips (MEASURE)
        p('look_joints', [0.05, 0.35, -1.20, 1.49, -1.52])   # PLACEHOLDER: a pose where the camera sees the table
        p('gripper_open', 0.04)
        p('gripper_closed', 0.002)
        p('approach_height', 0.10)            # metres above the grasp point to start the straight-down move
        p('lift_height', 0.10)
        p('grasp_z_offset', 0.0)              # + raises / - lowers the grasp relative to the detected point
        p('detections_to_average', 5)
        p('detection_timeout', 4.0)
        p('max_velocity', 0.2)                # fraction of the joint_limits.yaml speeds
        p('max_acceleration', 0.2)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        cb = ReentrantCallbackGroup()
        self.arm = MoveIt2(node=self, joint_names=ARM_JOINTS, base_link_name=gp('base_link'),
                           end_effector_name=gp('tool_link'), group_name=gp('arm_group'),
                           callback_group=cb, use_move_group_action=True)
        self.gripper = MoveIt2(node=self, joint_names=['left_finger_joint'], base_link_name=gp('base_link'),
                               end_effector_name=gp('tool_link'), group_name=gp('gripper_group'),
                               callback_group=cb, use_move_group_action=True)
        for m in (self.arm, self.gripper):
            m.max_velocity = gp('max_velocity')
            m.max_acceleration = gp('max_acceleration')

        self.lock = threading.Lock()
        self.detections = []                  # (receive_time, np.array([x, y, z]))
        self.create_subscription(PointStamped, '/bag/grasp_point', self.on_point, 10, callback_group=cb)
        self.create_service(Trigger, 'pick_bag', self.on_pick, callback_group=cb)
        self.busy = False
        self.get_logger().info("bag_picker ready: ros2 service call /pick_bag std_srvs/srv/Trigger")

    def on_point(self, msg: PointStamped):
        if msg.header.frame_id != self.get_parameter('base_link').value:
            self.get_logger().warn(f'grasp point in frame {msg.header.frame_id}, expected base_link',
                                   throttle_duration_sec=5.0)
            return
        with self.lock:
            self.detections.append((time.monotonic(), np.array([msg.point.x, msg.point.y, msg.point.z])))
            self.detections = self.detections[-50:]

    # ---------------- motion helpers ----------------
    def _done(self, m, what):
        ok = m.wait_until_executed()
        if ok is False:
            raise RuntimeError(f'{what} failed (see the move_group terminal)')

    def gripper_to(self, pos, what):
        self.gripper.move_to_configuration([pos])
        self._done(self.gripper, what)

    def arm_to_joints(self, q, what):
        self.arm.move_to_configuration(list(q))
        self._done(self.arm, what)

    def arm_to_pose(self, pos, quat, what, cartesian=False):
        self.arm.move_to_pose(position=list(pos), quat_xyzw=quat, cartesian=cartesian,
                              tolerance_position=0.005, tolerance_orientation=0.05)
        try:
            self._done(self.arm, what)
        except RuntimeError:
            if not cartesian:
                raise
            self.get_logger().warn(f'{what}: straight-line move failed, trying a normal plan')
            self.arm.move_to_pose(position=list(pos), quat_xyzw=quat,
                                  tolerance_position=0.005, tolerance_orientation=0.05)
            self._done(self.arm, what)

    def wait_for_bag(self, since):
        n = int(self.get_parameter('detections_to_average').value)
        deadline = time.monotonic() + float(self.get_parameter('detection_timeout').value)
        while time.monotonic() < deadline:
            with self.lock:
                fresh = [p for t, p in self.detections if t > since]
            if len(fresh) >= n:
                pts = np.array(fresh[-n:])
                spread = float(np.max(np.linalg.norm(pts - np.median(pts, axis=0), axis=1)))
                if spread > 0.03:
                    self.get_logger().warn(f'detections jump by {spread*100:.1f} cm, using the median')
                return np.median(pts, axis=0)
            time.sleep(0.05)
        raise RuntimeError('no bag seen from the look pose (check /bag/debug_image)')

    # ---------------- the pick ----------------
    def on_pick(self, request, response):
        if self.busy:
            response.success, response.message = False, 'already picking'
            return response
        self.busy = True
        try:
            gp = lambda n: self.get_parameter(n).value  # noqa: E731
            self.get_logger().info('pick: opening gripper')
            self.gripper_to(gp('gripper_open'), 'open gripper')

            self.get_logger().info('pick: going to the look pose')
            self.arm_to_joints(gp('look_joints'), 'look pose')
            time.sleep(0.8)                                    # let the arm and the image settle
            target = self.wait_for_bag(since=time.monotonic())
            target[2] += gp('grasp_z_offset')
            self.get_logger().info(f'pick: bag grasp point in base_link: '
                                   f'({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})')

            quat = top_down_quat(target[0], target[1])
            tcp = np.array([0.0, 0.0, gp('tcp_offset')])      # tool pointing down: tool_link sits above the fingertips
            grasp = target + tcp
            pre = grasp + np.array([0.0, 0.0, gp('approach_height')])
            lift = grasp + np.array([0.0, 0.0, gp('lift_height')])

            self.get_logger().info('pick: moving above the bag')
            self.arm_to_pose(pre, quat, 'pre-grasp')
            self.get_logger().info('pick: going down')
            self.arm_to_pose(grasp, quat, 'approach', cartesian=True)
            self.get_logger().info('pick: closing gripper')
            self.gripper_to(gp('gripper_closed'), 'close gripper')
            time.sleep(0.5)
            self.get_logger().info('pick: lifting')
            self.arm_to_pose(lift, quat, 'lift', cartesian=True)

            response.success, response.message = True, 'bag picked'
        except Exception as e:
            self.get_logger().error(f'pick failed: {e}')
            response.success, response.message = False, str(e)
        finally:
            self.busy = False
        return response


def main():
    rclpy.init()
    node = BagPicker()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

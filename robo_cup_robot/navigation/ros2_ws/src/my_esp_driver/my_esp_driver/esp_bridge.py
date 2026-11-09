#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
import math
import time

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import TransformStamped, Quaternion, Twist
from tf2_ros import TransformBroadcaster

def euler_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q

class EspBridge(Node):
    def __init__(self):
        super().__init__('esp_bridge')

        # --- ROS Parameters ---
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').get_parameter_value().string_value
        self.base_frame = self.declare_parameter('base_frame', 'base_footprint').get_parameter_value().string_value

        port = self.get_parameter('port').get_parameter_value().string_value
        baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value

        # Robot Physical Parameters
        self.wheel_radius = 0.065  # 65mm radius in meters
        self.wheel_base = 0.260   # Track width in meters

        # Odometry Integration State
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.last_time = self.get_clock().now()

        # Cumulative Wheel Rotation Positions (radians)
        self.left_wheel_pos = 0.0
        self.right_wheel_pos = 0.0

        # ROS Publishers & TF Broadcasters
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to /cmd_vel (From Nav2 & Teleop)
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10
        )

        # Serial Connection (DTR/RTS flags set for ESP32-S3 USB CDC)
        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
            self.ser.dtr = True
            self.ser.rts = True
            time.sleep(0.1)
            self.ser.reset_input_buffer()
            self.get_logger().info(f"Connected to ESP32-S3 on {port} at {baudrate} baud.")
        except serial.SerialException as e:
            self.get_logger().fatal(f"Failed to open port {port}: {e}")
            raise e

        # 50 Hz Serial polling loop (20ms)
        self.timer = self.create_timer(0.02, self.read_serial)

    def cmd_callback(self, msg: Twist):
        """Sends speed commands to ESP32-S3 over USB"""
        self.get_logger().info(f">> CMD RECEIVED: Vx={msg.linear.x:.2f}, Wz={msg.angular.z:.2f}")

        if self.ser and self.ser.is_open:
            cmd_str = f"$CMD,{msg.linear.x:.3f},{msg.angular.z:.3f}\n"
            try:
                self.ser.write(cmd_str.encode('utf-8'))
                self.ser.flush()
            except Exception as e:
                self.get_logger().warn(f"Failed to write /cmd_vel to serial: {e}")

    def read_serial(self):
        if not self.ser or not self.ser.is_open:
            return

        while self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            except Exception:
                continue

            # Forward debug prints from S3 to ROS terminal
            if not line.startswith('$WHEEL,'):
                if line:
                    self.get_logger().info(f"[S3 DEBUG]: {line}")
                continue

            # Parse formatted telemetry: $WHEEL, v_left, v_right, v_linear, gyro_z
            parts = [p.strip() for p in line.split(',')]
            
            if len(parts) >= 5:
                try:
                    v_left = float(parts[1])
                    v_right = float(parts[2])
                    v_x = float(parts[3])          # Whole robot linear speed in X (m/s)
                    omega_z = float(parts[4])      # MPU6050 Gyro Z (rad/s)

                    # =========================================================
                    # --- BULLETPROOF ZERO-VELOCITY LOCK (ZUPT) ---
                    # =========================================================
                    # If BOTH wheels are stopped, the robot CANNOT rotate!
                    if abs(v_left) < 0.003 and abs(v_right) < 0.003:
                        omega_z = 0.0
                        v_x = 0.0
                    # =========================================================

                    self.get_logger().info(
                        f"L: {v_left:+7.3f} m/s | R: {v_right:+7.3f} m/s | "
                        f"Vx: {v_x:+7.3f} m/s | GyroZ: {omega_z:+7.4f} rad/s",
                        throttle_duration_sec=0.2
                    )

                    self.update_state(v_left, v_right, v_x, omega_z)

                except ValueError:
                    continue

    def update_state(self, v_left: float, v_right: float, v_x: float, omega_z: float):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9

        if dt > 0.5:
            self.last_time = current_time
            return

        self.last_time = current_time

        # 1. Update Continuous Wheel Angles (displacement = (v / r) * dt)
        omega_left = v_left / self.wheel_radius
        omega_right = v_right / self.wheel_radius
        self.left_wheel_pos += omega_left * dt
        self.right_wheel_pos += omega_right * dt

        # 2. Planar Odometry Integration
        delta_th = omega_z * dt
        mid_th = self.th + (delta_th / 2.0)

        self.x += (v_x * math.cos(mid_th)) * dt
        self.y += (v_x * math.sin(mid_th)) * dt
        self.th += delta_th

        # Normalize yaw between -pi and +pi
        self.th = math.atan2(math.sin(self.th), math.cos(self.th))
        odom_quat = euler_to_quaternion(self.th)

        # 3. Publish Dynamic TF (DISABLED FOR EKF!)
        # The EKF node (robot_localization) now broadcasts the odom -> base_footprint transform!
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame

        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation = odom_quat
        # self.tf_broadcaster.sendTransform(t)  # <-- DISABLED TO LET EKF BROADCAST TF!

        # 4. Publish Raw /odom Topic for EKF
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = odom_quat

        odom.pose.covariance[0] = 0.001
        odom.pose.covariance[7] = 0.001
        odom.pose.covariance[35] = 0.001

        odom.twist.twist.linear.x = v_x
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.angular.z = omega_z

        odom.twist.covariance[0] = 0.001
        odom.twist.covariance[35] = 0.001

        self.odom_pub.publish(odom)

        # 5. Publish /joint_states Topic
        js = JointState()
        js.header.stamp = current_time.to_msg()
        js.name = [
            'left_wheel_joint', 
            'right_wheel_joint', 
            'left_caster_joint', 
            'right_caster_joint'
        ]
        js.position = [
            self.left_wheel_pos, 
            self.right_wheel_pos, 
            0.0, 
            0.0
        ]
        js.velocity = [
            omega_left, 
            omega_right, 
            0.0, 
            0.0
        ]
        self.joint_pub.publish(js)

        # 6. Publish /imu/data Topic for EKF
        imu = Imu()
        imu.header.stamp = current_time.to_msg()
        imu.header.frame_id = 'imu_link'
        imu.angular_velocity.z = omega_z
        imu.angular_velocity_covariance[8] = 0.0001
        self.imu_pub.publish(imu)

    def destroy_node(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = EspBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
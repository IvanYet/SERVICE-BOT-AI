#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
import time
from sensor_msgs.msg import JointState

class ArmBridge(Node):
    def __init__(self):
        super().__init__('arm_bridge')

        # Declare port parameter (can be overridden from terminal or launch file)
        self.declare_parameter('port', '/dev/esp_arm') 
        self.declare_parameter('baudrate', 115200)

        port = self.get_parameter('port').get_parameter_value().string_value
        baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value

        self.joint_pub = self.create_publisher(JointState, 'arm_joint_states', 10)

        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
            self.get_logger().info(f"Connected to Robotic Arm on {port} at {baudrate} baud.")
        except serial.SerialException as e:
            self.get_logger().fatal(f"Failed to open arm serial port {port}: {e}")
            raise e

        # 50 Hz loop (20ms) matching firmware output rate
        self.timer = self.create_timer(0.02, self.read_arm_serial)

    def read_arm_serial(self):
        if not self.ser or not self.ser.is_open:
            return

        while self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            except Exception:
                continue

            # Look for lines starting with 'J '
            # Firmware format: J <state> <q0> <q1> <q2> <q3> <q4> <g>
            if not line.startswith('J '):
                continue

            parts = line.split()
            if len(parts) >= 8:
                try:
                    q0 = float(parts[2]) # base_joint
                    q1 = float(parts[3]) # joint_1
                    q2 = float(parts[4]) # joint_2
                    q3 = float(parts[5]) # joint_3
                    q4 = float(parts[6]) # joint_4
                    g  = float(parts[7]) # left_finger_joint (gripper)

                    self.publish_arm_state([q0, q1, q2, q3, q4, g])

                except ValueError:
                    continue

    def publish_arm_state(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        # Exact matching joint names from final_robot.urdf.xacro:
        msg.name = [
            'base_joint',
            'joint_1',
            'joint_2',
            'joint_3',
            'joint_4',
            'left_finger_joint'
        ]
        msg.position = positions

        self.joint_pub.publish(msg)

    def destroy_node(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArmBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
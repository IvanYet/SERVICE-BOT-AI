import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class SafetyBrakeNode(Node):
    def __init__(self):
        super().__init__('safety_brake_node')

        # --- CONFIGURATION PARAMETERS ---
        self.stop_distance = 0.30      # 40cm: Stop if obstacle is closer than this
        self.robot_width = 0.38        # Width of your physical robot (meters)
        self.side_margin = 0.10        # 10cm safety buffer on each side of the robot
        self.sensor_timeout = 0.5      # 0.5 seconds: Stop robot if Lidar stops sending data
        # --------------------------------

        # Path Width calculation (Virtual Tunnel)
        self.path_width = self.robot_width + (2 * self.side_margin)
        
        # Internal State Variables
        self.obstacle_in_path = False
        self.last_scan_time = self.get_clock().now()

        # ROS 2 Subscribers
        # 1. Listen to raw Lidar data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        # 2. Listen to Nav2 velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)

        # ROS 2 Publisher
        # Publishes the SAFE velocity commands to the actual motor controller
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_out', 10)

        # Safety timer to monitor Lidar health
        self.timer = self.create_timer(0.1, self.health_check_callback)
        
        self.get_logger().info(f"Safety Brake Node Active. Tunnel Width: {self.path_width:.2f}m, Stop Distance: {self.stop_distance:.2f}m")

    def scan_callback(self, msg):
        self.last_scan_time = self.get_clock().now()
        danger_detected = False

        for i, distance in enumerate(msg.ranges):
            # Skip invalid/infinite readings
            if distance <= 0.05 or distance == float('inf'):
                continue

            # Convert Polar (angle/distance) to Cartesian (X/Y)
            angle = msg.angle_min + (i * msg.angle_increment)
            x = distance * math.cos(angle) # Forward distance
            y = distance * math.sin(angle) # Sideways distance (Left/Right)

            # --- THE CORRIDOR BOX FILTER ---
            # Is the obstacle physically in the path of the wheels?
            if abs(y) < (self.path_width / 2.0):
                # Is it too close in front of us?
                if 0.0 < x < self.stop_distance:
                    danger_detected = True
                    break # Stop checking, we found a danger

        self.obstacle_in_path = danger_detected

    def cmd_vel_callback(self, msg):
        safe_vel = Twist()

        if self.obstacle_in_path:
            # Danger: Force speed to 0 (Apply brakes)
            safe_vel.linear.x = 0.0
            safe_vel.angular.z = 0.0
            self.get_logger().warn("!!! EMERGENCY STOP !!! Obstacle detected in path.", throttle_duration_sec=1.0)
        else:
            # Safe: Pass the Nav2 command straight through to the motors
            safe_vel.linear.x = msg.linear.x
            safe_vel.angular.z = msg.angular.z

        self.cmd_vel_pub.publish(safe_vel)

    def health_check_callback(self):
        """Failsafe: Stops the robot if the LiDAR cable is pulled or driver crashes"""
        time_since_last_scan = (self.get_clock().now() - self.last_scan_time).nanoseconds / 1e9
        
        if time_since_last_scan > self.sensor_timeout:
            # Lidar is dead: Force stop for safety
            stop_vel = Twist()
            stop_vel.linear.x = 0.0
            stop_vel.angular.z = 0.0
            self.cmd_vel_pub.publish(stop_vel)
            self.obstacle_in_path = True # Lock safety until Lidar recovers
            self.get_logger().error(f"!!! SENSOR TIMEOUT !!! No Lidar data for {time_since_last_scan:.2f}s. Emergency Brake Applied.", throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = SafetyBrakeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
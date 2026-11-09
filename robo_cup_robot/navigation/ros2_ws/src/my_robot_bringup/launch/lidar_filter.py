#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class LidarFilter(Node):
    def __init__(self):
        super().__init__('lidar_filter')
        self.min_cutoff = 0.40  # Anything closer than 40cm is deleted!

        # 1. Listen to raw C1 lidar
        self.sub = self.create_subscription(LaserScan, '/scan_raw', self.scan_callback, 10)
        
        # 2. Publish clean lidar to the rest of ROS 2
        self.pub = self.create_publisher(LaserScan, '/scan', 10)

    def scan_callback(self, msg: LaserScan):
        # Convert any point closer than 0.4m to infinity (invisible)
        msg.ranges = [float('inf') if (r < self.min_cutoff and r > 0.0) else r for r in msg.ranges]
        msg.range_min = self.min_cutoff
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LidarFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
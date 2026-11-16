import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('bag_picking'), 'config', 'params.yaml')
    return LaunchDescription([
        Node(package='bag_picking', executable='bag_locator', name='bag_locator',
             parameters=[params], output='screen'),
        Node(package='bag_picking', executable='bag_picker', name='bag_picker',
             parameters=[params], output='screen'),
    ])

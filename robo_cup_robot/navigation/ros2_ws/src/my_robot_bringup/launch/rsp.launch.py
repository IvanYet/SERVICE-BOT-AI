import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Locate xacro file inside my_robot_description
    pkg_description = get_package_share_directory('my_robot_description')
    xacro_file = os.path.join(pkg_description, 'urdf', 'robot.urdf.xacro')

    # Convert Xacro into XML text
    robot_desc_xml = xacro.process_file(xacro_file).toxml()

    # Wrap the XML in ParameterValue to prevent string truncation / type deduction issues
    robot_description = ParameterValue(robot_desc_xml, value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),

        # 1. Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': use_sim_time,
                'transform_tolerance': 0.05
            }]
        )

    ])



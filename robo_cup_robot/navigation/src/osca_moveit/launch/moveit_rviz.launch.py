import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("final_robot", package_name="osca_moveit")
        .robot_description(file_path="config/final_robot.urdf.xacro")
        .to_moveit_configs()
    )

    # Path to the RViz configuration file
    pkg_share = get_package_share_directory("osca_moveit")
    rviz_config_file = os.path.join(pkg_share, "config", "moveit.rviz")

    # Start RViz with simulation time enabled
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription([rviz_node])
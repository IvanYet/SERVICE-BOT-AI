from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("final_robot", package_name="osca_moveit")
        .robot_description(file_path="config/final_robot.urdf.xacro")
        .to_moveit_configs()
    )

    # Start the actual move_group node with use_sim_time enabled directly
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription([move_group_node])
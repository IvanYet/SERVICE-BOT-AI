import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # 1. Package Directories
    # (If your package is named 'my_robot_gazebo', change 'robot_gazebo' to 'my_robot_gazebo')
    pkg_gazebo = get_package_share_directory('robot_gazebo')
    pkg_description = get_package_share_directory('robot_description')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # 2. Process XACRO into robot_description
    xacro_file = os.path.join(pkg_description, 'urdf', 'final_robot.urdf.xacro')
    robot_desc_xml = xacro.process_file(xacro_file).toxml()
    robot_description = ParameterValue(robot_desc_xml, value_type=str)

    # 3. Path to your Virtual World File
    # Handles both 'world' and 'worlds' folder names:
    world_file = os.path.join(pkg_gazebo, 'world', 'sim_world.world')
    if not os.path.exists(world_file):
        world_file = os.path.join(pkg_gazebo, 'world', 'sim_world.world')

    # 4. Include Gazebo Engine
    gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )

    # 5. Robot State Publisher (Publishes 3D model transforms with sim_time = true)
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
            'transform_tolerance': 0.05
        }]
    )

    # 6. Spawn Robot Node
    # (Set this to True if your .world file does NOT have the robot inside it!)
    spawn_robot = True

    nodes_to_launch = [
        gazebo_cmd,
        rsp_node
    ]

    if spawn_robot:
        spawn_entity_node = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_entity',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'front_diff_robot',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.1'
            ],
            output='screen'
        )
        nodes_to_launch.append(spawn_entity_node)

    return LaunchDescription(nodes_to_launch)
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Make sure this says 'robot_gazebo'!
    pkg_gazebo = get_package_share_directory('robot_gazebo')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    # Path to map and params inside the installed share directory
    map_file = os.path.join(pkg_gazebo, 'maps', 'sim_map.yaml')
    params_file = os.path.join(pkg_gazebo, 'config', 'nav2_sim_params.yaml')
    rviz_config = os.path.join(nav2_bringup_dir, 'rviz', 'nav2_default_view.rviz')

    # Launch Nav2
    nav2_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'map': map_file,
            'use_sim_time': 'true',
            'params_file': params_file,
            'autostart': 'true'
        }.items()
    )

    # Launch RViz2
    rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    return LaunchDescription([
        nav2_cmd,
        rviz_cmd
    ])
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Package Directories
    bringup_dir = get_package_share_directory('my_esp_driver')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    # 2. File Paths (Ensure these match your actual filenames!)
    default_map_path = os.path.join(bringup_dir, 'maps', 'my_room_map.yaml')
    default_params_path = os.path.join(bringup_dir, 'config', 'nav2_params.yaml')
    
    # Official Nav2 default RViz layout (Already has all costmaps and goal tools loaded!)
    default_rviz_path = os.path.join(nav2_bringup_dir, 'rviz', 'nav2_default_view.rviz')

    # 3. Launch Configurations
    map_yaml_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # 4. Declare Arguments
    declare_map_cmd = DeclareLaunchArgument(
        'map',
        default_value=default_map_path,
        description='Full path to map file to load'
    )

    declare_params_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_path,
        description='Full path to nav2_params.yaml'
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock (false for real robot)'
    )

    # 5. Nav2 Bringup (AMCL + Costmaps + Planners + Lifecycle Manager)
    nav2_bringup_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'map': map_yaml_file,
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'autostart': 'true' # Automatically activates all lifecycle nodes!
        }.items()
    )

    # 6. RViz2 Node
    rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', default_rviz_path],
        parameters=[{'use_sim_time': False}],
        output='screen'
    )

    return LaunchDescription([
        declare_map_cmd,
        declare_params_cmd,
        declare_use_sim_time_cmd,
        nav2_bringup_cmd,
        rviz_cmd
    ])
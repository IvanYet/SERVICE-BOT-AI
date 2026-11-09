import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # --- Launch Configurations ---
    use_sim_time = LaunchConfiguration('use_sim_time')
    channel_type = LaunchConfiguration('channel_type')
    serial_port = LaunchConfiguration('serial_port')
    serial_baudrate = LaunchConfiguration('serial_baudrate')
    frame_id = LaunchConfiguration('frame_id')
    inverted = LaunchConfiguration('inverted')
    angle_compensate = LaunchConfiguration('angle_compensate')
    scan_mode = LaunchConfiguration('scan_mode')

    # --- URDF / Xacro Processing ---
    pkg_description = get_package_share_directory('my_robot_description')
    pkg_bringup = get_package_share_directory('my_robot_bringup')
    
    xacro_file = os.path.join(pkg_description, 'urdf', 'robot.urdf.xacro')

    # Convert Xacro into XML text
    robot_desc_xml = xacro.process_file(xacro_file).toxml()
    robot_description = ParameterValue(robot_desc_xml, value_type=str)

    # --- EKF Configuration Path ---
    ekf_config_file = os.path.join(pkg_bringup, 'config', 'ekf.yaml')

    # --- Nodes Definition ---
    # 1. Robot State Publisher
    rsp_node = Node(
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

    # 2. SLLiDAR C1 Node
    lidar_node = Node(
        package='sllidar_ros2',
        executable='sllidar_node',
        name='sllidar_node',
        parameters=[{
            'channel_type': channel_type,
            'serial_port': serial_port,
            'serial_baudrate': serial_baudrate,
            'frame_id': frame_id,
            'inverted': inverted,
            'angle_compensate': angle_compensate,
            'scan_mode': scan_mode
        }],
        remappings=[('/scan', '/scan_raw')],
        output='screen'
    )

    # 3. LiDAR Filter Node (< 0.4m cutoff)
    lidar_filter_node = Node(
        package='my_robot_bringup',
        executable='lidar_filter.py',
        name='lidar_filter',
        output='screen'
    )

    # 4. EKF Node (robot_localization)
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            ekf_config_file,
            {'use_sim_time': use_sim_time}
        ]
    )

    return LaunchDescription([
        # --- Launch Arguments ---
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'channel_type',
            default_value='serial',
            description='Specifying channel type of lidar'
        ),
        DeclareLaunchArgument(
            'serial_port',
            default_value='/dev/ttyUSB0',
            description='Specifying usb port to connected lidar'
        ),
        DeclareLaunchArgument(
            'serial_baudrate',
            default_value='460800',
            description='Specifying usb port baudrate to connected lidar'
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='lidar_link',
            description='Specifying frame_id of lidar'
        ),
        DeclareLaunchArgument(
            'inverted',
            default_value='false',
            description='Specifying whether or not to invert scan data'
        ),
        DeclareLaunchArgument(
            'angle_compensate',
            default_value='true',
            description='Specifying whether or not to enable angle compensate of scan data'
        ),
        DeclareLaunchArgument(
            'scan_mode',
            default_value='Standard',
            description='Specifying scan mode of lidar'
        ),

        # --- Launch the Nodes ---
        rsp_node,
        lidar_node,
        lidar_filter_node,
        ekf_node
    ])
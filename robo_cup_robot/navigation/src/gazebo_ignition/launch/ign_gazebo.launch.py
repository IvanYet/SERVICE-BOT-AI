import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # 1. Package directories
    pkg_gazebo_ignition = get_package_share_directory('gazebo_ignition')
    pkg_robot_description = get_package_share_directory('robot_description')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Parent directory for resolving mesh files
    install_share_dir = os.path.dirname(pkg_robot_description)

    # 2. Tell Ignition Gazebo where to find your meshes
    set_ign_resource = SetEnvironmentVariable(
        'IGN_GAZEBO_RESOURCE_PATH',
        f"{install_share_dir}:{os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')}"
    )
    set_gz_resource = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        f"{install_share_dir}:{os.environ.get('GZ_SIM_RESOURCE_PATH', '')}"
    )

    # 3. Path to Virtual World File
    world_file = os.path.join(pkg_gazebo_ignition, 'world', 'room_world.sdf')
    if not os.path.exists(world_file):
        world_file = os.path.join(pkg_gazebo_ignition, 'worlds', 'room_world.sdf')

    # 4. Parse URDF from xacro
    xacro_file = os.path.join(pkg_robot_description, 'urdf', 'final_robot.urdf.xacro')
    robot_desc_xml = xacro.process_file(xacro_file).toxml()
    robot_description = ParameterValue(robot_desc_xml, value_type=str)

    # 5. Start Gazebo Sim with the world
    gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r -v 3 {world_file}'}.items()
    )

    # 6. Robot State Publisher (using simulation time)
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

    # 7. Spawn Robot Using -string directly into room_world
    spawn_entity_node = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_entity',
        arguments=[
            '-world', 'room_world',
            '-name', 'final_robot',
            '-string', robot_desc_xml,
            '-x', '-2.5',
            '-y', '-2.5',
            '-z', '0.1',
            '-Y', '1.57'
        ],
        output='screen'
    )

    # Delay spawn by 3 seconds so Gazebo's /create service is up
    delayed_spawn = TimerAction(
        period=3.0,
        actions=[spawn_entity_node]
    )

    # 8. Controller Spawners (with use_sim_time enabled)
    jsb_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    arm_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller'],
        output='screen'
    )

    gripper_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller'],
        output='screen'
    )

    # Delay spawners by 6 seconds so robot and controller_manager are fully loaded
    delayed_controllers = TimerAction(
        period=6.0,
        actions=[jsb_spawner, arm_spawner, gripper_spawner]
    )

    # 9. ROS-Gazebo Bridge (Cleaned: removed duplicate /joint_states)
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
        ],
        output='screen'
    )

    return LaunchDescription([
        set_ign_resource,
        set_gz_resource,
        gazebo_cmd,
        rsp_node,
        bridge_node,
        delayed_spawn,
        delayed_controllers
    ])
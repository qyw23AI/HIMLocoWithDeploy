"""Launch file for motor debug node with RViz visualization.

Sends zero parameters to all motors, reads back joint angles,
and displays the robot state in RViz for visual verification.

Usage:
  ros2 launch deploy_cpp motor_debug.launch.py
  ros2 launch deploy_cpp motor_debug.launch.py port0:=/dev/ttyUSB2 port1:=/dev/ttyUSB3
  ros2 launch deploy_cpp motor_debug.launch.py rviz:=false   # 不启动 RViz
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('deploy_cpp')

    # URDF file
    urdf_file = os.path.join(pkg_dir, 'robot', 'mybot_v2', 'urdf', 'mybot_v2.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # RViz config
    rviz_config = os.path.join(pkg_dir, 'config', 'mybot_v2.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('port0', default_value='/dev/ttyUSB1',
                              description='Serial port for front legs'),
        DeclareLaunchArgument('port1', default_value='/dev/ttyUSB0',
                              description='Serial port for rear legs'),
        DeclareLaunchArgument('rate_hz', default_value='50.0',
                              description='Read rate in Hz'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Launch RViz for visualization'),
        DeclareLaunchArgument('config_file',
                              default_value=os.path.join(pkg_dir, 'config', 'robots', 'mybot_v2_real.yaml'),
                              description='Robot configuration YAML file'),

        # Motor debug node (publishes /joint_states)
        Node(
            package='deploy_cpp',
            executable='motor_debug_node',
            name='motor_debug_node',
            output='screen',
            parameters=[{
                'port0': LaunchConfiguration('port0'),
                'port1': LaunchConfiguration('port1'),
                'rate_hz': LaunchConfiguration('rate_hz'),
                'robot_config_file': LaunchConfiguration('config_file'),
            }],
        ),

        # Robot State Publisher (URDF -> TF)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            condition=IfCondition(LaunchConfiguration('rviz')),
            parameters=[{
                'robot_description': robot_description,
            }],
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            condition=IfCondition(LaunchConfiguration('rviz')),
            arguments=['-d', rviz_config],
        ),
    ])

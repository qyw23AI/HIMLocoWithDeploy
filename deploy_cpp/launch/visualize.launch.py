"""Launch file for RViz visualization of Mybot quadruped robot.

Starts:
  1. robot_state_publisher  - loads URDF and publishes TF from /joint_states
  2. joint_state_publisher   - publishes default joint states (standing pose)
  3. rviz2                   - 3D visualization with pre-configured display

Usage (standalone, robot in default standing pose):
  ros2 launch deploy_cpp visualize.launch.py

Usage (with deploy node, real joint states):
  ros2 launch deploy_cpp visualize.launch.py use_jsp:=false
  ros2 launch deploy_cpp deploy.launch.py debug_no_motor:=true

Usage (with GUI sliders to manually move joints):
  ros2 launch deploy_cpp visualize.launch.py use_gui:=true
"""

import os
import math

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('deploy_cpp')

    # URDF file path
    urdf_file = os.path.join(pkg_dir, 'robot', 'mybot_v2', 'urdf', 'mybot_v2.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # RViz config file path
    rviz_config = os.path.join(pkg_dir, 'config', 'mybot_v2.rviz')

    # Default standing joint positions [rad] (from robot_config.h STANDUP_TARGET_POS)
    # FR: hip=-0.1, thigh=0.67, calf=-1.3
    # FL: hip=0.1,  thigh=0.67, calf=-1.3
    # RR: hip=-0.1, thigh=0.67, calf=-1.3
    # RL: hip=0.1,  thigh=0.67, calf=-1.3
    default_joint_positions = {
        'FR_hip_joint':   -0.1,
        'FR_thigh_joint':  0.67,
        'FR_calf_joint':  -1.3,
        'FL_hip_joint':    0.1,
        'FL_thigh_joint':  0.67,
        'FL_calf_joint':  -1.3,
        'RR_hip_joint':   -0.1,
        'RR_thigh_joint':  0.67,
        'RR_calf_joint':  -1.3,
        'RL_hip_joint':    0.1,
        'RL_thigh_joint':  0.67,
        'RL_calf_joint':  -1.3,
    }

    return LaunchDescription([
        DeclareLaunchArgument('use_jsp', default_value='true',
                              description='Start joint_state_publisher (false when using deploy_node)'),
        DeclareLaunchArgument('use_gui', default_value='false',
                              description='Use joint_state_publisher_gui instead of static publisher'),

        # Robot State Publisher: reads URDF, subscribes to /joint_states,
        # publishes TF transforms for all links
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
            }],
        ),

        # Joint State Publisher (static default pose)
        # Only started when use_jsp:=true and use_gui:=false
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_jsp')),
            parameters=[{
                'zeros': default_joint_positions,
            }],
        ),

        # RViz2: 3D visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        ),
    ])

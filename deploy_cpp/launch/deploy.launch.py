"""Launch file for deploy_cpp deploy_node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_dir = get_package_share_directory('deploy_cpp')
    default_cfg = os.path.join(pkg_dir, 'config', 'robots', 'mybot.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('robot_config_file', default_value=default_cfg,
                              description='Path to robot runtime yaml config'),
        DeclareLaunchArgument('debug_no_motor', default_value='false',
                              description='Use fake motor driver for testing'),
        DeclareLaunchArgument('sim_mode', default_value='false',
                              description='Use MuJoCo bridge topic motor driver'),
        DeclareLaunchArgument('sim_pingpong_mode', default_value='false',
                              description='Enable state-triggered ping-pong control timing in deploy_node'),
        DeclareLaunchArgument('enable_rl_joint_csv', default_value='false',
                              description='Record RL joint q/dq/tau to CSV in real motor mode'),
        DeclareLaunchArgument('rl_joint_csv_path', default_value='',
                              description='Absolute path of RL joint CSV output file'),
        DeclareLaunchArgument('rl_joint_csv_flush_interval', default_value='50',
                              description='Flush CSV every N rows'),

        Node(
            package='deploy_cpp',
            executable='deploy_node',
            name='deploy_node',
            output='screen',
            parameters=[{
                'robot_config_file': LaunchConfiguration('robot_config_file'),
                'debug_no_motor': LaunchConfiguration('debug_no_motor'),
                'sim_mode': LaunchConfiguration('sim_mode'),
                'sim_pingpong_mode': LaunchConfiguration('sim_pingpong_mode'),
                'enable_rl_joint_csv': LaunchConfiguration('enable_rl_joint_csv'),
                'rl_joint_csv_path': LaunchConfiguration('rl_joint_csv_path'),
                'rl_joint_csv_flush_interval': LaunchConfiguration('rl_joint_csv_flush_interval'),
            }],
        ),
    ])

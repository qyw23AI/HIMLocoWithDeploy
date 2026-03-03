"""Launch file for deploy_cpp deploy_node."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('policy_path', default_value='policy/policy.pt',
                              description='Path to JIT policy model'),
        DeclareLaunchArgument('device', default_value='cuda:0',
                              description='Torch device (cuda:0 or cpu)'),
        DeclareLaunchArgument('port0', default_value='/dev/ttyUSB0',
                              description='Serial port for front legs'),
        DeclareLaunchArgument('port1', default_value='/dev/ttyUSB1',
                              description='Serial port for rear legs'),
        DeclareLaunchArgument('debug_no_motor', default_value='false',
                              description='Use fake motor driver for testing'),
        DeclareLaunchArgument('imu_topic', default_value='/livox/imu',
                              description='IMU topic name'),

        Node(
            package='deploy_cpp',
            executable='deploy_node',
            name='deploy_node',
            output='screen',
            parameters=[{
                'policy_path': LaunchConfiguration('policy_path'),
                'device': LaunchConfiguration('device'),
                'port0': LaunchConfiguration('port0'),
                'port1': LaunchConfiguration('port1'),
                'debug_no_motor': LaunchConfiguration('debug_no_motor'),
                'imu_topic': LaunchConfiguration('imu_topic'),
            }],
        ),
    ])

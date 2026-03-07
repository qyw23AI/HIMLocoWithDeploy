"""Launch file for MuJoCo simulation mode.

Starts the C++ deploy_node in sim_mode, which communicates with
the Python MuJoCo simulation node via ROS2 topics.

NOTE: The Python MuJoCo sim node must be started separately in the
      mujoco_sim conda environment:
        conda activate mujoco_sim
        source /opt/ros/humble/setup.bash
        python3 sim/mujoco_sim_node.py

Usage (terminal 1 - MuJoCo sim):
  cd ~/humble/Quadruped/HIMLoco/deploy_cpp
  conda activate mujoco_sim && source /opt/ros/humble/setup.bash
  python3 sim/mujoco_sim_node.py

Usage (terminal 2 - deploy_node):
  source /opt/ros/humble/setup.bash
  source ~/humble/Quadruped/HIMLoco/install/setup.bash
  ros2 launch deploy_cpp sim.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('deploy_cpp')

    # URDF for robot_state_publisher (optional RViz)
    urdf_file = os.path.join(pkg_dir, 'robot', 'mybot', 'urdf', 'mybot.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        DeclareLaunchArgument('policy_path', default_value=os.path.join(
            pkg_dir, 'policy', 'policy.pt'),
            description='Path to policy.pt'),
        DeclareLaunchArgument('device', default_value='cuda:0',
                              description='Torch device'),
        DeclareLaunchArgument('sim_pingpong_mode', default_value='false',
                              description='Enable state-triggered ping-pong control timing in deploy_node'),

        # Deploy node in simulation mode
        Node(
            package='deploy_cpp',
            executable='deploy_node',
            name='deploy_node',
            output='screen',
            parameters=[{
                'sim_mode': True,
                'sim_pingpong_mode': LaunchConfiguration('sim_pingpong_mode'),
                'debug_no_motor': False,
                'policy_path': LaunchConfiguration('policy_path'),
                'device': LaunchConfiguration('device'),
            }],
        ),

        # Robot state publisher (for RViz, optional)
        # MuJoCo sim node already publishes /joint_states
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
            }],
        ),
    ])

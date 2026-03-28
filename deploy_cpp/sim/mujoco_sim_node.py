#!/usr/bin/env python3
"""
MuJoCo simulation node for Mybot quadruped robot.

Bridges MuJoCo physics simulation with the C++ deploy_node via ROS2 topics:
    - Subscribes: /mujoco/joint_cmd
            - 12 floats: target positions
            - 14 floats: target + scalar kp/kd
            - 36 floats: target + per-joint kp[12] + per-joint kd[12]
  - Publishes:  /mujoco/joint_state (Float32MultiArray, 24 floats: 12 pos + 12 vel)
  - Publishes:  /fast_livo2/state6_imu_prop  (Float32MultiArray, 6 floats: 3 ang_vel + 3 proj_grav)
  - Publishes:  /joint_states       (JointState, for RViz visualization)

Usage:
  conda activate mujoco_sim
  source /opt/ros/humble/setup.bash
  python3 sim/mujoco_sim_node.py

  Or via launch:
  ros2 launch deploy_cpp sim.launch.py
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
import yaml

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

# MuJoCo
import mujoco
import mujoco.viewer

# ============================================================
# Robot configuration (must match robot_config.h)
# ============================================================
NUM_JOINTS = 12

# Joint names in DOF order
DEFAULT_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

# Default standing pose
DEFAULT_DOF_POS = np.array([
    -0.35,  0.99, -2.57,   # FR
     0.35,  0.99, -2.57,   # FL
    -0.35,  0.99, -2.57,   # RR
     0.35,  0.99, -2.57,   # RL
], dtype=np.float32)

# Torque limit (from mybot.xml motor ctrlrange)
TORQUE_LIMIT = 33.5


def load_robot_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    required = [
        'dt', 'decimation', 'num_of_dofs', 'default_dof_pos', 'joint_names',
        'joint_controller_names', 'torque_limits', 'kp_joint', 'kd_joint', 'joint_transmission_ratio',
        'mujoco_xml_relpath'
    ]
    for key in required:
        if key not in cfg:
            raise RuntimeError(f'Missing required key in robot yaml: {key}')

    if int(cfg['num_of_dofs']) != NUM_JOINTS:
        raise RuntimeError('num_of_dofs must be 12 for this node')

    if len(cfg['joint_names']) != NUM_JOINTS:
        raise RuntimeError('joint_names length must be 12')
    if len(cfg['joint_controller_names']) != NUM_JOINTS:
        raise RuntimeError('joint_controller_names length must be 12')
    if len(cfg['default_dof_pos']) != NUM_JOINTS:
        raise RuntimeError('default_dof_pos length must be 12')
    if len(cfg['torque_limits']) != NUM_JOINTS:
        raise RuntimeError('torque_limits length must be 12')
    if len(cfg['joint_transmission_ratio']) != NUM_JOINTS:
        raise RuntimeError('joint_transmission_ratio length must be 12')

    return cfg


def parse_scalar_or_array(value, length: int, name: str) -> np.ndarray:
    """Parse YAML scalar or length-N sequence into float32 numpy array."""
    if isinstance(value, (int, float)):
        return np.full(length, float(value), dtype=np.float32)
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise RuntimeError(f'{name} length must be {length}')
        return np.array(value, dtype=np.float32)
    raise RuntimeError(f'{name} must be scalar or length-{length} array')


def resolve_path(pkg_dir: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(pkg_dir, path_value)


class MujocoSimNode(Node):
    """ROS2 node that runs MuJoCo simulation and publishes sensor data."""

    def __init__(self, xml_path: str, cfg: dict):
        super().__init__('mujoco_sim_node')
        self.cfg = cfg
        self.control_dt = float(cfg['control_dt']) if 'control_dt' in cfg else float(cfg['dt']) * int(cfg['decimation'])
        self.sim_dt = float(cfg['dt'])
        self.decimation = int(cfg['decimation'])
        self.joint_names = list(cfg['joint_names'])
        self.joint_controller_names = list(cfg['joint_controller_names'])
        self.default_dof_pos = np.array(cfg['default_dof_pos'], dtype=np.float32)
        self.torque_limits = np.array(cfg['torque_limits'], dtype=np.float32)
        self.joint_transmission_ratio = np.array(cfg['joint_transmission_ratio'], dtype=np.float32)

        # Optional pure-simulation ping-pong mode:
        # publish state -> wait for /mujoco/joint_cmd -> step physics
        self.declare_parameter('pingpong_mode', False)
        self.pingpong_mode = bool(self.get_parameter('pingpong_mode').value)

        # ---- Load MuJoCo model ----
        self.get_logger().info(f'Loading MuJoCo model: {xml_path}')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Verify timestep
        assert abs(self.model.opt.timestep - self.sim_dt) < 1e-6, \
            f"XML timestep {self.model.opt.timestep} != expected {self.sim_dt}"

        # ---- Build joint index mapping ----
        # Map JOINT_NAMES to MuJoCo joint qpos/qvel indices
        self.joint_qpos_idx = []  # qpos index for each DOF
        self.joint_qvel_idx = []  # qvel index for each DOF
        self.actuator_idx = []    # actuator index for each DOF

        for i, name in enumerate(self.joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                self.get_logger().error(f'Joint "{name}" not found in model!')
                sys.exit(1)
            # For hinge joints: qpos has 1 element, qvel has 1 element
            self.joint_qpos_idx.append(self.model.jnt_qposadr[jid])
            self.joint_qvel_idx.append(self.model.jnt_dofadr[jid])

        # Map actuators in the exact policy DOF order from YAML.
        for name in self.joint_controller_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                self.get_logger().error(f'Actuator "{name}" not found in model!')
                sys.exit(1)
            self.actuator_idx.append(aid)

        # Find IMU site
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")

        # ---- Set initial pose ----
        mujoco.mj_resetData(self.model, self.data)
        for i in range(NUM_JOINTS):
            self.data.qpos[self.joint_qpos_idx[i]] = self.default_dof_pos[i]
        mujoco.mj_forward(self.model, self.data)

        # ---- Target positions (updated by subscriber) ----
        self.target_pos = self.default_dof_pos.copy()
        self.target_kp = parse_scalar_or_array(cfg['kp_joint'], NUM_JOINTS, 'kp_joint')
        self.target_kd = parse_scalar_or_array(cfg['kd_joint'], NUM_JOINTS, 'kd_joint')
        self.cmd_lock = threading.Lock()
        self.cmd_event = threading.Event()

        # ---- ROS2 QoS Profiles ----
        # Low-latency QoS for real-time control/sensor data:
        #   BEST_EFFORT, depth=1, VOLATILE
        qos_fast = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Standard QoS for visualization (latency-insensitive)
        qos_viz = QoSProfile(depth=10)

        # ---- ROS2 Publishers ----
        self.pub_joint_state = self.create_publisher(
            Float32MultiArray, '/mujoco/joint_state', qos_fast)

        self.pub_imu = self.create_publisher(
            Float32MultiArray, '/fast_livo2/state6_imu_prop', qos_fast)

        self.pub_rviz_joint = self.create_publisher(
            JointState, '/joint_states', qos_viz)

        # ---- ROS2 Subscriber ----
        self.sub_cmd = self.create_subscription(
            Float32MultiArray, '/mujoco/joint_cmd', self.cmd_callback, qos_fast)

        self.get_logger().info('MuJoCo sim node initialized.')
        self.get_logger().info(f'  Sim DT: {self.sim_dt}s, Decimation: {self.decimation}, Control DT: {self.control_dt}s')
        self.get_logger().info(
            f'  PD gains (DOF0): kp={self.target_kp[0]:.3f}, kd={self.target_kd[0]:.3f}')

    def cmd_callback(self, msg: Float32MultiArray):
        """Receive target joint positions from deploy_node."""
        if len(msg.data) >= NUM_JOINTS:
            with self.cmd_lock:
                self.target_pos[:] = msg.data[:NUM_JOINTS]
                # Optional kp/kd payloads:
                # 14 = target + scalar kp/kd, 36 = target + kp[12] + kd[12]
                if len(msg.data) >= NUM_JOINTS * 3:
                    self.target_kp[:] = msg.data[NUM_JOINTS:2 * NUM_JOINTS]
                    self.target_kd[:] = msg.data[2 * NUM_JOINTS:3 * NUM_JOINTS]
                elif len(msg.data) >= NUM_JOINTS + 2:
                    self.target_kp.fill(msg.data[NUM_JOINTS])
                    self.target_kd.fill(msg.data[NUM_JOINTS + 1])
            self.cmd_event.set()

    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions [rad]."""
        return np.array([self.data.qpos[idx] for idx in self.joint_qpos_idx],
                        dtype=np.float32)

    def get_joint_vel(self) -> np.ndarray:
        """Get current joint velocities [rad/s]."""
        return np.array([self.data.qvel[idx] for idx in self.joint_qvel_idx],
                        dtype=np.float32)

    def get_imu_data(self) -> tuple:
        """
        Get IMU sensor data from MuJoCo.
        Returns: (ang_vel[3], projected_gravity[3])
        """
        # Angular velocity from gyro sensor
        gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "Body_Gyro")
        gyro_adr = self.model.sensor_adr[gyro_id]
        ang_vel = self.data.sensordata[gyro_adr:gyro_adr + 3].copy()

        # Compute projected gravity from body quaternion
        quat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "Body_Quat")
        quat_adr = self.model.sensor_adr[quat_id]
        quat = self.data.sensordata[quat_adr:quat_adr + 4].copy()  # w, x, y, z

        # Rotate world gravity [0, 0, -1] into body frame using quaternion
        # MuJoCo quaternion convention: (w, x, y, z)
        gravity_world = np.array([0.0, 0.0, -1.0])
        proj_grav = self._rotate_vector_by_quat_inv(gravity_world, quat)

        return ang_vel.astype(np.float32), proj_grav.astype(np.float32)

    @staticmethod
    def _rotate_vector_by_quat_inv(v, q):
        """Rotate vector v by inverse of quaternion q (w,x,y,z)."""
        w, x, y, z = q
        # Correct Rotation matrix from quaternion (w, x, y, z)
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])
        # Transpose to get inverse rotation (world -> body)
        return R.T @ v

    def step_sim(self):
        """Step MuJoCo simulation with PD control (per sub-step, matching IsaacGym)."""
        with self.cmd_lock:
            target = self.target_pos.copy()
            kp = self.target_kp.copy()
            kd = self.target_kd.copy()

        # Step simulation (DECIMATION sub-steps)
        # PD torque is recomputed at EVERY sub-step using latest dof_pos/dof_vel
        # This matches IsaacGym's _compute_torques() called inside the decimation loop
        for _ in range(self.decimation):
            cur_pos = self.get_joint_pos()
            cur_vel = self.get_joint_vel()
            tau = kp * (target - cur_pos) - kd * cur_vel
            tau = np.clip(tau, -self.torque_limits, self.torque_limits)
            for i in range(NUM_JOINTS):
                self.data.ctrl[self.actuator_idx[i]] = tau[i]
            mujoco.mj_step(self.model, self.data)

    def publish_state(self):
        """Publish joint states and IMU data."""
        pos = self.get_joint_pos()
        vel = self.get_joint_vel()
        ang_vel, proj_grav = self.get_imu_data()

        # /mujoco/joint_state: 24 floats (12 pos + 12 vel)
        state_msg = Float32MultiArray()
        state_msg.data = pos.tolist() + vel.tolist()
        self.pub_joint_state.publish(state_msg)

        # /fast_livo2/state6_imu_prop : 6 floats (3 ang_vel + 3 proj_grav)
        imu_msg = Float32MultiArray()
        imu_msg.data = ang_vel.tolist() + proj_grav.tolist()
        self.pub_imu.publish(imu_msg)

        # /joint_states: for RViz
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.name = self.joint_names
        js_msg.position = pos.astype(float).tolist()
        js_msg.velocity = vel.astype(float).tolist()
        js_msg.effort = [0.0] * NUM_JOINTS
        self.pub_rviz_joint.publish(js_msg)

    def run(self):
        """Main simulation loop with MuJoCo viewer."""
        self.get_logger().info('Starting MuJoCo simulation with viewer...')
        self.get_logger().info('Waiting for /mujoco/joint_cmd from deploy_node...')
        if self.pingpong_mode:
            self.get_logger().info(
                'Ping-pong mode enabled: publish state -> wait cmd -> step physics (no wall-clock rate control).')

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step_count = 0
            while viewer.is_running() and rclpy.ok():
                if self.pingpong_mode:
                    # Process callbacks and publish latest state snapshot.
                    rclpy.spin_once(self, timeout_sec=0)
                    self.publish_state()
                    viewer.sync()

                    # Wait until controller sends next command.
                    while viewer.is_running() and rclpy.ok():
                        if self.cmd_event.wait(timeout=0.001):
                            break
                        rclpy.spin_once(self, timeout_sec=0)

                    if not viewer.is_running() or not rclpy.ok():
                        break

                    self.cmd_event.clear()
                    self.step_sim()
                else:
                    loop_start = time.time()

                    # Process ROS2 callbacks
                    rclpy.spin_once(self, timeout_sec=0)

                    # Step simulation
                    self.step_sim()

                    # Publish sensor data
                    self.publish_state()

                    # Update viewer
                    viewer.sync()

                    # Rate control
                    elapsed = time.time() - loop_start
                    sleep_time = self.control_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                # Print status periodically
                step_count += 1
                if step_count % 50 == 0:  # Every 50 simulation control steps
                    pos = self.get_joint_pos()
                    self.get_logger().info(
                        f'Step {step_count}: FR_hip={pos[0]:.3f} '
                        f'FR_thigh={pos[1]:.3f} FR_calf={pos[2]:.3f}'
                        f'FL_hip={pos[3]:.3f} FL_thigh={pos[4]:.3f} FL_calf={pos[5]:.3f}'
                        f'RR_hip={pos[6]:.3f} RR_thigh={pos[7]:.3f} RR_calf={pos[8]:.3f}'
                        f'RL_hip={pos[9]:.3f} RL_thigh={pos[10]:.3f} RL_calf={pos[11]:.3f}'
                    )

        self.get_logger().info('MuJoCo viewer closed. Shutting down.')


def main():
    rclpy.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-config', default='',
                        help='Path to robot yaml config')
    args, _ = parser.parse_known_args()

    # Find package directory (source workspace preferred)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(script_dir)  # deploy_cpp/

    cfg_path = args.robot_config
    if not cfg_path:
        cfg_path = os.path.join(pkg_dir, 'config', 'robots', 'mybot.yaml')
    if not os.path.exists(cfg_path):
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory('deploy_cpp')
            if not args.robot_config:
                cfg_path = os.path.join(pkg_share, 'config', 'robots', 'mybot.yaml')
        except Exception:
            pass

    if not os.path.exists(cfg_path):
        print(f'ERROR: Cannot find robot yaml config at {cfg_path}')
        sys.exit(1)

    cfg = load_robot_config(cfg_path)
    xml_path = resolve_path(pkg_dir, cfg['mujoco_xml_relpath'])
    if not os.path.exists(xml_path):
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory('deploy_cpp')
            xml_path = resolve_path(pkg_share, cfg['mujoco_xml_relpath'])
        except Exception:
            pass

    if not os.path.exists(xml_path):
        print(f"ERROR: Cannot find MuJoCo XML from config: {cfg['mujoco_xml_relpath']}")
        sys.exit(1)

    print(f"[mujoco_sim_node] robot={cfg.get('robot_name', 'unknown')} cfg={cfg_path}")
    print(f"[mujoco_sim_node] xml={xml_path}")

    node = MujocoSimNode(xml_path, cfg)

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

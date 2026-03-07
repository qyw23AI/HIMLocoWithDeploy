#!/usr/bin/env python3
"""
MuJoCo simulation node for Mybot quadruped robot.

Bridges MuJoCo physics simulation with the C++ deploy_node via ROS2 topics:
  - Subscribes: /mujoco/joint_cmd  (Float32MultiArray, 12 floats: target positions)
  - Publishes:  /mujoco/joint_state (Float32MultiArray, 24 floats: 12 pos + 12 vel)
  - Publishes:  /fast_livo2/state6  (Float32MultiArray, 6 floats: 3 ang_vel + 3 proj_grav)
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
import threading
import numpy as np

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
CONTROL_DT = 0.02      # 50 Hz control loop
SIM_DT = 0.002         # ？？Must match IsaacGym sim.dt (legged_robot_config.py)
DECIMATION = int(CONTROL_DT / SIM_DT)  # 4 sub-steps (must match control.decimation)

# Joint-side PD gains (from robot_config.h)
KP_JOINT = 40.0
KD_JOINT = 1.0

# Joint names in DOF order
JOINT_NAMES = [
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


class MujocoSimNode(Node):
    """ROS2 node that runs MuJoCo simulation and publishes sensor data."""

    def __init__(self, xml_path: str):
        super().__init__('mujoco_sim_node')

        # Optional pure-simulation ping-pong mode:
        # publish state -> wait for /mujoco/joint_cmd -> step physics
        self.declare_parameter('pingpong_mode', False)
        self.pingpong_mode = bool(self.get_parameter('pingpong_mode').value)

        # ---- Load MuJoCo model ----
        self.get_logger().info(f'Loading MuJoCo model: {xml_path}')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Verify timestep
        assert abs(self.model.opt.timestep - SIM_DT) < 1e-6, \
            f"XML timestep {self.model.opt.timestep} != expected {SIM_DT}"

        # ---- Build joint index mapping ----
        # Map JOINT_NAMES to MuJoCo joint qpos/qvel indices
        self.joint_qpos_idx = []  # qpos index for each DOF
        self.joint_qvel_idx = []  # qvel index for each DOF
        self.actuator_idx = []    # actuator index for each DOF

        for i, name in enumerate(JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                self.get_logger().error(f'Joint "{name}" not found in model!')
                sys.exit(1)
            # For hinge joints: qpos has 1 element, qvel has 1 element
            self.joint_qpos_idx.append(self.model.jnt_qposadr[jid])
            self.joint_qvel_idx.append(self.model.jnt_dofadr[jid])

        # Map actuator names (same order as JOINT_NAMES: FR_hip, FR_thigh, ...)
        actuator_names = [
            "FR_hip", "FR_thigh", "FR_calf",
            "FL_hip", "FL_thigh", "FL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
            "RL_hip", "RL_thigh", "RL_calf",
        ]
        for name in actuator_names:
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
            self.data.qpos[self.joint_qpos_idx[i]] = DEFAULT_DOF_POS[i]
        mujoco.mj_forward(self.model, self.data)

        # ---- Target positions (updated by subscriber) ----
        self.target_pos = DEFAULT_DOF_POS.copy()
        self.target_kp = KP_JOINT
        self.target_kd = KD_JOINT
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
            Float32MultiArray, '/fast_livo2/state6', qos_fast)

        self.pub_rviz_joint = self.create_publisher(
            JointState, '/joint_states', qos_viz)

        # ---- ROS2 Subscriber ----
        self.sub_cmd = self.create_subscription(
            Float32MultiArray, '/mujoco/joint_cmd', self.cmd_callback, qos_fast)

        self.get_logger().info('MuJoCo sim node initialized.')
        self.get_logger().info(f'  Sim DT: {SIM_DT}s, Decimation: {DECIMATION}, Control DT: {CONTROL_DT}s')
        self.get_logger().info(f'  PD gains: kp={KP_JOINT}, kd={KD_JOINT}')

    def cmd_callback(self, msg: Float32MultiArray):
        """Receive target joint positions from deploy_node."""
        if len(msg.data) >= NUM_JOINTS:
            with self.cmd_lock:
                self.target_pos[:] = msg.data[:NUM_JOINTS]
                # Optionally receive kp, kd if provided
                if len(msg.data) >= NUM_JOINTS + 2:
                    self.target_kp = msg.data[NUM_JOINTS]
                    self.target_kd = msg.data[NUM_JOINTS + 1]
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
            kp = self.target_kp
            kd = self.target_kd

        # Step simulation (DECIMATION sub-steps)
        # PD torque is recomputed at EVERY sub-step using latest dof_pos/dof_vel
        # This matches IsaacGym's _compute_torques() called inside the decimation loop
        for _ in range(DECIMATION):
            cur_pos = self.get_joint_pos()
            cur_vel = self.get_joint_vel()
            tau = kp * (target - cur_pos) - kd * cur_vel
            tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
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

        # /fast_livo2/state6: 6 floats (3 ang_vel + 3 proj_grav)
        imu_msg = Float32MultiArray()
        imu_msg.data = ang_vel.tolist() + proj_grav.tolist()
        self.pub_imu.publish(imu_msg)

        # /joint_states: for RViz
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.name = list(JOINT_NAMES)
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
                    sleep_time = CONTROL_DT - elapsed
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

    # Find XML path
    # Try relative path first, then package share directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(script_dir)  # deploy_cpp/
    xml_path = os.path.join(pkg_dir, 'robot', 'mybot', 'xml', 'mybot.xml')

    if not os.path.exists(xml_path):
        # Try from ROS2 package share
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory('deploy_cpp')
            xml_path = os.path.join(pkg_share, 'robot', 'mybot', 'xml', 'mybot.xml')
        except Exception:
            pass

    if not os.path.exists(xml_path):
        print(f'ERROR: Cannot find mybot.xml at {xml_path}')
        sys.exit(1)

    node = MujocoSimNode(xml_path)

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

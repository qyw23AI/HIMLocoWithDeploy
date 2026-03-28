/**
 * @file robot_runtime_config.h
 * @brief Runtime robot configuration loaded from YAML.
 *
 * All robot-specific parameters are defined here and populated from YAML.
 * The compile-time dimension constants come from robot_config.h.
 */
#pragma once

#include <array>
#include <string>
#include <vector>

#include "robot_config.h"

namespace deploy {

// ========================== Motor Mapping ==========================

/**
 * @brief Per-joint motor hardware mapping.
 *
 * 每个关节对应一个物理电机，描述：
 *   - motor_id:    电机 CAN ID (1-12)
 *   - port_idx:    串口索引 (0 = port0, 1 = port1)
 *   - is_reversed: 电机方向是否与关节正方向相反
 */
struct MotorMapping {
  int motor_id = 0;
  int port_idx = 0;
  bool is_reversed = false;
};

// ========================== Runtime Config ==========================

struct RobotRuntimeConfig {
  int num_of_dofs = NUM_JOINTS;
  float dt = 0.005f;
  int decimation = 4;

  std::array<float, NUM_JOINTS> fixed_kp{};
  std::array<float, NUM_JOINTS> fixed_kd{};
  std::array<float, NUM_JOINTS> torque_limits{};

  // ---- 关节角度参数 ----
  std::array<float, NUM_JOINTS> default_dof_pos{};
  std::array<float, NUM_JOINTS> standup_target_pos{};
  std::array<float, NUM_JOINTS> policy_dof_pos{};
  std::array<float, NUM_JOINTS> joint_pos_lower{};
  std::array<float, NUM_JOINTS> joint_pos_upper{};

  // ---- 关节名称 & 控制器名称 ----
  std::array<std::string, NUM_JOINTS> joint_names{};
  std::array<std::string, NUM_JOINTS> joint_controller_names{};

  // ---- 关节-电机映射 ----
  //
  // joint_mapping[policy_dof_idx] = motor CAN ID (1-12)
  //   policy DOF 索引 → 对应的电机 CAN ID
  //   实机部署时在 YAML 中设置, 仿真时可省略 (默认 identity)
  //
  // motor_map[policy_dof_idx] 包含该 policy DOF 对应的电机硬件信息
  //   motor_id, port_idx, is_reversed 直接按 policy 顺序索引
  //   MotorDriver::send_single(dof_idx, ...) 直接用 motor_map[dof_idx]
  //
  std::array<int, NUM_JOINTS> joint_mapping{};
  std::array<bool, NUM_JOINTS> motor_is_reversed{};
  std::array<MotorMapping, NUM_JOINTS> motor_map{};

  std::vector<int> wheel_indices;

  // ---- 传动比 ----
  // q_motor = q_joint * ratio
  std::array<float, NUM_JOINTS> joint_transmission_ratio{};

  // ---- 路径 & 设备 ----
  std::string policy_path = "policy/policy.pt";
  std::string device = "cuda:0";
  std::string port0 = "/dev/motor_front";
  std::string port1 = "/dev/motor_rear";
  std::string imu_topic = "/fast_livo2/state6";
  float imu_yaw_correction_deg = 0.0f;

  // ---- PD 控制参数 ----
  // 支持按 policy DOF 顺序配置每个关节的增益:
  // DOF 0-2 FL, 3-5 FR, 6-8 RL, 9-11 RR
  std::array<float, NUM_JOINTS> kp_joint{};
  std::array<float, NUM_JOINTS> kd_joint{};
  float kd_damp_motor = 0.1f;

  // ---- 观测缩放 ----
  float lin_vel_scale = 2.0f;
  float ang_vel_scale = 0.25f;
  std::array<float, NUM_JOINTS> dof_pos_scale{};
  std::array<float, NUM_JOINTS> dof_vel_scale{};
  std::array<float, NUM_JOINTS> action_scale{};

  // ---- 机器人名称 & 模型路径 ----
  std::string robot_name = "mybot";
  std::string urdf_relpath = "robot/mybot/urdf/mybot.urdf";
  std::string mujoco_xml_relpath = "robot/mybot/xml/mybot.xml";
  std::string isaac_xml_relpath = "robot/mybot/xml/mybot.xml";

  // ---- 控制 & 安全参数 ----
  float cmd_deadband = 0.05f;
  float control_dt = 0.02f;
  float standup_duration = 2.0f;

  float cmd_vx_min = -1.0f;
  float cmd_vx_max = 1.0f;
  float cmd_vy_min = -1.0f;
  float cmd_vy_max = 1.0f;
  float cmd_yaw_min = -3.14f;
  float cmd_yaw_max = 3.14f;

  float cmd_vx_step = 0.1f;
  float cmd_vy_step = 0.1f;
  float cmd_yaw_step = 0.3f;

  float clip_obs = 100.0f;
  float clip_actions = 100.0f;
  float hip_reduction = 1.0f;
  std::array<int, 4> hip_indices = {0, 3, 6, 9};

  // ---- UDP Teleop (手机遥控) ----
  bool teleop_udp_enable = false;
  int teleop_udp_port = 9870;

  // ---- 便捷函数 ----
  float motor_kp(int joint_idx) const {
    const float r = joint_transmission_ratio[joint_idx];
    return kp_joint[joint_idx] / (r * r);
  }

  float motor_kd(int joint_idx) const {
    const float r = joint_transmission_ratio[joint_idx];
    return kd_joint[joint_idx] / (r * r);
  }
};

RobotRuntimeConfig default_robot_runtime_config();
RobotRuntimeConfig load_robot_runtime_config(const std::string &yaml_file);

} // namespace deploy

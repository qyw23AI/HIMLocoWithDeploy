/**
 * @file robot_runtime_config.cpp
 * @brief YAML-based robot configuration loading.
 *
 * All robot parameters are loaded from YAML at runtime.
 * No hardcoded robot-specific constants.
 */

#include "robot_runtime_config.h"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace deploy {
namespace {

template <typename T, size_t N>
std::array<T, N> parse_required_array(const YAML::Node &root,
                                      const std::string &key) {
  const YAML::Node n = root[key];
  if (!n || !n.IsSequence() || n.size() != N) {
    throw std::runtime_error("Invalid or missing array key: " + key);
  }
  std::array<T, N> out{};
  for (size_t i = 0; i < N; ++i) {
    out[i] = n[i].as<T>();
  }
  return out;
}

template <typename T, size_t N>
std::array<T, N> parse_scalar_or_array(const YAML::Node &root,
                                       const std::string &key,
                                       T default_value) {
  const YAML::Node n = root[key];
  std::array<T, N> out{};
  out.fill(default_value);
  if (!n) {
    return out;
  }
  if (n.IsSequence()) {
    if (n.size() != N) {
      throw std::runtime_error("Invalid array length for key: " + key);
    }
    for (size_t i = 0; i < N; ++i) {
      out[i] = n[i].as<T>();
    }
    return out;
  }
  const T v = n.as<T>();
  out.fill(v);
  return out;
}

void validate_positive_array(const std::array<float, NUM_JOINTS> &vals,
                             const std::string &name) {
  for (size_t i = 0; i < vals.size(); ++i) {
    if (vals[i] <= 0.0f) {
      throw std::runtime_error(name + " has non-positive value at index " +
                               std::to_string(i));
    }
  }
}

} // namespace

// ============================================================
//  默认配置 (兜底值，正常运行时应始终由 YAML 覆盖)
// ============================================================

RobotRuntimeConfig default_robot_runtime_config() {
  RobotRuntimeConfig cfg;

  // ---- 默认关节名 (FR/FL/RR/RL 顺序, 与旧 robot_config.h 兼容) ----
  const char *default_names[NUM_JOINTS] = {
      "FR_hip_joint",   "FR_thigh_joint",  "FR_calf_joint",
      "FL_hip_joint",   "FL_thigh_joint",  "FL_calf_joint",
      "RR_hip_joint",   "RR_thigh_joint",  "RR_calf_joint",
      "RL_hip_joint",   "RL_thigh_joint",  "RL_calf_joint",
  };

  constexpr float GEAR_RATIO = 6.33f;

  for (int i = 0; i < NUM_JOINTS; ++i) {
    cfg.kp_joint[i] = 80.0f;
    cfg.kd_joint[i] = 2.0f;
    cfg.fixed_kp[i] = cfg.kp_joint[i];
    cfg.fixed_kd[i] = 3.0f;
    cfg.torque_limits[i] = 23.5f;
    cfg.joint_names[i] = default_names[i];
    cfg.joint_controller_names[i] = std::string(default_names[i]) + "_controller";

    // 默认: identity mapping (driver idx == policy idx)
    cfg.joint_mapping[i] = i;
    cfg.motor_is_reversed[i] = false;

    cfg.dof_pos_scale[i] = 1.0f;
    cfg.dof_vel_scale[i] = 0.08f;
    cfg.action_scale[i] = 0.15f;
    cfg.joint_transmission_ratio[i] = GEAR_RATIO;

    // 默认 motor_map: motor_id = i+1, port 按前后分
    cfg.motor_map[i].motor_id = i + 1;
    cfg.motor_map[i].port_idx = (i < 6) ? 0 : 1;
    cfg.motor_map[i].is_reversed = false;
  }

  // 默认关节角度
  cfg.default_dof_pos = {
      -0.35f, 0.99f, -2.57f,  // FR
       0.35f, 0.99f, -2.57f,  // FL
      -0.35f, 0.99f, -2.57f,  // RR
       0.35f, 0.99f, -2.57f,  // RL
  };
  cfg.standup_target_pos = {
      -0.1f, 0.8f, -1.5f,  // FR
       0.1f, 0.8f, -1.5f,  // FL
      -0.1f, 1.0f, -1.5f,  // RR
       0.1f, 1.0f, -1.5f,  // RL
  };
  cfg.policy_dof_pos = cfg.standup_target_pos;

  cfg.joint_pos_lower.fill(-2.50f);
  cfg.joint_pos_upper.fill(2.50f);

  // 注意: 小腿的额外 7/3 减速比已在 YAML 的 joint_transmission_ratio 中体现
  // (例如 14.77 = 6.33 × 7/3)，这里的默认值仅作为兜底，
  // 正常运行时会被 load_robot_runtime_config() 从 YAML 完全覆盖。

  cfg.robot_name = "mybot";
  cfg.urdf_relpath = "robot/mybot/urdf/mybot.urdf";
  cfg.mujoco_xml_relpath = "robot/mybot/xml/mybot.xml";
  cfg.isaac_xml_relpath = "robot/mybot/xml/mybot.xml";

  return cfg;
}

// ============================================================
//  从 YAML 加载配置
// ============================================================

RobotRuntimeConfig load_robot_runtime_config(const std::string &yaml_file) {
  if (yaml_file.empty()) {
    throw std::runtime_error("robot_config_file is required");
  }

  YAML::Node root = YAML::LoadFile(yaml_file);
  RobotRuntimeConfig cfg = default_robot_runtime_config();

  cfg.num_of_dofs = root["num_of_dofs"].as<int>();
  if (cfg.num_of_dofs != NUM_JOINTS) {
    throw std::runtime_error("num_of_dofs must be 12 for this build");
  }

  cfg.dt = root["dt"].as<float>();
  cfg.decimation = root["decimation"].as<int>();

  cfg.fixed_kp = parse_required_array<float, NUM_JOINTS>(root, "fixed_kp");
  cfg.fixed_kd = parse_required_array<float, NUM_JOINTS>(root, "fixed_kd");
  cfg.torque_limits =
      parse_required_array<float, NUM_JOINTS>(root, "torque_limits");

  cfg.default_dof_pos =
      parse_required_array<float, NUM_JOINTS>(root, "default_dof_pos");
  cfg.standup_target_pos =
      parse_required_array<float, NUM_JOINTS>(root, "standup_target_pos");
  cfg.policy_dof_pos =
      parse_required_array<float, NUM_JOINTS>(root, "policy_dof_pos");

  cfg.joint_pos_lower =
      parse_required_array<float, NUM_JOINTS>(root, "joint_pos_lower");
  cfg.joint_pos_upper =
      parse_required_array<float, NUM_JOINTS>(root, "joint_pos_upper");

  // ---- 关节名称 & 控制器名称 ----
  const YAML::Node names = root["joint_names"];
  const YAML::Node ctrl_names = root["joint_controller_names"];
  if (!names || !names.IsSequence() || names.size() != NUM_JOINTS) {
    throw std::runtime_error("joint_names must be an array of size 12");
  }
  if (!ctrl_names || !ctrl_names.IsSequence() ||
      ctrl_names.size() != NUM_JOINTS) {
    throw std::runtime_error(
        "joint_controller_names must be an array of size 12");
  }
  for (int i = 0; i < NUM_JOINTS; ++i) {
    cfg.joint_names[i] = names[i].as<std::string>();
    cfg.joint_controller_names[i] = ctrl_names[i].as<std::string>();
  }

  // ---- joint_mapping (可选; 不提供则默认 identity: 1,2,...,12) ----
  //   joint_mapping[policy_dof_idx] = 对应电机 CAN ID (1-12)
  //   实机部署时必须提供, 仿真时可省略 (关节名已按 policy 顺序)
  const YAML::Node mapping = root["joint_mapping"];
  if (mapping && mapping.IsSequence() && mapping.size() == NUM_JOINTS) {
    for (int i = 0; i < NUM_JOINTS; ++i) {
      cfg.joint_mapping[i] = mapping[i].as<int>();
    }
  } else if (!mapping) {
    // identity: policy DOF i → motor CAN ID (i+1)
    for (int i = 0; i < NUM_JOINTS; ++i) {
      cfg.joint_mapping[i] = i + 1;
    }
  } else {
    throw std::runtime_error("joint_mapping must be an array of size 12");
  }

  // ---- motor_is_reversed (可选; 不提供则默认全 false) ----
  //   motor_is_reversed[policy_dof_idx] = 该 DOF 对应电机是否反转
  const YAML::Node reversed = root["motor_is_reversed"];
  if (reversed && reversed.IsSequence() && reversed.size() == NUM_JOINTS) {
    for (int i = 0; i < NUM_JOINTS; ++i) {
      cfg.motor_is_reversed[i] = reversed[i].as<bool>();
    }
  } else if (!reversed) {
    cfg.motor_is_reversed.fill(false);
  } else {
    throw std::runtime_error("motor_is_reversed must be an array of size 12");
  }

  // ---- 轮子索引 (可选) ----
  const YAML::Node wheels = root["wheel_indices"];
  cfg.wheel_indices.clear();
  if (wheels && wheels.IsSequence()) {
    for (const auto &w : wheels) {
      cfg.wheel_indices.push_back(w.as<int>());
    }
  }

  // ---- 传动比 ----
  cfg.joint_transmission_ratio =
      parse_required_array<float, NUM_JOINTS>(root, "joint_transmission_ratio");
  validate_positive_array(cfg.joint_transmission_ratio,
                          "joint_transmission_ratio");

  // ---- 路径 & 设备 ----
  cfg.device = root["device"].as<std::string>();
  cfg.port0 = root["port0"].as<std::string>();
  cfg.port1 = root["port1"].as<std::string>();
  cfg.imu_topic = root["imu_topic"].as<std::string>();
  if (root["imu_yaw_correction_deg"]) {
    cfg.imu_yaw_correction_deg = root["imu_yaw_correction_deg"].as<float>();
  }
  cfg.robot_name = root["robot_name"].as<std::string>();
  cfg.urdf_relpath = root["urdf_relpath"].as<std::string>();
  cfg.mujoco_xml_relpath = root["mujoco_xml_relpath"].as<std::string>();
  cfg.isaac_xml_relpath = root["isaac_xml_relpath"].as<std::string>();
  if (root["policy_path"]) {
    cfg.policy_path = root["policy_path"].as<std::string>();
  }

  // ---- PD 控制参数 ----
    cfg.kp_joint = parse_scalar_or_array<float, NUM_JOINTS>(
      root, "kp_joint", cfg.kp_joint[0]);
    cfg.kd_joint = parse_scalar_or_array<float, NUM_JOINTS>(
      root, "kd_joint", cfg.kd_joint[0]);
  cfg.kd_damp_motor = root["kd_damp_motor"].as<float>();

  // ---- 观测缩放 ----
  cfg.lin_vel_scale = root["lin_vel_scale"].as<float>();
  cfg.ang_vel_scale = root["ang_vel_scale"].as<float>();
  cfg.dof_pos_scale =
      parse_scalar_or_array<float, NUM_JOINTS>(root, "dof_pos_scale", 1.0f);
  cfg.dof_vel_scale =
      parse_scalar_or_array<float, NUM_JOINTS>(root, "dof_vel_scale", 0.08f);
  cfg.action_scale =
      parse_scalar_or_array<float, NUM_JOINTS>(root, "action_scale", 0.15f);

  // ---- 控制 & 安全参数 ----
  cfg.cmd_deadband = root["cmd_deadband"].as<float>();
  cfg.control_dt = root["control_dt"].as<float>();
  cfg.standup_duration = root["standup_duration"].as<float>();

  if (root["cmd_vx_min"]) cfg.cmd_vx_min = root["cmd_vx_min"].as<float>();
  if (root["cmd_vx_max"]) cfg.cmd_vx_max = root["cmd_vx_max"].as<float>();
  if (root["cmd_vy_min"]) cfg.cmd_vy_min = root["cmd_vy_min"].as<float>();
  if (root["cmd_vy_max"]) cfg.cmd_vy_max = root["cmd_vy_max"].as<float>();
  if (root["cmd_yaw_min"]) cfg.cmd_yaw_min = root["cmd_yaw_min"].as<float>();
  if (root["cmd_yaw_max"]) cfg.cmd_yaw_max = root["cmd_yaw_max"].as<float>();

  if (root["cmd_vx_step"]) cfg.cmd_vx_step = root["cmd_vx_step"].as<float>();
  if (root["cmd_vy_step"]) cfg.cmd_vy_step = root["cmd_vy_step"].as<float>();
  if (root["cmd_yaw_step"]) {
    cfg.cmd_yaw_step = root["cmd_yaw_step"].as<float>();
  }

  if (root["clip_obs"]) cfg.clip_obs = root["clip_obs"].as<float>();
  if (root["clip_actions"]) {
    cfg.clip_actions = root["clip_actions"].as<float>();
  }
  if (root["hip_reduction"]) {
    cfg.hip_reduction = root["hip_reduction"].as<float>();
  }

  // ---- Hip indices (可选, 默认 {0, 3, 6, 9}) ----
  if (root["hip_indices"]) {
    const YAML::Node hi = root["hip_indices"];
    if (hi.IsSequence() && hi.size() == 4) {
      for (int i = 0; i < 4; ++i) {
        cfg.hip_indices[i] = hi[i].as<int>();
      }
    }
  }

  // ============================================================
  //  构建 motor_map — 按 **policy DOF 顺序** 索引
  //
  //  motor_map[p] 描述 policy DOF p 对应的物理电机:
  //    .motor_id    = joint_mapping[p]  (CAN ID, 1-12)
  //    .port_idx    = 由 motor_id 推导 (1-6→port0, 7-12→port1)
  //                   或由 YAML motor_port_idx 显式覆盖
  //    .is_reversed = motor_is_reversed[p]
  //
  //  这样所有数组统一按 policy 顺序访问, 无需双重间接.
  // ============================================================

  // 可选: YAML 显式指定 motor_port_idx (按 policy 顺序)
  const YAML::Node motor_ports = root["motor_port_idx"];
  std::array<int, NUM_JOINTS> explicit_port{};
  explicit_port.fill(-1);
  if (motor_ports && motor_ports.IsSequence() &&
      motor_ports.size() == NUM_JOINTS) {
    for (int i = 0; i < NUM_JOINTS; ++i) {
      explicit_port[i] = motor_ports[i].as<int>();
      if (explicit_port[i] != 0 && explicit_port[i] != 1) {
        throw std::runtime_error("motor_port_idx must be 0 or 1 at index " +
                                 std::to_string(i) + ": " +
                                 std::to_string(explicit_port[i]));
      }
    }
  }

  // 验证 joint_mapping 是双射 (每个 motor_id 只出现一次)
  std::array<bool, NUM_JOINTS + 1> seen{};  // seen[motor_id], 1-based
  seen.fill(false);

  for (int p = 0; p < NUM_JOINTS; ++p) {
    const int motor_id = cfg.joint_mapping[p];
    if (motor_id < 1 || motor_id > NUM_JOINTS) {
      throw std::runtime_error(
          "joint_mapping[" + std::to_string(p) + "] = " +
          std::to_string(motor_id) + " out of range [1, 12]");
    }
    if (seen[motor_id]) {
      throw std::runtime_error(
          "joint_mapping is not one-to-one, repeated motor_id " +
          std::to_string(motor_id));
    }
    seen[motor_id] = true;

    cfg.motor_map[p].motor_id = motor_id;
    cfg.motor_map[p].port_idx =
        (explicit_port[p] != -1) ? explicit_port[p]
                                 : ((motor_id >= 7) ? 1 : 0);
    cfg.motor_map[p].is_reversed = cfg.motor_is_reversed[p];
  }

  // ---- UDP Teleop (可选; 默认 disabled) ----
  if (root["teleop_udp_enable"]) {
    cfg.teleop_udp_enable = root["teleop_udp_enable"].as<bool>();
  }
  if (cfg.teleop_udp_enable) {
    if (!root["teleop_udp_port"]) {
      throw std::runtime_error(
          "teleop_udp_enable is true but teleop_udp_port is missing in YAML");
    }
    cfg.teleop_udp_port = root["teleop_udp_port"].as<int>();
    if (cfg.teleop_udp_port <= 0 || cfg.teleop_udp_port > 65535) {
      throw std::runtime_error(
          "teleop_udp_port must be in range [1, 65535], got: " +
          std::to_string(cfg.teleop_udp_port));
    }
  } else if (root["teleop_udp_port"]) {
    cfg.teleop_udp_port = root["teleop_udp_port"].as<int>();
  }

  return cfg;
}

} // namespace deploy

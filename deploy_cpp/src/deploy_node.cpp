/**
 * @file deploy_node.cpp
 * @brief Main deployment node for Mybot quadruped robot.
 *
 * ROS2 node + 50Hz control loop integrating:
 *   - IMU subscriber (ROS2 /fast_livo2/state6)
 *   - Motor driver (Unitree GO-M8010-6 SDK)
 *   - HIM policy inference (LibTorch JIT)
 *   - Keyboard controller (velocity commands + state switching)
 *   - State machine (Idle / StandUp / RL / JointDamping)
 *
 * Usage:
 *   ros2 run deploy_cpp deploy_node --ros-args \
 *     -p policy_path:=/path/to/policy.pt \
 *     -p device:=cuda:0 \
 *     -p port0:=/dev/ttyUSB0 \
 *     -p port1:=/dev/ttyUSB1 \
 *     -p debug_no_motor:=false
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "imu_subscriber.h"
#include "keyboard_controller.h"
#include "motor_driver.h"
#include "policy_runner.h"
#include "udp_controller.h"

#include "robot_runtime_config.h"
#include "robot_visualizer.h"
#include "state_machine.h"

namespace deploy {

static constexpr const char *BANNER = R"(
╔══════════════════════════════════════════════════╗
║         Mybot Quadruped Deployment Node (C++)    ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  State switching:                                ║
║    0 : Idle          (零力矩, 电机不发力)           ║
║    1 : StandUp       (缓慢站起)                   ║
║    2 : RL            (策略控制)                   ║
║    3 : JointDamping  (关节阻尼缓冲)                ║
║    5 : SingleStepRL  (单步策略调试)               ║
║    6 : JointSweep    (单关节方向调试)               ║
║                                                  ║
║  Velocity controls (RL state only):              ║
║    W/S : vx  ↑↓     Q/E : vy  ↑↓                 ║
║    A/D : yaw ↑↓     R   : reset vel to zero      ║
║                                                  ║
║  Debug controls (JointSweep / SingleStepRL):     ║
║    Enter : Confirm & execute current step        ║
║    J/K   : Next/Prev joint   (JointSweep)        ║
║    +/-   : Adjust offset     (JointSweep)        ║
║                                                  ║
║  Safety:                                         ║
║    Space : Emergency stop → Idle                 ║
║    Esc   : Exit program                          ║
║                                                  ║
╚══════════════════════════════════════════════════╝
)";

// ====================================================================== //
//  FakeMotorDriver — software-only simulation for offline testing        //
// ====================================================================== //

class FakeMotorDriver {
public:
  explicit FakeMotorDriver(const RobotRuntimeConfig &config) : config_(config) {
    dof_pos_ = config_.default_dof_pos;
    dof_vel_.fill(0.0f);
  }

  void send_commands(const std::array<float, NUM_JOINTS> &target_dof_pos,
                     const std::array<float, NUM_JOINTS> &kp,
                     const std::array<float, NUM_JOINTS> &kd) {
    const float dt = config_.control_dt;
    for (int i = 0; i < NUM_JOINTS; ++i) {
      float pos_err = target_dof_pos[i] - dof_pos_[i];
      float acc = std::clamp(kp[i] * pos_err - kd[i] * dof_vel_[i], -20.0f, 20.0f);
      dof_vel_[i] = (dof_vel_[i] + acc * dt) * 0.98f;
      dof_pos_[i] = std::clamp(dof_pos_[i] + dof_vel_[i] * dt,
                               config_.joint_pos_lower[i],
                               config_.joint_pos_upper[i]);
    }
  }

  void send_damping(float kd) {
    const float dt = config_.control_dt;
    float damping = std::clamp(kd * 0.6f, 0.0f, 0.9f);
    for (int i = 0; i < NUM_JOINTS; ++i) {
      dof_vel_[i] *= (1.0f - damping);
      dof_pos_[i] = std::clamp(dof_pos_[i] + dof_vel_[i] * dt,
                               config_.joint_pos_lower[i],
                               config_.joint_pos_upper[i]);
    }
  }

  void set_zero_torque() {
    const float dt = config_.control_dt;
    for (int i = 0; i < NUM_JOINTS; ++i) {
      dof_vel_[i] *= 0.98f;
      dof_pos_[i] = std::clamp(dof_pos_[i] + dof_vel_[i] * dt,
                               config_.joint_pos_lower[i],
                               config_.joint_pos_upper[i]);
    }
  }

  bool check_errors() const { return false; }

  const std::array<float, NUM_JOINTS> &dof_pos() const { return dof_pos_; }
  const std::array<float, NUM_JOINTS> &dof_vel() const { return dof_vel_; }

private:
  RobotRuntimeConfig config_;
  std::array<float, NUM_JOINTS> dof_pos_;
  std::array<float, NUM_JOINTS> dof_vel_;
};

// ====================================================================== //
//  MujocoMotorDriver — ROS2 topic bridge to MuJoCo Python simulation     //
// ====================================================================== //

class MujocoMotorDriver {
public:
  MujocoMotorDriver(rclcpp::Node *node, const RobotRuntimeConfig &config)
      : config_(config) {
    dof_pos_ = config_.default_dof_pos;
    dof_vel_.fill(0.0f);

    // Low-latency QoS for real-time control: BEST_EFFORT, depth=1, VOLATILE
    rclcpp::QoS qos(rclcpp::KeepLast(1));
    qos.best_effort();
    qos.durability_volatile();

    // Publisher: send target joint positions to MuJoCo sim
    pub_ = node->create_publisher<std_msgs::msg::Float32MultiArray>(
        "/mujoco/joint_cmd", qos);

    // Subscriber: receive joint states from MuJoCo sim
    sub_ = node->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/mujoco/joint_state", qos,
        [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
          if (msg->data.size() >= NUM_JOINTS * 2) {
            for (int i = 0; i < NUM_JOINTS; ++i) {
              dof_pos_[i] = msg->data[i];
              dof_vel_[i] = msg->data[NUM_JOINTS + i];
            }
            msg_count_.fetch_add(1, std::memory_order_relaxed);
          }
        });

    RCLCPP_INFO(
        node->get_logger(),
        "MujocoMotorDriver: pub=/mujoco/joint_cmd, sub=/mujoco/joint_state");
  }

  void send_commands(const std::array<float, NUM_JOINTS> &target_dof_pos,
                     const std::array<float, NUM_JOINTS> &kp,
                     const std::array<float, NUM_JOINTS> &kd) {
    std_msgs::msg::Float32MultiArray msg;
    msg.data.resize(NUM_JOINTS * 3);
    for (int i = 0; i < NUM_JOINTS; ++i) {
      msg.data[i] = target_dof_pos[i];
      msg.data[NUM_JOINTS + i] = kp[i];
      msg.data[2 * NUM_JOINTS + i] = kd[i];
    }
    pub_->publish(msg);
  }

  void send_damping(float kd) {
    // Send current position as target with only kd
    std_msgs::msg::Float32MultiArray msg;
    msg.data.resize(NUM_JOINTS * 3);
    for (int i = 0; i < NUM_JOINTS; ++i) {
      msg.data[i] = dof_pos_[i];
      msg.data[NUM_JOINTS + i] = 0.0f;
      msg.data[2 * NUM_JOINTS + i] = kd;
    }
    pub_->publish(msg);
  }

  void set_zero_torque() {
    std_msgs::msg::Float32MultiArray msg;
    msg.data.resize(NUM_JOINTS * 3);
    for (int i = 0; i < NUM_JOINTS; ++i) {
      msg.data[i] = dof_pos_[i];
      msg.data[NUM_JOINTS + i] = 0.0f;
      msg.data[2 * NUM_JOINTS + i] = 0.0f;
    }
    pub_->publish(msg);
  }

  bool check_errors() const { return false; }

  uint64_t msg_count() const { return msg_count_.load(std::memory_order_relaxed); }

  const std::array<float, NUM_JOINTS> &dof_pos() const { return dof_pos_; }
  const std::array<float, NUM_JOINTS> &dof_vel() const { return dof_vel_; }

private:
  RobotRuntimeConfig config_;
  std::array<float, NUM_JOINTS> dof_pos_;
  std::array<float, NUM_JOINTS> dof_vel_;
  std::atomic<uint64_t> msg_count_{0};
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
};

// ====================================================================== //
//  DeployNode — main ROS2 deployment controller                          //
// ====================================================================== //

class DeployNode : public rclcpp::Node {
public:
  DeployNode() : Node("deploy_node") {
    // Declare parameters
    this->declare_parameter<std::string>("robot_config_file", "");
    this->declare_parameter<bool>("debug_no_motor", false);
    this->declare_parameter<bool>("sim_mode", false);
    this->declare_parameter<bool>("sim_pingpong_mode", false);
  }

  void initialize() {
    auto config_file = this->get_parameter("robot_config_file").as_string();
    if (config_file.empty()) {
      throw std::runtime_error("robot_config_file is required");
    }
    config_ = load_robot_runtime_config(config_file);
    build_dof_permutation();

    debug_no_motor_ = this->get_parameter("debug_no_motor").as_bool();
    sim_mode_ = this->get_parameter("sim_mode").as_bool();
    sim_pingpong_mode_ = this->get_parameter("sim_pingpong_mode").as_bool();
    const auto standup_target_driver =
      reorder_policy_to_driver(config_.standup_target_pos);
    last_safe_target_ = standup_target_driver;
    sweep_last_sent_ = standup_target_driver;

    std::cout << BANNER << std::endl;

    // ---- State subscriber (ang_vel + projected_gravity from fast_livo2) ----
    imu_ = std::make_shared<IMUSubscriber>(config_.imu_topic,
                         config_.imu_yaw_correction_deg);
    RCLCPP_INFO(this->get_logger(), "IMU subscriber initialized");

    // ---- Motor driver ----
    if (sim_mode_) {
      mujoco_motor_ = std::make_unique<MujocoMotorDriver>(this, config_);
      RCLCPP_INFO(this->get_logger(),
                  "SIM: Using MuJoCo motor driver (ROS2 topics)");
      if (sim_pingpong_mode_) {
        RCLCPP_INFO(this->get_logger(),
                    "SIM ping-pong mode enabled: state-triggered control (no wall-clock control sleep)");
      }
    } else if (!debug_no_motor_) {
      motor_ = std::make_unique<MotorDriver>(config_, config_.port0, config_.port1);
      RCLCPP_INFO(this->get_logger(), "Motor driver initialized");
    } else {
      fake_motor_ = std::make_unique<FakeMotorDriver>(config_);
      RCLCPP_INFO(this->get_logger(), "DEBUG: Using fake motor driver");
    }

    // ---- Policy runner ----
    policy_ = std::make_unique<PolicyRunner>(config_.policy_path, config_);
    RCLCPP_INFO(this->get_logger(), "Policy runner initialized");

    // ---- Keyboard controller ----
    keyboard_ = std::make_unique<KeyboardController>(config_);
    RCLCPP_INFO(this->get_logger(), "Keyboard controller initialized");

    // ---- UDP teleop controller (optional) ----
    if (config_.teleop_udp_enable) {
      udp_ctrl_ = std::make_unique<UdpController>(config_.teleop_udp_port);
      RCLCPP_INFO(this->get_logger(),
                  "UDP teleop controller listening on port %d",
                  config_.teleop_udp_port);
    }

    // ---- State machine ----
    sm_ = std::make_unique<StateMachine>(config_);
    RCLCPP_INFO(this->get_logger(), "State machine initialized (IDLE)");

    // ---- RViz visualizer ----
    visualizer_ = std::make_unique<RobotVisualizer>(this, config_.joint_names);
    RCLCPP_INFO(this->get_logger(), "RViz visualizer initialized");

    RCLCPP_INFO(this->get_logger(), "All modules ready. Press 1 to stand up.");
  }

  void run() {
    rclcpp::Rate rate(1.0 / config_.control_dt);
    uint64_t loop_count = 0;
    uint64_t last_joint_msg_count = 0;
    uint64_t last_imu_msg_count = 0;
    auto last_print_time = std::chrono::steady_clock::now();

    while (rclcpp::ok() && !keyboard_->is_exit()) {
      // 1. Spin / wait for fresh state
      if (sim_mode_ && sim_pingpong_mode_ && mujoco_motor_) {
        while (rclcpp::ok() && !keyboard_->is_exit()) {
          rclcpp::spin_some(imu_);
          rclcpp::spin_some(this->shared_from_this());

          const uint64_t joint_count = mujoco_motor_->msg_count();
          const uint64_t imu_count = imu_->msg_count();
          if (joint_count > last_joint_msg_count && imu_count > last_imu_msg_count) {
            last_joint_msg_count = joint_count;
            last_imu_msg_count = imu_count;
            break;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      } else {
        rclcpp::spin_some(imu_);
        if (sim_mode_) {
          rclcpp::spin_some(this->shared_from_this());
        }
      }

      // 2a. Handle UDP teleop commands (if enabled)
      if (udp_ctrl_ && udp_ctrl_->has_data()) {
        auto cmd = udp_ctrl_->get_latest();
        if (cmd.e_stop) {
          // E-STOP: force IDLE + zero velocities
          handle_state_request({RobotState::IDLE, true});
          udp_vx_ = 0.0f; udp_vy_ = 0.0f; udp_yaw_ = 0.0f;
        } else {
          // Mode change
          auto target = static_cast<RobotState>(cmd.mode);
          if (target != sm_->state()) {
            handle_state_request({target, false});
          }
          // Update velocity commands (ratio × max)
          udp_vx_  = cmd.vx  * config_.cmd_vx_max;
          udp_vy_  = cmd.vy  * config_.cmd_vy_max;
          udp_yaw_ = cmd.yaw * config_.cmd_yaw_max;
        }
      }

      // 2b. Handle keyboard state transitions
      auto req = keyboard_->consume_state_request();
      if (req.has_value()) {
        handle_state_request(req.value());
      }

      // 3. Execute current state logic
      switch (sm_->state()) {
      case RobotState::IDLE:
        handle_idle();
        break;
      case RobotState::STAND_UP:
        handle_standup();
        break;
      case RobotState::RL:
        handle_rl();
        break;
      case RobotState::JOINT_DAMPING:
        handle_joint_damping();
        break;
      case RobotState::JOINT_SWEEP:
        handle_joint_sweep();
        break;
      case RobotState::SINGLE_STEP_RL:
        handle_single_step_rl();
        break;
      }

      // 3.5. Publish joint states for RViz visualization
      if (visualizer_) {
        visualizer_->publish_joint_states(get_dof_pos(), get_dof_vel());
      }

      // 4. Periodic status printing (every 1s)
      loop_count++;
      auto now = std::chrono::steady_clock::now();
      float elapsed =
          std::chrono::duration<float>(now - last_print_time).count();
      if (elapsed >= 1.0f) {
        print_status(loop_count);
        last_print_time = now;
      }

      if (!(sim_mode_ && sim_pingpong_mode_)) {
        rate.sleep();
      }
    }

    shutdown();
  }

private:
  void build_dof_permutation() {
    // motor_map 已按 policy DOF 顺序索引 (在 load_robot_runtime_config 中构建),
    // 因此 policy_to_driver_idx_ 和 driver_to_policy_idx_ 都是 identity mapping.
    // MotorDriver / MujocoMotorDriver / FakeMotorDriver 的 dof_pos_/dof_vel_
    // 也按 policy 顺序存储, 无需重排序.
    for (int i = 0; i < NUM_JOINTS; ++i) {
      policy_to_driver_idx_[i] = i;
      driver_to_policy_idx_[i] = i;
    }
  }

  std::array<float, NUM_JOINTS>
  reorder_driver_to_policy(const std::array<float, NUM_JOINTS> &driver_vals) const {
    // Identity: motor_map 已按 policy 顺序, 无需重排
    return driver_vals;
  }

  std::array<float, NUM_JOINTS>
  reorder_policy_to_driver(const std::array<float, NUM_JOINTS> &policy_vals) const {
    // Identity: motor_map 已按 policy 顺序, 无需重排
    return policy_vals;
  }

  // ------------------------------------------------------------------ //
  //  State request handler                                              //
  // ------------------------------------------------------------------ //

  void handle_state_request(const StateRequest &req) {
    if (req.emergency) {
      RCLCPP_WARN(this->get_logger(), "EMERGENCY STOP!");
    }

    RobotState old_state = sm_->state();
    bool accepted = sm_->request_transition(req.target, req.emergency);

    if (accepted && old_state != req.target) {
      // On entering RL or SingleStepRL, reset policy history
      if (req.target == RobotState::RL ||
          req.target == RobotState::SINGLE_STEP_RL) {
        policy_->reset();
        RCLCPP_INFO(this->get_logger(), "Policy history reset");
        if (req.target == RobotState::SINGLE_STEP_RL) {
          single_step_count_ = 0;
          single_step_pending_ = false;
          last_safe_target_ =
              reorder_policy_to_driver(config_.standup_target_pos);
        }
      }
      // On entering JointSweep, reset sweep state
      if (req.target == RobotState::JOINT_SWEEP) {
        keyboard_->reset_sweep();
        RCLCPP_INFO(this->get_logger(),
                    "JointSweep: J/K=joint  +/-=offset  Enter=send");
      }
    }
  }

  // ------------------------------------------------------------------ //
  //  Per-state handlers                                                //
  // ------------------------------------------------------------------ //

  void handle_idle() {
    if (motor_) {
      motor_->set_zero_torque();
    } else if (mujoco_motor_) {
      mujoco_motor_->set_zero_torque();
    } else if (fake_motor_) {
      fake_motor_->set_zero_torque();
    }
  }

  void handle_standup() {
    // StateMachine standup target is defined in policy DOF order.
    // Convert feedback to policy order first, then convert target back to
    // driver order before sending motors.
    const auto &cur_pos_driver = get_dof_pos();
    auto cur_pos_policy = reorder_driver_to_policy(cur_pos_driver);
    auto target_policy = sm_->get_standup_target(cur_pos_policy);
    auto target_driver = reorder_policy_to_driver(target_policy);

    if (motor_) {
      motor_->send_commands(target_driver, config_.kp_joint, config_.kd_joint);
    } else if (mujoco_motor_) {
      mujoco_motor_->send_commands(target_driver, config_.kp_joint,
                                   config_.kd_joint);
    } else if (fake_motor_) {
      fake_motor_->send_commands(target_driver, config_.kp_joint,
                                 config_.kd_joint);
    }

    if (sm_->standup_complete()) {
      static bool printed = false;
      if (!printed) {
        RCLCPP_INFO(this->get_logger(),
                    "Standup complete! Press 2 to enter RL mode.");
        printed = true;
      }
    }
  }

  void handle_rl() {
    // Gather sensor data
    auto ang_vel = imu_->get_ang_vel();
    auto projected_gravity = imu_->get_projected_gravity();
    const auto &dof_pos_driver = get_dof_pos();
    const auto &dof_vel_driver = get_dof_vel();
    auto dof_pos_policy = reorder_driver_to_policy(dof_pos_driver);
    auto dof_vel_policy = reorder_driver_to_policy(dof_vel_driver);
    auto commands = (udp_ctrl_ && udp_ctrl_->has_data())
        ? std::array<float, 3>{udp_vx_, udp_vy_, udp_yaw_}
        : keyboard_->get_commands();

    // Run policy inference
    std::array<float, NUM_JOINTS> target_dof_pos_policy;
    std::array<float, NUM_ACTIONS> actions;
    policy_->step(commands, ang_vel, projected_gravity, dof_pos_policy,
            dof_vel_policy, target_dof_pos_policy, actions);
    auto target_dof_pos_driver = reorder_policy_to_driver(target_dof_pos_policy);
    // test
    //     target_dof_pos = {
    //     -0.39f, 0.77f, -1.50f,
    //     0.1f, 0.75f, -1.81f,
    //     0.1f, 0.93f, -1.54f,
    //     0.1f, 0.71f, -0.98f
    // };

    // Send to motors
    if (motor_) {
      motor_->send_commands(target_dof_pos_driver, config_.kp_joint,
                            config_.kd_joint);
    } else if (mujoco_motor_) {
      mujoco_motor_->send_commands(target_dof_pos_driver, config_.kp_joint,
                                   config_.kd_joint);
    } else if (fake_motor_) {
      fake_motor_->send_commands(target_dof_pos_driver, config_.kp_joint,
                                 config_.kd_joint);
    }
  }

  void handle_joint_damping() {
    if (motor_) {
      motor_->send_damping(config_.kd_damp_motor);
    } else if (mujoco_motor_) {
      mujoco_motor_->send_damping(config_.kd_damp_motor);
    } else if (fake_motor_) {
      fake_motor_->send_damping(config_.kd_damp_motor);
    }
  }

  // ------------------------------------------------------------------ //
  //  JointSweep — single joint direction verification                  //
  // ------------------------------------------------------------------ //

  void handle_joint_sweep() {
    int idx = keyboard_->get_sweep_joint_idx();
    float offset = keyboard_->get_sweep_offset();

    // Build target in policy order: standup pose + offset on selected joint.
    std::array<float, NUM_JOINTS> target_policy = config_.standup_target_pos;
    target_policy[idx] = std::clamp(config_.standup_target_pos[idx] + offset,
                                    config_.joint_pos_lower[idx],
                                    config_.joint_pos_upper[idx]);
    auto target_driver = reorder_policy_to_driver(target_policy);
    const int driver_idx = policy_to_driver_idx_[idx];

    // Print status (throttled to ~2 Hz to avoid flood)
    static auto last_print = std::chrono::steady_clock::now();
    auto now_t = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now_t - last_print).count();
    if (dt >= 0.5f) {
      last_print = now_t;
      std::cout << "\n\033[1;36m[JOINT_SWEEP]\033[0m  Selected: \033[1m"
                << config_.joint_names[idx] << "\033[0m (DOF " << idx
                << ")  motor_id=" << config_.motor_map[driver_idx].motor_id
                << "  reversed="
                << (config_.motor_map[driver_idx].is_reversed ? "YES" : "NO")
                << std::endl;
      std::cout << "  STANDUP_POS = " << std::fixed << std::setprecision(3)
                << config_.standup_target_pos[idx] << "  offset = " << std::showpos
                << offset << std::noshowpos << "  target = "
                << target_policy[idx]
                << std::endl;
      const auto &cur_pos_driver = get_dof_pos();
      const auto &cur_vel_driver = get_dof_vel();
      auto cur_pos_policy = reorder_driver_to_policy(cur_pos_driver);
      auto cur_vel_policy = reorder_driver_to_policy(cur_vel_driver);
      std::cout << "  Current pos = " << cur_pos_policy[idx]
                << "  Current vel = " << cur_vel_policy[idx] << std::endl;
      std::cout << "  Controls: J/K=prev/next joint  +/-=adjust(0.05)  "
                << "Enter=\033[1;32mSEND\033[0m  Space=STOP" << std::endl;
    }

    // Check Enter confirmation
    if (keyboard_->consume_step_confirm()) {
      std::cout << "\n\033[1;32m>>> SENDING\033[0m target[" << idx
                << "] = " << target_policy[idx] << " to "
                << config_.joint_names[idx]
                << std::endl;
      send_to_motors(target_driver, config_.kp_joint, config_.kd_joint);
      sweep_last_sent_ = target_driver;
      sweep_has_sent_ = true;
    } else {
      // Hold previous sent position (or standup pose if nothing sent yet)
      if (sweep_has_sent_) {
        send_to_motors(sweep_last_sent_, config_.kp_joint, config_.kd_joint);
      } else {
        send_to_motors(reorder_policy_to_driver(config_.standup_target_pos),
                       config_.kp_joint,
                       config_.kd_joint);
      }
    }
  }

  // ------------------------------------------------------------------ //
  //  SingleStepRL — step RL policy with confirm before execution       //
  // ------------------------------------------------------------------ //

  void handle_single_step_rl() {
    // If a step result is pending confirmation, keep sending last safe target
    if (single_step_pending_) {
      send_to_motors(last_safe_target_, config_.kp_joint, config_.kd_joint);

      // Check for Enter confirmation
      if (keyboard_->consume_step_confirm()) {
        std::cout << "\n\033[1;32m>>> EXECUTING step " << single_step_count_
                  << "\033[0m" << std::endl;
        send_to_motors(pending_target_, config_.kp_joint, config_.kd_joint);
        last_safe_target_ = pending_target_;
        single_step_pending_ = false;
      }
      return;
    }

    // ---- Compute next step (but don't send yet) ----
    auto ang_vel = imu_->get_ang_vel();
    auto projected_gravity = imu_->get_projected_gravity();
    const auto &dof_pos_driver = get_dof_pos();
    const auto &dof_vel_driver = get_dof_vel();
    auto dof_pos_policy = reorder_driver_to_policy(dof_pos_driver);
    auto dof_vel_policy = reorder_driver_to_policy(dof_vel_driver);
    auto commands = (udp_ctrl_ && udp_ctrl_->has_data())
        ? std::array<float, 3>{udp_vx_, udp_vy_, udp_yaw_}
        : keyboard_->get_commands();

    std::array<float, NUM_JOINTS> target_dof_pos_policy;
    std::array<float, NUM_ACTIONS> actions;
    policy_->step(commands, ang_vel, projected_gravity, dof_pos_policy,
            dof_vel_policy, target_dof_pos_policy, actions);
    auto target_dof_pos_driver = reorder_policy_to_driver(target_dof_pos_policy);

    single_step_count_++;

    // ---- Print complete parameter table ----
    std::cout << "\n\033[1;33m"
              << "============ SINGLE STEP RL [step " << single_step_count_
              << "] ============\033[0m" << std::endl;
    std::cout << "Commands: vx=" << std::fixed << std::setprecision(2)
              << commands[0] << "  vy=" << commands[1]
              << "  yaw=" << commands[2] << std::endl;
    std::cout << "IMU: ang_vel=[" << std::setprecision(3) << ang_vel[0] << ", "
              << ang_vel[1] << ", " << ang_vel[2] << "]  proj_grav=["
              << projected_gravity[0] << ", " << projected_gravity[1] << ", "
              << projected_gravity[2] << "]" << std::endl;
    std::cout << std::endl;

    // Table header
    std::cout << std::left << std::setw(20) << "Joint" << std::right
              << std::setw(10) << "dof_pos" << std::setw(10) << "dof_vel"
              << std::setw(10) << "action" << std::setw(10) << "target"
              << std::setw(10) << "reversed" << std::setw(10) << "motor_id"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (int i = 0; i < NUM_JOINTS; ++i) {
      const int driver_idx = policy_to_driver_idx_[i];
      std::cout << std::left << std::setw(20) << config_.joint_names[i] << std::right
                << std::fixed << std::setprecision(3) << std::setw(10)
                << dof_pos_policy[i] << std::setw(10) << dof_vel_policy[i]
                << std::setw(10) << actions[i] << std::setw(10)
                << target_dof_pos_policy[i]
                << std::setw(10)
                << (config_.motor_map[driver_idx].is_reversed ? "YES" : "NO")
                << std::setw(10) << config_.motor_map[driver_idx].motor_id
                << std::endl;
    }

    std::cout << std::string(80, '-') << std::endl;
        std::cout << "PD gains (joint, DOF0 example): kp=" << std::setprecision(4)
          << config_.kp_joint[0] << "  kd=" << config_.kd_joint[0]
          << std::endl;
    std::cout << "\n\033[1;33mPress ENTER to execute, Space to STOP\033[0m"
              << std::endl;

    // Store pending and wait
    pending_target_ = target_dof_pos_driver;
    single_step_pending_ = true;

    // Keep holding current position
    send_to_motors(last_safe_target_, config_.kp_joint, config_.kd_joint);
  }

  // ------------------------------------------------------------------ //
  //  Motor send helper (abstracts real/fake/mujoco)                    //
  // ------------------------------------------------------------------ //

  void send_to_motors(const std::array<float, NUM_JOINTS> &target,
                      const std::array<float, NUM_JOINTS> &kp,
                      const std::array<float, NUM_JOINTS> &kd) {
    if (motor_) {
      motor_->send_commands(target, kp, kd);
    } else if (mujoco_motor_) {
      mujoco_motor_->send_commands(target, kp, kd);
    } else if (fake_motor_) {
      fake_motor_->send_commands(target, kp, kd);
    }
  }

  // ------------------------------------------------------------------ //
  //  Sensor accessors (abstract over real/fake motor)                  //
  // ------------------------------------------------------------------ //

  const std::array<float, NUM_JOINTS> &get_dof_pos() const {
    if (motor_)
      return motor_->dof_pos();
    if (mujoco_motor_)
      return mujoco_motor_->dof_pos();
    return fake_motor_->dof_pos();
  }

  const std::array<float, NUM_JOINTS> &get_dof_vel() const {
    if (motor_)
      return motor_->dof_vel();
    if (mujoco_motor_)
      return mujoco_motor_->dof_vel();
    return fake_motor_->dof_vel();
  }

  // ------------------------------------------------------------------ //
  //  Status printing                                                    //
  // ------------------------------------------------------------------ //

  void print_status(uint64_t loop_count) const {
    auto commands = keyboard_->get_commands();

    // Get IMU data for debugging
    auto ang_vel = imu_->get_ang_vel();
    auto proj_grav = imu_->get_projected_gravity();

    std::cout
        << "\r[" << robot_state_name(sm_->state()) << "] "
        << "loop=" << loop_count
        << " imu=" << (imu_->is_ready() ? "OK" : "WAIT") << " cmd=["
        << std::fixed << std::setprecision(2) << commands[0] << ","
        << commands[1] << "," << commands[2]
        << "]"
        //   << " pos[0:3]=[" << pos[0] << "," << pos[1] << "," << pos[2] << "]"
        << " ang_vel=[" << ang_vel[0] << "," << ang_vel[1] << "," << ang_vel[2]
        << "]"
        << " grav=[" << proj_grav[0] << "," << proj_grav[1] << ","
        << proj_grav[2] << "]"
        << "       " << std::flush;
  }

  // ------------------------------------------------------------------ //
  //  Shutdown                                                           //
  // ------------------------------------------------------------------ //

  void shutdown() {
    RCLCPP_INFO(this->get_logger(), "Shutting down...");

    // Force idle (zero torque)
    if (motor_) {
      motor_->set_zero_torque();
      motor_->set_zero_torque(); // Send twice for safety
    }

    if (keyboard_) {
      keyboard_->cleanup();
    }

    std::cout << std::endl << "[Deploy] Shutdown complete." << std::endl;
  }

  // ------------------------------------------------------------------ //
  //  Members                                                            //
  // ------------------------------------------------------------------ //

  bool debug_no_motor_ = false;
  bool sim_mode_ = false;
  bool sim_pingpong_mode_ = false;
  RobotRuntimeConfig config_ = default_robot_runtime_config();
  std::array<int, NUM_JOINTS> policy_to_driver_idx_{};
  std::array<int, NUM_JOINTS> driver_to_policy_idx_{};

  std::shared_ptr<IMUSubscriber> imu_;
  std::unique_ptr<MotorDriver> motor_;
  std::unique_ptr<FakeMotorDriver> fake_motor_;
  std::unique_ptr<MujocoMotorDriver> mujoco_motor_;
  std::unique_ptr<PolicyRunner> policy_;
  std::unique_ptr<KeyboardController> keyboard_;
  std::unique_ptr<UdpController> udp_ctrl_;
  std::unique_ptr<StateMachine> sm_;
  std::unique_ptr<RobotVisualizer> visualizer_;

  // ---- UDP teleop velocity cache ----
  float udp_vx_ = 0.0f;
  float udp_vy_ = 0.0f;
  float udp_yaw_ = 0.0f;

  // ---- JointSweep state ----
  std::array<float, NUM_JOINTS> sweep_last_sent_{};
  bool sweep_has_sent_ = false;

  // ---- SingleStepRL state ----
  uint64_t single_step_count_ = 0;
  bool single_step_pending_ = false;
  std::array<float, NUM_JOINTS> pending_target_{};
  std::array<float, NUM_JOINTS> last_safe_target_{};
};

} // namespace deploy

// ====================================================================== //
//  Signal handling                                                        //
// ====================================================================== //

static std::atomic<bool> g_shutdown{false};

void signal_handler(int /*sig*/) {
  g_shutdown.store(true);
  rclcpp::shutdown();
}

// ====================================================================== //
//  Main                                                                   //
// ====================================================================== //

int main(int argc, char *argv[]) {
  // Register signal handlers
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  // Initialize ROS2
  rclcpp::init(argc, argv);

  auto node = std::make_shared<deploy::DeployNode>();

  try {
    node->initialize();
    node->run();
  } catch (const std::exception &e) {
    RCLCPP_ERROR(node->get_logger(), "Fatal error: %s", e.what());
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}

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
#include "robot_config.h"
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
  FakeMotorDriver() {
    dof_pos_.fill(0.0f);
    dof_vel_.fill(0.0f);
  }

  void send_commands(const std::array<float, NUM_JOINTS> &target_dof_pos,
                     float kp = KP_MOTOR, float kd = KD_MOTOR) {
    constexpr float dt = CONTROL_DT;
    for (int i = 0; i < NUM_JOINTS; ++i) {
      float pos_err = target_dof_pos[i] - dof_pos_[i];
      float acc = std::clamp(kp * pos_err - kd * dof_vel_[i], -20.0f, 20.0f);
      dof_vel_[i] = (dof_vel_[i] + acc * dt) * 0.98f;
      dof_pos_[i] = std::clamp(dof_pos_[i] + dof_vel_[i] * dt,
                               JOINT_POS_LOWER[i], JOINT_POS_UPPER[i]);
    }
  }

  void send_damping(float kd = KD_DAMP_MOTOR) {
    constexpr float dt = CONTROL_DT;
    float damping = std::clamp(kd * 0.6f, 0.0f, 0.9f);
    for (int i = 0; i < NUM_JOINTS; ++i) {
      dof_vel_[i] *= (1.0f - damping);
      dof_pos_[i] = std::clamp(dof_pos_[i] + dof_vel_[i] * dt,
                               JOINT_POS_LOWER[i], JOINT_POS_UPPER[i]);
    }
  }

  void set_zero_torque() {
    constexpr float dt = CONTROL_DT;
    for (int i = 0; i < NUM_JOINTS; ++i) {
      dof_vel_[i] *= 0.98f;
      dof_pos_[i] = std::clamp(dof_pos_[i] + dof_vel_[i] * dt,
                               JOINT_POS_LOWER[i], JOINT_POS_UPPER[i]);
    }
  }

  bool check_errors() const { return false; }

  const std::array<float, NUM_JOINTS> &dof_pos() const { return dof_pos_; }
  const std::array<float, NUM_JOINTS> &dof_vel() const { return dof_vel_; }

private:
  std::array<float, NUM_JOINTS> dof_pos_;
  std::array<float, NUM_JOINTS> dof_vel_;
};

// ====================================================================== //
//  MujocoMotorDriver — ROS2 topic bridge to MuJoCo Python simulation     //
// ====================================================================== //

class MujocoMotorDriver {
public:
  MujocoMotorDriver(rclcpp::Node *node) {
    dof_pos_ = DEFAULT_DOF_POS;
    dof_vel_.fill(0.0f);

    // Publisher: send target joint positions to MuJoCo sim
    pub_ = node->create_publisher<std_msgs::msg::Float32MultiArray>(
        "/mujoco/joint_cmd", 10);

    // Subscriber: receive joint states from MuJoCo sim
    sub_ = node->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/mujoco/joint_state", 10,
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
                     float kp = KP_MOTOR, float kd = KD_MOTOR) {
    std_msgs::msg::Float32MultiArray msg;
    msg.data.resize(NUM_JOINTS + 2);
    for (int i = 0; i < NUM_JOINTS; ++i) {
      msg.data[i] = target_dof_pos[i];
    }
    // Send joint-side PD gains (convert back from motor-side)
    msg.data[NUM_JOINTS] = kp * GEAR_RATIO * GEAR_RATIO;
    msg.data[NUM_JOINTS + 1] = kd * GEAR_RATIO * GEAR_RATIO;
    pub_->publish(msg);
  }

  void send_damping(float kd = KD_DAMP_MOTOR) {
    // Send current position as target with only kd
    std_msgs::msg::Float32MultiArray msg;
    msg.data.resize(NUM_JOINTS + 2);
    for (int i = 0; i < NUM_JOINTS; ++i) {
      msg.data[i] = dof_pos_[i];
    }
    msg.data[NUM_JOINTS] = 0.0f; // kp = 0
    msg.data[NUM_JOINTS + 1] = kd * GEAR_RATIO * GEAR_RATIO;
    pub_->publish(msg);
  }

  void set_zero_torque() {
    std_msgs::msg::Float32MultiArray msg;
    msg.data.resize(NUM_JOINTS + 2);
    for (int i = 0; i < NUM_JOINTS; ++i) {
      msg.data[i] = dof_pos_[i];
    }
    msg.data[NUM_JOINTS] = 0.0f;
    msg.data[NUM_JOINTS + 1] = 0.0f;
    pub_->publish(msg);
  }

  bool check_errors() const { return false; }

  uint64_t msg_count() const { return msg_count_.load(std::memory_order_relaxed); }

  const std::array<float, NUM_JOINTS> &dof_pos() const { return dof_pos_; }
  const std::array<float, NUM_JOINTS> &dof_vel() const { return dof_vel_; }

private:
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
    this->declare_parameter<std::string>("policy_path", "policy/policy.pt");
    this->declare_parameter<std::string>("device", "cuda:0");
    this->declare_parameter<std::string>("port0", "/dev/ttyUSB0");
    this->declare_parameter<std::string>("port1", "/dev/ttyUSB1");
    this->declare_parameter<bool>("debug_no_motor", false);
    this->declare_parameter<bool>("sim_mode", false);
    this->declare_parameter<bool>("sim_pingpong_mode", false);
    this->declare_parameter<std::string>("imu_topic", "/fast_livo2/state6");
  }

  void initialize() {
    auto policy_path = this->get_parameter("policy_path").as_string();
    auto device = this->get_parameter("device").as_string();
    auto port0 = this->get_parameter("port0").as_string();
    auto port1 = this->get_parameter("port1").as_string();
    debug_no_motor_ = this->get_parameter("debug_no_motor").as_bool();
    sim_mode_ = this->get_parameter("sim_mode").as_bool();
    sim_pingpong_mode_ = this->get_parameter("sim_pingpong_mode").as_bool();
    auto imu_topic = this->get_parameter("imu_topic").as_string();

    std::cout << BANNER << std::endl;

    // ---- State subscriber (ang_vel + projected_gravity from fast_livo2) ----
    imu_ = std::make_shared<IMUSubscriber>(imu_topic);
    RCLCPP_INFO(this->get_logger(), "IMU subscriber initialized");

    // ---- Motor driver ----
    if (sim_mode_) {
      mujoco_motor_ = std::make_unique<MujocoMotorDriver>(this);
      RCLCPP_INFO(this->get_logger(),
                  "SIM: Using MuJoCo motor driver (ROS2 topics)");
      if (sim_pingpong_mode_) {
        RCLCPP_INFO(this->get_logger(),
                    "SIM ping-pong mode enabled: state-triggered control (no wall-clock control sleep)");
      }
    } else if (!debug_no_motor_) {
      motor_ = std::make_unique<MotorDriver>(port0, port1);
      RCLCPP_INFO(this->get_logger(), "Motor driver initialized");
    } else {
      fake_motor_ = std::make_unique<FakeMotorDriver>();
      RCLCPP_INFO(this->get_logger(), "DEBUG: Using fake motor driver");
    }

    // ---- Policy runner ----
    policy_ = std::make_unique<PolicyRunner>(policy_path, device);
    RCLCPP_INFO(this->get_logger(), "Policy runner initialized");

    // ---- Keyboard controller ----
    keyboard_ = std::make_unique<KeyboardController>();
    RCLCPP_INFO(this->get_logger(), "Keyboard controller initialized");

    // ---- State machine ----
    sm_ = std::make_unique<StateMachine>();
    RCLCPP_INFO(this->get_logger(), "State machine initialized (IDLE)");

    // ---- RViz visualizer ----
    visualizer_ = std::make_unique<RobotVisualizer>(this);
    RCLCPP_INFO(this->get_logger(), "RViz visualizer initialized");

    RCLCPP_INFO(this->get_logger(), "All modules ready. Press 1 to stand up.");
  }

  void run() {
    rclcpp::Rate rate(1.0 / CONTROL_DT); // 50 Hz
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

      // 2. Handle keyboard state transitions
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
          last_safe_target_ = STANDUP_TARGET_POS;
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
    // Get current joint positions
    const auto &cur_pos = get_dof_pos();
    auto target = sm_->get_standup_target(cur_pos);

    if (motor_) {
      motor_->send_commands(target, KP_MOTOR, KD_MOTOR);
    } else if (mujoco_motor_) {
      mujoco_motor_->send_commands(target, KP_MOTOR, KD_MOTOR);
    } else if (fake_motor_) {
      fake_motor_->send_commands(target, KP_MOTOR, KD_MOTOR);
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
    const auto &dof_pos = get_dof_pos();
    const auto &dof_vel = get_dof_vel();
    auto commands = keyboard_->get_commands();

    // Run policy inference
    std::array<float, NUM_JOINTS> target_dof_pos;
    std::array<float, NUM_ACTIONS> actions;
    policy_->step(commands, ang_vel, projected_gravity, dof_pos, dof_vel,
                  target_dof_pos, actions);
    // test
    //     target_dof_pos = {
    //     -0.39f, 0.77f, -1.50f,
    //     0.1f, 0.75f, -1.81f,
    //     0.1f, 0.93f, -1.54f,
    //     0.1f, 0.71f, -0.98f
    // };

    // Send to motors
    if (motor_) {
      motor_->send_commands(target_dof_pos, KP_MOTOR, KD_MOTOR);
    } else if (mujoco_motor_) {
      mujoco_motor_->send_commands(target_dof_pos, KP_MOTOR, KD_MOTOR);
    } else if (fake_motor_) {
      fake_motor_->send_commands(target_dof_pos, KP_MOTOR, KD_MOTOR);
    }
  }

  void handle_joint_damping() {
    if (motor_) {
      motor_->send_damping(KD_DAMP_MOTOR);
    } else if (mujoco_motor_) {
      mujoco_motor_->send_damping(KD_DAMP_MOTOR);
    } else if (fake_motor_) {
      fake_motor_->send_damping(KD_DAMP_MOTOR);
    }
  }

  // ------------------------------------------------------------------ //
  //  JointSweep — single joint direction verification                  //
  // ------------------------------------------------------------------ //

  void handle_joint_sweep() {
    int idx = keyboard_->get_sweep_joint_idx();
    float offset = keyboard_->get_sweep_offset();

    // Build target: standup pose + offset on selected joint only
    std::array<float, NUM_JOINTS> target = STANDUP_TARGET_POS;
    target[idx] = std::clamp(STANDUP_TARGET_POS[idx] + offset,
                             JOINT_POS_LOWER[idx], JOINT_POS_UPPER[idx]);

    // Print status (throttled to ~2 Hz to avoid flood)
    static auto last_print = std::chrono::steady_clock::now();
    auto now_t = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now_t - last_print).count();
    if (dt >= 0.5f) {
      last_print = now_t;
      std::cout << "\n\033[1;36m[JOINT_SWEEP]\033[0m  Selected: \033[1m"
                << JOINT_NAMES[idx] << "\033[0m (DOF " << idx
                << ")  motor_id=" << MOTOR_MAP[idx].motor_id
                << "  reversed=" << (MOTOR_MAP[idx].is_reversed ? "YES" : "NO")
                << std::endl;
      std::cout << "  STANDUP_POS = " << std::fixed << std::setprecision(3)
                << STANDUP_TARGET_POS[idx] << "  offset = " << std::showpos
                << offset << std::noshowpos << "  target = " << target[idx]
                << std::endl;
      const auto &cur_pos = get_dof_pos();
      const auto &cur_vel = get_dof_vel();
      std::cout << "  Current pos = " << cur_pos[idx]
                << "  Current vel = " << cur_vel[idx] << std::endl;
      std::cout << "  Controls: J/K=prev/next joint  +/-=adjust(0.05)  "
                << "Enter=\033[1;32mSEND\033[0m  Space=STOP" << std::endl;
    }

    // Check Enter confirmation
    if (keyboard_->consume_step_confirm()) {
      std::cout << "\n\033[1;32m>>> SENDING\033[0m target[" << idx
                << "] = " << target[idx] << " to " << JOINT_NAMES[idx]
                << std::endl;
      send_to_motors(target, KP_MOTOR, KD_MOTOR);
      sweep_last_sent_ = target;
      sweep_has_sent_ = true;
    } else {
      // Hold previous sent position (or standup pose if nothing sent yet)
      if (sweep_has_sent_) {
        send_to_motors(sweep_last_sent_, KP_MOTOR, KD_MOTOR);
      } else {
        send_to_motors(STANDUP_TARGET_POS, KP_MOTOR, KD_MOTOR);
      }
    }
  }

  // ------------------------------------------------------------------ //
  //  SingleStepRL — step RL policy with confirm before execution       //
  // ------------------------------------------------------------------ //

  void handle_single_step_rl() {
    // If a step result is pending confirmation, keep sending last safe target
    if (single_step_pending_) {
      send_to_motors(last_safe_target_, KP_MOTOR, KD_MOTOR);

      // Check for Enter confirmation
      if (keyboard_->consume_step_confirm()) {
        std::cout << "\n\033[1;32m>>> EXECUTING step " << single_step_count_
                  << "\033[0m" << std::endl;
        send_to_motors(pending_target_, KP_MOTOR, KD_MOTOR);
        last_safe_target_ = pending_target_;
        single_step_pending_ = false;
      }
      return;
    }

    // ---- Compute next step (but don't send yet) ----
    auto ang_vel = imu_->get_ang_vel();
    auto projected_gravity = imu_->get_projected_gravity();
    const auto &dof_pos = get_dof_pos();
    const auto &dof_vel = get_dof_vel();
    auto commands = keyboard_->get_commands();

    std::array<float, NUM_JOINTS> target_dof_pos;
    std::array<float, NUM_ACTIONS> actions;
    policy_->step(commands, ang_vel, projected_gravity, dof_pos, dof_vel,
                  target_dof_pos, actions);

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
      std::cout << std::left << std::setw(20) << JOINT_NAMES[i] << std::right
                << std::fixed << std::setprecision(3) << std::setw(10)
                << dof_pos[i] << std::setw(10) << dof_vel[i] << std::setw(10)
                << actions[i] << std::setw(10) << target_dof_pos[i]
                << std::setw(10) << (MOTOR_MAP[i].is_reversed ? "YES" : "NO")
                << std::setw(10) << MOTOR_MAP[i].motor_id << std::endl;
    }

    std::cout << std::string(80, '-') << std::endl;
    std::cout << "PD gains: kp_motor=" << std::setprecision(4) << KP_MOTOR
              << "  kd_motor=" << KD_MOTOR << "  (joint: kp=" << KP_JOINT
              << " kd=" << KD_JOINT << ")" << std::endl;
    std::cout << "\n\033[1;33mPress ENTER to execute, Space to STOP\033[0m"
              << std::endl;

    // Store pending and wait
    pending_target_ = target_dof_pos;
    single_step_pending_ = true;

    // Keep holding current position
    send_to_motors(last_safe_target_, KP_MOTOR, KD_MOTOR);
  }

  // ------------------------------------------------------------------ //
  //  Motor send helper (abstracts real/fake/mujoco)                    //
  // ------------------------------------------------------------------ //

  void send_to_motors(const std::array<float, NUM_JOINTS> &target, float kp,
                      float kd) {
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

  std::shared_ptr<IMUSubscriber> imu_;
  std::unique_ptr<MotorDriver> motor_;
  std::unique_ptr<FakeMotorDriver> fake_motor_;
  std::unique_ptr<MujocoMotorDriver> mujoco_motor_;
  std::unique_ptr<PolicyRunner> policy_;
  std::unique_ptr<KeyboardController> keyboard_;
  std::unique_ptr<StateMachine> sm_;
  std::unique_ptr<RobotVisualizer> visualizer_;

  // ---- JointSweep state ----
  std::array<float, NUM_JOINTS> sweep_last_sent_ = STANDUP_TARGET_POS;
  bool sweep_has_sent_ = false;

  // ---- SingleStepRL state ----
  uint64_t single_step_count_ = 0;
  bool single_step_pending_ = false;
  std::array<float, NUM_JOINTS> pending_target_{};
  std::array<float, NUM_JOINTS> last_safe_target_ = STANDUP_TARGET_POS;
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

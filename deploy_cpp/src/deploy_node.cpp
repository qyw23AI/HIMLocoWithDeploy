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

#include "imu_subscriber.h"
#include "keyboard_controller.h"
#include "motor_driver.h"
#include "policy_runner.h"
#include "robot_config.h"
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
║                                                  ║
║  Velocity controls (RL state only):              ║
║    W/S : vx  ↑↓     Q/E : vy  ↑↓                 ║
║    A/D : yaw ↑↓     R   : reset vel to zero      ║
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
    this->declare_parameter<std::string>("imu_topic", "/fast_livo2/state6");
  }

  void initialize() {
    auto policy_path = this->get_parameter("policy_path").as_string();
    auto device = this->get_parameter("device").as_string();
    auto port0 = this->get_parameter("port0").as_string();
    auto port1 = this->get_parameter("port1").as_string();
    debug_no_motor_ = this->get_parameter("debug_no_motor").as_bool();
    auto imu_topic = this->get_parameter("imu_topic").as_string();

    std::cout << BANNER << std::endl;

    // ---- State subscriber (ang_vel + projected_gravity from fast_livo2) ----
    imu_ = std::make_shared<IMUSubscriber>(imu_topic);
    RCLCPP_INFO(this->get_logger(), "IMU subscriber initialized");

    // ---- Motor driver ----
    if (!debug_no_motor_) {
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

    RCLCPP_INFO(this->get_logger(), "All modules ready. Press 1 to stand up.");
  }

  void run() {
    rclcpp::Rate rate(1.0 / CONTROL_DT); // 50 Hz
    uint64_t loop_count = 0;
    auto last_print_time = std::chrono::steady_clock::now();

    while (rclcpp::ok() && !keyboard_->is_exit()) {
      // 1. Spin IMU subscriber
      rclcpp::spin_some(imu_);

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

      rate.sleep();
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
      // On entering RL, reset policy history
      if (req.target == RobotState::RL) {
        policy_->reset();
        RCLCPP_INFO(this->get_logger(), "Policy history reset");
      }
    }
  }

  // ------------------------------------------------------------------ //
  //  Per-state handlers                                                //
  // ------------------------------------------------------------------ //

  void handle_idle() {
    if (motor_) {
      motor_->set_zero_torque();
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

    // Send to motors
    if (motor_) {
      motor_->send_commands(target_dof_pos, KP_MOTOR, KD_MOTOR);
    } else if (fake_motor_) {
      fake_motor_->send_commands(target_dof_pos, KP_MOTOR, KD_MOTOR);
    }
  }

  void handle_joint_damping() {
    if (motor_) {
      motor_->send_damping(KD_DAMP_MOTOR);
    } else if (fake_motor_) {
      fake_motor_->send_damping(KD_DAMP_MOTOR);
    }
  }

  // ------------------------------------------------------------------ //
  //  Sensor accessors (abstract over real/fake motor)                  //
  // ------------------------------------------------------------------ //

  const std::array<float, NUM_JOINTS> &get_dof_pos() const {
    if (motor_)
      return motor_->dof_pos();
    return fake_motor_->dof_pos();
  }

  const std::array<float, NUM_JOINTS> &get_dof_vel() const {
    if (motor_)
      return motor_->dof_vel();
    return fake_motor_->dof_vel();
  }

  // ------------------------------------------------------------------ //
  //  Status printing                                                    //
  // ------------------------------------------------------------------ //

  void print_status(uint64_t loop_count) const {
    auto commands = keyboard_->get_commands();
    const auto &pos = get_dof_pos();
    const auto &vel = get_dof_vel();

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

  std::shared_ptr<IMUSubscriber> imu_;
  std::unique_ptr<MotorDriver> motor_;
  std::unique_ptr<FakeMotorDriver> fake_motor_;
  std::unique_ptr<PolicyRunner> policy_;
  std::unique_ptr<KeyboardController> keyboard_;
  std::unique_ptr<StateMachine> sm_;
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

/**
 * @file motor_debug_node.cpp
 * @brief Motor debug node for reading joint angles.
 *
 * Sends zero parameters (q=0, dq=0, kp=0, kd=0, tau=0) to all 12 motors
 * and reads back the current joint angles from encoder feedback.
 *
 * This is a standalone debug tool — no IMU, policy, or keyboard needed.
 *
 * Usage:
 *   ros2 run deploy_cpp motor_debug_node --ros-args \
 *     -p port0:=/dev/ttyUSB0 \
 *     -p port1:=/dev/ttyUSB1 \
 *     -p rate_hz:=50.0
 */

#include <atomic>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "motor_driver.h"
#include "robot_runtime_config.h"
#include "robot_visualizer.h"

namespace deploy {

static const char *DEBUG_BANNER = R"(
╔══════════════════════════════════════════════════╗
║       Mybot Motor Debug Node                     ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  发送全零参数，仅读取电机角度                        ║
║  (q=0, dq=0, kp=0, kd=0, tau=0)                 ║
║                                                  ║
║  Ctrl+C 退出                                     ║
║                                                  ║
╚══════════════════════════════════════════════════╝
)";

class MotorDebugNode : public rclcpp::Node {
public:
  MotorDebugNode() : Node("motor_debug_node") {
    this->declare_parameter<std::string>("port0", "/dev/ttyUSB0");
    this->declare_parameter<std::string>("port1", "/dev/ttyUSB1");
    this->declare_parameter<double>("rate_hz", 50.0);
    this->declare_parameter<std::string>("robot_config_file", "");
  }

  void initialize() {
    auto port0 = this->get_parameter("port0").as_string();
    auto port1 = this->get_parameter("port1").as_string();
    auto robot_config_file = this->get_parameter("robot_config_file").as_string();
    rate_hz_ = this->get_parameter("rate_hz").as_double();

    std::cout << DEBUG_BANNER << std::endl;

    RCLCPP_INFO(this->get_logger(),
                "Connecting motors: port0=%s, port1=%s, rate=%.1f Hz",
                port0.c_str(), port1.c_str(), rate_hz_);

    // Load robot configuration
    if (!robot_config_file.empty()) {
      config_ = load_robot_runtime_config(robot_config_file);
      RCLCPP_INFO(this->get_logger(),
                  "Loaded config from %s (robot: %s)",
                  robot_config_file.c_str(), config_.robot_name.c_str());
    } else {
      config_ = default_robot_runtime_config();
      RCLCPP_INFO(this->get_logger(),
                  "Using default config (robot: %s)",
                  config_.robot_name.c_str());
    }

    motor_ = std::make_unique<MotorDriver>(config_, port0, port1);

    // RViz visualizer (publishes /joint_states)
    visualizer_ = std::make_unique<RobotVisualizer>(this, config_.joint_names);

    RCLCPP_INFO(this->get_logger(),
                "Motor driver initialized. Reading angles...");

    // Print header
    std::cout << "\n";
    print_header();
  }

  void run() {
    rclcpp::Rate rate(rate_hz_);

    while (rclcpp::ok()) {
      // Send zero torque command to all motors
      // This sends q=0, dq=0, kp=0, kd=0, tau=0
      // The motor driver still reads back encoder feedback
      motor_->set_zero_torque();

      // Check for motor errors
      if (motor_->check_errors()) {
        RCLCPP_WARN(this->get_logger(), "Motor errors detected!");
      }

      // Publish joint states for RViz
      if (visualizer_) {
        visualizer_->publish_joint_states(motor_->dof_pos(), motor_->dof_vel());
      }

      // Print current joint angles
      print_angles();

      rate.sleep();
    }

    shutdown();
  }

private:
  void print_header() {
    std::cout << std::left << std::setw(18) << "Joint" << std::setw(12)
              << "Angle(rad)" << std::setw(12) << "Angle(deg)" << std::setw(12)
              << "Vel(rad/s)" << std::endl;
    std::cout << std::string(54, '-') << std::endl;
  }

  void print_angles() {
    const auto &pos = motor_->dof_pos();
    const auto &vel = motor_->dof_vel();

    // Move cursor up to overwrite previous output (12 joints + 2 header lines)
    std::cout << "\033[" << (NUM_JOINTS + 2) << "A";

    print_header();

    for (int i = 0; i < NUM_JOINTS; ++i) {
      float deg = pos[i] * 180.0f / 3.14159265f;
      std::cout << std::left << std::setw(18) << config_.joint_names[i] << std::fixed
                << std::setprecision(4) << std::setw(12) << pos[i]
                << std::setw(12) << deg << std::setw(12) << vel[i] << std::endl;
    }
    std::cout << std::flush;
  }

  void shutdown() {
    RCLCPP_INFO(this->get_logger(), "Shutting down motor debug node...");
    if (motor_) {
      motor_->set_zero_torque();
      motor_->set_zero_torque();
    }
    std::cout << "\n[MotorDebug] Shutdown complete." << std::endl;
  }

  double rate_hz_ = 50.0;
  RobotRuntimeConfig config_;
  std::unique_ptr<MotorDriver> motor_;
  std::unique_ptr<RobotVisualizer> visualizer_;
};

} // namespace deploy

// Signal handling
static std::atomic<bool> g_shutdown{false};

void signal_handler(int /*sig*/) {
  g_shutdown.store(true);
  rclcpp::shutdown();
}

int main(int argc, char *argv[]) {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  rclcpp::init(argc, argv);

  auto node = std::make_shared<deploy::MotorDebugNode>();

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

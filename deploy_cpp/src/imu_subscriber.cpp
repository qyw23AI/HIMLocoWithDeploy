/**
 * @file imu_subscriber.cpp
 * @brief ROS2 state subscriber implementation.
 *
 * Subscribes to /fast_livo2/state6 (std_msgs/Float32MultiArray).
 * Data layout: [ang_vel_x, ang_vel_y, ang_vel_z, grav_x, grav_y, grav_z]
 */

#include "imu_subscriber.h"

#include <cmath>
#include <iostream>

namespace deploy {

namespace {
constexpr float kPi = 3.14159265358979323846f;
}

IMUSubscriber::IMUSubscriber(const std::string &topic, float yaw_correction_deg)
    : Node("imu_subscriber") {
  const float yaw_rad = yaw_correction_deg * kPi / 180.0f;
  yaw_cos_ = std::cos(yaw_rad);
  yaw_sin_ = std::sin(yaw_rad);

  // Low-latency QoS for real-time sensor data: BEST_EFFORT, depth=1, VOLATILE
  rclcpp::QoS qos(rclcpp::KeepLast(1));
  qos.best_effort();
  qos.durability_volatile();

  sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
      topic, qos,
      std::bind(&IMUSubscriber::state_callback, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(),
              "Subscribing to state topic: %s (imu_yaw_correction_deg=%.1f)",
              topic.c_str(), yaw_correction_deg);
}

void IMUSubscriber::state_callback(
    const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
  if (msg->data.size() < 6) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "State message has %zu floats, expected 6. Ignoring.",
                         msg->data.size());
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  // Apply yaw correction on XY plane: v_robot = Rz(yaw_correction) * v_sensor.
  const float wx = msg->data[0];
  const float wy = msg->data[1];
  const float gx = msg->data[3];
  const float gy = msg->data[4];

  // Angular velocity [wx, wy, wz]
  ang_vel_[0] = yaw_cos_ * wx - yaw_sin_ * wy;
  ang_vel_[1] = yaw_sin_ * wx + yaw_cos_ * wy;
  ang_vel_[2] = msg->data[2];

  // Projected gravity [gx, gy, gz]
  projected_gravity_[0] = yaw_cos_ * gx - yaw_sin_ * gy;
  projected_gravity_[1] = yaw_sin_ * gx + yaw_cos_ * gy;
  projected_gravity_[2] = msg->data[5];

  received_.store(true);
  msg_count_.fetch_add(1);
}

std::array<float, 3> IMUSubscriber::get_ang_vel() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return ang_vel_;
}

std::array<float, 3> IMUSubscriber::get_projected_gravity() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return projected_gravity_;
}

} // namespace deploy

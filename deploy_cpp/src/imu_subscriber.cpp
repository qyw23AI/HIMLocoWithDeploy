/**
 * @file imu_subscriber.cpp
 * @brief ROS2 state subscriber implementation.
 *
 * Subscribes to /fast_livo2/state6 (std_msgs/Float32MultiArray).
 * Data layout: [ang_vel_x, ang_vel_y, ang_vel_z, grav_x, grav_y, grav_z]
 */

#include "imu_subscriber.h"

#include <iostream>

namespace deploy {

IMUSubscriber::IMUSubscriber(const std::string &topic)
    : Node("imu_subscriber") {
  // Low-latency QoS for real-time sensor data: BEST_EFFORT, depth=1, VOLATILE
  rclcpp::QoS qos(rclcpp::KeepLast(1));
  qos.best_effort();
  qos.durability_volatile();

  sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
      topic, qos,
      std::bind(&IMUSubscriber::state_callback, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(), "Subscribing to state topic: %s",
              topic.c_str());
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

  // Angular velocity [wx, wy, wz]
  ang_vel_[0] = msg->data[0];
  ang_vel_[1] = msg->data[1];
  ang_vel_[2] = msg->data[2];

  // Projected gravity [gx, gy, gz]
  projected_gravity_[0] = msg->data[3];
  projected_gravity_[1] = msg->data[4];
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

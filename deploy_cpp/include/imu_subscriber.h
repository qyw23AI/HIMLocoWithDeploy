/**
 * @file imu_subscriber.h
 * @brief ROS2 subscriber for state data from /fast_livo2/state6 topic.
 *
 * Receives angular velocity and projected gravity vector directly as
 * std_msgs/Float32MultiArray with 6 floats:
 *   data[0..2] = angular velocity  [wx, wy, wz]  (rad/s)
 *   data[3..5] = projected gravity  [gx, gy, gz]
 */
#pragma once

#include <array>
#include <atomic>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

namespace deploy {

class IMUSubscriber : public rclcpp::Node {
public:
  explicit IMUSubscriber(const std::string &topic = "/fast_livo2/state6",
                         int qos_depth = 10);

  /// Get angular velocity [wx, wy, wz] in body frame [rad/s].
  std::array<float, 3> get_ang_vel() const;

  /// Get gravity vector projected into body frame.
  std::array<float, 3> get_projected_gravity() const;

  /// Whether at least one message has been received.
  bool is_ready() const { return received_.load(); }

  /// Number of messages received so far.
  uint64_t msg_count() const { return msg_count_.load(); }

private:
  void state_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);

  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;

  mutable std::mutex mutex_;
  std::array<float, 3> ang_vel_ = {0.0f, 0.0f, 0.0f};
  std::array<float, 3> projected_gravity_ = {0.0f, 0.0f, -1.0f};

  std::atomic<bool> received_{false};
  std::atomic<uint64_t> msg_count_{0};
};

} // namespace deploy

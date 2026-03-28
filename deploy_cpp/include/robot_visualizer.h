/**
 * @file robot_visualizer.h
 * @brief RViz visualization via JointState publishing.
 *
 * Publishes sensor_msgs/JointState on /joint_states topic so that
 * robot_state_publisher can compute TF transforms for RViz display.
 */
#pragma once

#include <array>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "robot_config.h"

namespace deploy {

class RobotVisualizer {
public:
  /**
   * @brief Construct visualizer attached to an existing ROS2 node.
   * @param node        Pointer to the owning node (used to create publisher)
   * @param joint_names Joint name list matching URDF (size NUM_JOINTS)
   */
  explicit RobotVisualizer(rclcpp::Node *node,
                           const std::array<std::string, NUM_JOINTS> &joint_names);

  /**
   * @brief Publish current joint positions as JointState message.
   * @param dof_pos  Joint positions [rad], size NUM_JOINTS (12)
   * @param dof_vel  Joint velocities [rad/s], size NUM_JOINTS (12)
   */
  void publish_joint_states(const std::array<float, NUM_JOINTS> &dof_pos,
                            const std::array<float, NUM_JOINTS> &dof_vel);

private:
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
  std::vector<std::string> joint_names_;
};

} // namespace deploy

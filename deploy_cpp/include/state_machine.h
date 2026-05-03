/**
 * @file state_machine.h
 * @brief State machine for quadruped robot deployment.
 *
 * States:
 *   IDLE            – Zero torque, motors free-spinning
 *   STAND_UP        – Interpolate from current pose to standing pose
 *   RL              – Execute RL policy actions
 *   JOINT_DAMPING   – Passive damping for safe deceleration
 *   RETURN_DEFAULT  – Interpolate from current pose to startup default pose
 *   JOINT_SWEEP     – Manual single-joint sweep for direction verification
 *   SINGLE_STEP_RL  – Single-step RL with confirm before execution
 */
#pragma once

#include <array>
#include <chrono>
#include <string>

#include "robot_runtime_config.h"

namespace deploy {

enum class RobotState {
  IDLE = 0,           ///< Zero torque, motors free-spinning
  STAND_UP = 1,       ///< Interpolate to standing pose
  RL = 2,             ///< Execute RL policy actions
  JOINT_DAMPING = 3,  ///< Passive damping for safe deceleration
  RETURN_DEFAULT = 4, ///< Interpolate to startup default pose
  SINGLE_STEP_RL = 5, ///< Single-step RL with confirm before execution
  JOINT_SWEEP = 6,    ///< Manual single-joint sweep (direction debug)
};

/// Convert RobotState to human-readable string.
inline const char *robot_state_name(RobotState s) {
  switch (s) {
  case RobotState::IDLE:
    return "IDLE";
  case RobotState::STAND_UP:
    return "STAND_UP";
  case RobotState::RL:
    return "RL";
  case RobotState::JOINT_DAMPING:
    return "JOINT_DAMPING";
  case RobotState::RETURN_DEFAULT:
    return "RETURN_DEFAULT";
  case RobotState::JOINT_SWEEP:
    return "JOINT_SWEEP";
  case RobotState::SINGLE_STEP_RL:
    return "SINGLE_STEP_RL";
  default:
    return "UNKNOWN";
  }
}

class StateMachine {
public:
  explicit StateMachine(const RobotRuntimeConfig &config);

  /// Current state.
  RobotState state() const { return state_; }

  /// Whether standup interpolation is complete.
  bool standup_complete() const { return standup_complete_; }

  /// Whether return-default interpolation is complete.
  bool return_default_complete() const { return return_default_complete_; }

  /**
   * @brief Request a state transition.
   * @param target Desired target state
   * @param emergency If true, force transition to IDLE regardless of rules
   * @return true if transition was accepted
   */
  bool request_transition(RobotState target, bool emergency = false);

  /**
   * @brief Compute interpolated target position during STAND_UP.
   *
   * On first call after entering STAND_UP, the current position is
   * captured as the start pose. Then linear interpolation proceeds
   * over standup_duration_ seconds.
   *
   * @param current_dof_pos Current joint positions [rad]
   * @return Interpolated target joint positions [rad]
   */
  std::array<float, NUM_JOINTS>
  get_standup_target(const std::array<float, NUM_JOINTS> &current_dof_pos);

  /**
   * @brief Compute interpolated target position during RETURN_DEFAULT.
   *
   * On first call after entering RETURN_DEFAULT, the current position is
   * captured as the start pose. Then linear interpolation proceeds over the
   * same duration as STAND_UP, toward config.default_dof_pos.
   *
   * @param current_dof_pos Current joint positions [rad]
   * @return Interpolated target joint positions [rad]
   */
  std::array<float, NUM_JOINTS>
  get_return_default_target(
      const std::array<float, NUM_JOINTS> &current_dof_pos);

private:
  void enter_state(RobotState new_state);
  bool is_valid_transition(RobotState from, RobotState to) const;

  RobotState state_ = RobotState::IDLE;
  std::array<float, NUM_JOINTS> standup_target_pos_{};  // Set from config in ctor
  std::array<float, NUM_JOINTS> default_dof_pos_{};      // Set from config in ctor
  float standup_duration_ = 2.0f;                        // Set from config in ctor

  // StandUp interpolation state
  std::chrono::steady_clock::time_point standup_start_time_;
  std::array<float, NUM_JOINTS> standup_start_pos_{};
  bool standup_pos_captured_ = false;
  bool standup_complete_ = false;

  // ReturnDefault interpolation state
  std::chrono::steady_clock::time_point return_default_start_time_;
  std::array<float, NUM_JOINTS> return_default_start_pos_{};
  bool return_default_pos_captured_ = false;
  bool return_default_complete_ = false;
};

} // namespace deploy

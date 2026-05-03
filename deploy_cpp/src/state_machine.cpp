/**
 * @file state_machine.cpp
 * @brief State machine implementation.
 */

#include "state_machine.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace deploy {

StateMachine::StateMachine(const RobotRuntimeConfig &config) {
  state_ = RobotState::IDLE;
  standup_start_pos_.fill(0.0f);
  return_default_start_pos_.fill(0.0f);
  standup_target_pos_ = config.standup_target_pos;
  default_dof_pos_ = config.default_dof_pos;
  standup_duration_ = config.standup_duration;
}

bool StateMachine::is_valid_transition(RobotState from, RobotState to) const {
  // Valid transitions (not including emergency IDLE which is always allowed)
  switch (from) {
  case RobotState::IDLE:
    return to == RobotState::STAND_UP || to == RobotState::RETURN_DEFAULT;

  case RobotState::STAND_UP:
    return to == RobotState::RL || to == RobotState::JOINT_DAMPING ||
           to == RobotState::RETURN_DEFAULT || to == RobotState::JOINT_SWEEP ||
           to == RobotState::SINGLE_STEP_RL;

  case RobotState::RL:
    return to == RobotState::STAND_UP || to == RobotState::JOINT_DAMPING ||
           to == RobotState::RETURN_DEFAULT;

  case RobotState::JOINT_DAMPING:
    return to == RobotState::IDLE || to == RobotState::STAND_UP ||
           to == RobotState::RETURN_DEFAULT;

  case RobotState::RETURN_DEFAULT:
    return to == RobotState::IDLE || to == RobotState::STAND_UP;

  case RobotState::JOINT_SWEEP:
    return to == RobotState::IDLE || to == RobotState::STAND_UP ||
           to == RobotState::JOINT_DAMPING || to == RobotState::RETURN_DEFAULT;

  case RobotState::SINGLE_STEP_RL:
    return to == RobotState::IDLE || to == RobotState::STAND_UP ||
           to == RobotState::JOINT_DAMPING || to == RobotState::RETURN_DEFAULT;

  default:
    return false;
  }
}

bool StateMachine::request_transition(RobotState target, bool emergency) {
  if (target == state_) {
    return true;
  }

  // Emergency stop → always allowed to IDLE
  if (emergency && target == RobotState::IDLE) {
    enter_state(target);
    return true;
  }

  // STAND_UP → RL / debug states requires standup to be complete
  if (state_ == RobotState::STAND_UP &&
      (target == RobotState::RL || target == RobotState::JOINT_SWEEP ||
       target == RobotState::SINGLE_STEP_RL) &&
      !standup_complete_) {
    std::cout << "[StateMachine] Cannot enter " << robot_state_name(target)
              << ": standup not complete yet" << std::endl;
    return false;
  }

  // Check normal transition rules
  if (is_valid_transition(state_, target)) {
    enter_state(target);
    return true;
  }

  std::cout << "[StateMachine] Invalid transition: " << robot_state_name(state_)
            << " -> " << robot_state_name(target) << std::endl;
  return false;
}

void StateMachine::enter_state(RobotState new_state) {
  std::cout << "[StateMachine] " << robot_state_name(state_) << " -> "
            << robot_state_name(new_state) << std::endl;

  state_ = new_state;

  if (new_state == RobotState::STAND_UP) {
    standup_start_time_ = std::chrono::steady_clock::now();
    standup_pos_captured_ = false;
    standup_complete_ = false;
  }

  if (new_state == RobotState::RETURN_DEFAULT) {
    return_default_start_time_ = std::chrono::steady_clock::now();
    return_default_pos_captured_ = false;
    return_default_complete_ = false;
  }
}

std::array<float, NUM_JOINTS> StateMachine::get_standup_target(
    const std::array<float, NUM_JOINTS> &current_dof_pos) {
  // Capture start position on first call
  if (!standup_pos_captured_) {
    standup_start_pos_ = current_dof_pos;
    standup_pos_captured_ = true;
  }

  auto now = std::chrono::steady_clock::now();
  float elapsed =
      std::chrono::duration<float>(now - standup_start_time_).count();
  float alpha = std::clamp(elapsed / standup_duration_, 0.0f, 1.0f);

  if (alpha >= 1.0f) {
    standup_complete_ = true;
  }

  // Linear interpolation: start * (1 - alpha) + target * alpha
  std::array<float, NUM_JOINTS> target;
  for (int i = 0; i < NUM_JOINTS; ++i) {
    target[i] =
      standup_start_pos_[i] * (1.0f - alpha) + standup_target_pos_[i] * alpha;
  }

  return target;
}

std::array<float, NUM_JOINTS> StateMachine::get_return_default_target(
    const std::array<float, NUM_JOINTS> &current_dof_pos) {
  // Capture start position on first call
  if (!return_default_pos_captured_) {
    return_default_start_pos_ = current_dof_pos;
    return_default_pos_captured_ = true;
  }

  auto now = std::chrono::steady_clock::now();
  float elapsed =
      std::chrono::duration<float>(now - return_default_start_time_).count();
  float alpha = std::clamp(elapsed / standup_duration_, 0.0f, 1.0f);

  if (alpha >= 1.0f) {
    return_default_complete_ = true;
  }

  // Linear interpolation: start * (1 - alpha) + default * alpha
  std::array<float, NUM_JOINTS> target;
  for (int i = 0; i < NUM_JOINTS; ++i) {
    target[i] =
        return_default_start_pos_[i] * (1.0f - alpha) + default_dof_pos_[i] * alpha;
  }

  return target;
}

} // namespace deploy

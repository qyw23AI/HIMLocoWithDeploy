/**
 * @file robot_config.h
 * @brief Compile-time dimension constants for quadruped robot deployment.
 *
 * All robot-specific parameters (joint names, motor mapping, PD gains, etc.)
 * are loaded at runtime from YAML via RobotRuntimeConfig.
 * This file only defines fixed array dimensions needed at compile time.
 */
#pragma once

namespace deploy {

// ========================== Compile-time Dimensions ==========================

/// Number of actuated joints (3 per leg × 4 legs)
constexpr int NUM_JOINTS = 12;

/// Number of policy action outputs
constexpr int NUM_ACTIONS = 12;

/// Single-step observation dimension:
///   3 (commands) + 3 (ang_vel) + 3 (gravity) + 12 (dof_pos) + 12 (dof_vel) + 12 (last_actions) = 45
constexpr int NUM_ONE_STEP_OBS = 45;

/// Number of observation history steps
constexpr int HISTORY_LENGTH = 6;

/// Total observation dimension (45 × 6 = 270)
constexpr int NUM_OBS = NUM_ONE_STEP_OBS * HISTORY_LENGTH;

} // namespace deploy

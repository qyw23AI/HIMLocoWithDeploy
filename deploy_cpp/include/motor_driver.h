/**
 * @file motor_driver.h
 * @brief GO-M8010-6 motor driver wrapper for Unitree Actuator SDK.
 *
 * Handles 12 motors across 2 serial ports, with automatic joint-side ↔
 * motor-side conversion and direction reversal.
 *
 * All motor mapping (motor_id, port_idx, is_reversed) and transmission
 * ratios are read from RobotRuntimeConfig (YAML-driven).
 */
#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <string>

#include "robot_runtime_config.h"

// Forward declarations for Unitree SDK types
struct MotorCmd;
struct MotorData;
class SerialPort;

namespace deploy {

class MotorDriver {
public:
  /**
   * @brief Construct motor driver with two serial ports.
   * @param config  Robot runtime configuration (contains motor_map, etc.)
   * @param port0   Device path for serial port 0
   * @param port1   Device path for serial port 1
   */
  explicit MotorDriver(const RobotRuntimeConfig &config,
                       const std::string &port0,
                       const std::string &port1);
  ~MotorDriver();

  // Non-copyable
  MotorDriver(const MotorDriver &) = delete;
  MotorDriver &operator=(const MotorDriver &) = delete;

  // ------------------------------------------------------------------ //
  //  High-level commands                                                //
  // ------------------------------------------------------------------ //

  /**
   * @brief Send position commands to all 12 motors with PD control.
   * @param target_dof_pos Target joint positions [rad], size 12
    * @param kp Joint-side position gains [N·m/rad], size 12
    * @param kd Joint-side velocity gains [N·m·s/rad], size 12
   *
   * Internally updates cached joint states from motor feedback.
   */
  void send_commands(const std::array<float, NUM_JOINTS> &target_dof_pos,
                const std::array<float, NUM_JOINTS> &kp,
                const std::array<float, NUM_JOINTS> &kd);

  /**
   * @brief Send damping-only commands (kp=0, tau=0).
   * @param kd Motor-side damping gain
   */
  void send_damping(float kd);

  /**
   * @brief Emergency stop: set all gains and torques to zero.
   */
  void set_zero_torque();

  /**
   * @brief Check if any motor has non-zero error codes.
   * @return true if errors found
   */
  bool check_errors() const;

  /**
   * @brief Calibrate encoder offsets at startup.
   *
   * Reads raw encoder positions and computes offsets so that
   * reported joint angles match default_dof_pos (assumed initial pose).
   * Called automatically in constructor.
   */
  void calibrate_offsets();

  // ------------------------------------------------------------------ //
  //  State accessors (updated after every send)                        //
  // ------------------------------------------------------------------ //
  const std::array<float, NUM_JOINTS> &dof_pos() const { return dof_pos_; }
  const std::array<float, NUM_JOINTS> &dof_vel() const { return dof_vel_; }
  const std::array<float, NUM_JOINTS> &dof_tau() const { return dof_tau_; }
  const std::array<float, NUM_JOINTS> &motor_temps() const {
    return motor_temps_;
  }

private:
  /**
   * @brief Send command to a single motor and update cache.
   */
  void send_single(int dof_idx, float q_joint, float dq_joint, float kp,
                   float kd, float tau);

  std::unique_ptr<SerialPort> serials_[2];
  std::unique_ptr<MotorCmd> cmd_;
  std::unique_ptr<MotorData> data_;
  int foc_mode_;
  RobotRuntimeConfig config_;

  // Cached joint states
  std::array<float, NUM_JOINTS> dof_pos_{};
  std::array<float, NUM_JOINTS> dof_vel_{};
  std::array<float, NUM_JOINTS> dof_tau_{};
  std::array<float, NUM_JOINTS> motor_temps_{};
  std::array<int, NUM_JOINTS> motor_errors_{};
  std::array<float, NUM_JOINTS> motor_offsets_{}; // encoder zero offsets
};

} // namespace deploy

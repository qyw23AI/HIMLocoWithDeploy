/**
 * @file motor_driver.cpp
 * @brief GO-M8010-6 motor driver implementation.
 *
 * 所有电机映射信息 (motor_id, port_idx, is_reversed) 来自
 * RobotRuntimeConfig，由 YAML 配置文件加载。
 */

#include "motor_driver.h"

#include <cmath>
#include <iostream>

#include "serialPort/SerialPort.h"
#include "unitreeMotor/unitreeMotor.h"

namespace deploy {

MotorDriver::MotorDriver(const RobotRuntimeConfig &config,
                         const std::string &port0,
                         const std::string &port1)
    : config_(config) {
  std::cout << "[MotorDriver] Opening serial ports: " << port0 << ", " << port1
            << std::endl;

  serials_[0] = std::make_unique<SerialPort>(port0);
  serials_[1] = std::make_unique<SerialPort>(port1);

  cmd_ = std::make_unique<MotorCmd>();
  data_ = std::make_unique<MotorData>();
  cmd_->motorType = MotorType::GO_M8010_6;
  data_->motorType = MotorType::GO_M8010_6;
  foc_mode_ = queryMotorMode(MotorType::GO_M8010_6, MotorMode::FOC);

  // Verify gear ratio from SDK
  float sdk_ratio = queryGearRatio(MotorType::GO_M8010_6);
  if (std::abs(sdk_ratio - config_.joint_transmission_ratio[0]) > 0.1f) {
    std::cout << "[MotorDriver] WARNING: SDK gear ratio " << sdk_ratio
              << " != config " << config_.joint_transmission_ratio[0]
              << std::endl;
  }

  // Initialize offsets to zero (will be calibrated below)
  motor_offsets_.fill(0.0f);
  dof_pos_ = config_.default_dof_pos; // Initialize to configured pose
  dof_vel_.fill(0.0f);
  dof_tau_.fill(0.0f);
  motor_temps_.fill(0.0f);
  motor_errors_.fill(0);

  // Calibrate encoder offsets so initial readings = default_dof_pos
  calibrate_offsets();
}

MotorDriver::~MotorDriver() {
  // Serials are cleaned up by unique_ptr
}

// ------------------------------------------------------------------ //
//  Encoder offset calibration                                         //
// ------------------------------------------------------------------ //

void MotorDriver::calibrate_offsets() {
  std::cout << "[MotorDriver] Calibrating encoder offsets..." << std::endl;
  std::cout << "[MotorDriver] Assuming current pose = default_dof_pos"
            << std::endl;

  // Read raw encoder positions by sending zero-torque commands
  // (offsets are currently 0, so data_->q is the raw encoder value)
  // Send a few times to ensure we get valid readings
  constexpr int NUM_READS = 3;
  for (int r = 0; r < NUM_READS; ++r) {
    for (int i = 0; i < NUM_JOINTS; ++i) {
      const auto &mapping = config_.motor_map[i];
      float direction = mapping.is_reversed ? -1.0f : 1.0f;

      cmd_->mode = foc_mode_;
      cmd_->id = static_cast<unsigned short>(mapping.motor_id);
      cmd_->q = 0.0f;
      cmd_->dq = 0.0f;
      cmd_->kp = 0.0f;
      cmd_->kd = 0.0f;
      cmd_->tau = 0.0f;

      data_->correct = false;
      serials_[mapping.port_idx]->sendRecv(cmd_.get(), data_.get());

      if (r == NUM_READS - 1 && data_->correct &&
          static_cast<int>(data_->motor_id) == mapping.motor_id) {
        const float ratio = config_.joint_transmission_ratio[i];
        // offset = raw_q - direction * q_joint * ratio
        float raw_q = data_->q;
        motor_offsets_[i] =
          raw_q - direction * config_.default_dof_pos[i] * ratio;
        dof_pos_[i] = config_.default_dof_pos[i];
        dof_vel_[i] = direction * data_->dq / ratio;
        dof_tau_[i] = direction * data_->tau * ratio;

        std::cout << "[MotorDriver] " << config_.joint_names[i]
                  << " (motor " << mapping.motor_id << ")"
                  << ": raw_q=" << raw_q
                  << " offset=" << motor_offsets_[i]
                  << " -> joint=" << config_.default_dof_pos[i] << " rad"
                  << std::endl;
      } else if (r == NUM_READS - 1) {
        std::cout << "[MotorDriver] WARNING: " << config_.joint_names[i]
                  << " (motor " << mapping.motor_id
                  << ") did not reply, using offset=0" << std::endl;
      }
    }
  }

  std::cout << "[MotorDriver] Calibration complete." << std::endl;
}

// ------------------------------------------------------------------ //
//  Low-level: single motor                                            //
// ------------------------------------------------------------------ //

void MotorDriver::send_single(int dof_idx, float q_joint, float dq_joint,
                              float kp, float kd, float tau) {
  const auto &mapping = config_.motor_map[dof_idx];
  const float ratio = config_.joint_transmission_ratio[dof_idx];
  float direction = mapping.is_reversed ? -1.0f : 1.0f;

  // Convert joint-side → motor-side
  float q_motor = direction * q_joint * ratio + motor_offsets_[dof_idx];
  float dq_motor = direction * dq_joint * ratio;
  float tau_motor = direction * tau / ratio;
  float kp_motor = kp / (ratio * ratio);
  float kd_motor = kd / (ratio * ratio);

  cmd_->mode = foc_mode_;
  cmd_->id = static_cast<unsigned short>(mapping.motor_id);
  cmd_->q = q_motor;
  cmd_->dq = dq_motor;
  cmd_->kp = kp_motor;
  cmd_->kd = kd_motor;
  cmd_->tau = tau_motor;

  // Reset correct flag before sendRecv
  data_->correct = false;

  serials_[mapping.port_idx]->sendRecv(cmd_.get(), data_.get());

  // Only update cached values if the motor actually replied
  if (data_->correct && static_cast<int>(data_->motor_id) == mapping.motor_id) {
    dof_pos_[dof_idx] = direction * (data_->q - motor_offsets_[dof_idx]) / ratio;
    dof_vel_[dof_idx] = direction * data_->dq / ratio;
    dof_tau_[dof_idx] = direction * data_->tau * ratio;
    motor_temps_[dof_idx] = static_cast<float>(data_->temp);
    motor_errors_[dof_idx] = data_->merror;
  }
  // else: keep previous cached values for this DOF unchanged
}

// ------------------------------------------------------------------ //
//  High-level commands                                                //
// ------------------------------------------------------------------ //

void MotorDriver::send_commands(
    const std::array<float, NUM_JOINTS> &target_dof_pos,
    const std::array<float, NUM_JOINTS> &kp,
    const std::array<float, NUM_JOINTS> &kd) {
  for (int i = 0; i < NUM_JOINTS; ++i) {
    send_single(i, target_dof_pos[i], 0.0f, kp[i], kd[i], 0.0f);
  }
}

void MotorDriver::send_damping(float kd) {
  for (int i = 0; i < NUM_JOINTS; ++i) {
    send_single(i, 0.0f, 0.0f, 0.0f, kd, 0.0f);
  }
}

void MotorDriver::set_zero_torque() {
  for (int i = 0; i < NUM_JOINTS; ++i) {
    send_single(i, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  }
}

bool MotorDriver::check_errors() const {
  bool has_error = false;
  for (int i = 0; i < NUM_JOINTS; ++i) {
    if (motor_errors_[i] != 0) {
      std::cout << "[MotorDriver] ERROR on " << config_.joint_names[i]
                << " (motor " << config_.motor_map[i].motor_id << "): code "
                << motor_errors_[i] << std::endl;
      has_error = true;
    }
  }
  return has_error;
}

} // namespace deploy

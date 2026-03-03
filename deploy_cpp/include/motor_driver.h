/**
 * @file motor_driver.h
 * @brief GO-M8010-6 motor driver wrapper for Unitree Actuator SDK.
 *
 * Handles 12 motors across 2 serial ports, with automatic joint-side ↔
 * motor-side conversion and direction reversal.
 */
#pragma once

#include <array>
#include <memory>
#include <string>
#include <mutex>

#include "robot_config.h"

// Forward declarations for Unitree SDK types
struct MotorCmd;
struct MotorData;
class SerialPort;

namespace deploy {

class MotorDriver {
public:
    /**
     * @brief Construct motor driver with two serial ports.
     * @param port0 Device path for front legs (FR+FL), e.g. "/dev/ttyUSB0"
     * @param port1 Device path for rear legs (RR+RL), e.g. "/dev/ttyUSB1"
     */
    MotorDriver(const std::string& port0 = SERIAL_PORTS[0],
                const std::string& port1 = SERIAL_PORTS[1]);
    ~MotorDriver();

    // Non-copyable
    MotorDriver(const MotorDriver&) = delete;
    MotorDriver& operator=(const MotorDriver&) = delete;

    // ------------------------------------------------------------------ //
    //  High-level commands                                                //
    // ------------------------------------------------------------------ //

    /**
     * @brief Send position commands to all 12 motors with PD control.
     * @param target_dof_pos Target joint positions [rad], size 12
     * @param kp Motor-side position gain
     * @param kd Motor-side velocity gain
     *
     * Internally updates cached joint states from motor feedback.
     */
    void send_commands(const std::array<float, NUM_JOINTS>& target_dof_pos,
                       float kp = KP_MOTOR,
                       float kd = KD_MOTOR);

    /**
     * @brief Send damping-only commands (kp=0, tau=0).
     * @param kd Motor-side damping gain
     */
    void send_damping(float kd = KD_DAMP_MOTOR);

    /**
     * @brief Emergency stop: set all gains and torques to zero.
     */
    void set_zero_torque();

    /**
     * @brief Check if any motor has non-zero error codes.
     * @return true if errors found
     */
    bool check_errors() const;

    // ------------------------------------------------------------------ //
    //  State accessors (updated after every send)                        //
    // ------------------------------------------------------------------ //
    const std::array<float, NUM_JOINTS>& dof_pos() const { return dof_pos_; }
    const std::array<float, NUM_JOINTS>& dof_vel() const { return dof_vel_; }
    const std::array<float, NUM_JOINTS>& motor_temps() const { return motor_temps_; }

private:
    /**
     * @brief Send command to a single motor and update cache.
     */
    void send_single(int dof_idx, float q_joint, float dq_joint,
                     float kp, float kd, float tau);

    std::unique_ptr<SerialPort> serials_[2];
    std::unique_ptr<MotorCmd>   cmd_;
    std::unique_ptr<MotorData>  data_;
    int foc_mode_;

    // Cached joint states
    std::array<float, NUM_JOINTS> dof_pos_{};
    std::array<float, NUM_JOINTS> dof_vel_{};
    std::array<float, NUM_JOINTS> motor_temps_{};
    std::array<int,   NUM_JOINTS> motor_errors_{};
    std::array<float, NUM_JOINTS> motor_offsets_{};  // encoder zero offsets
};

}  // namespace deploy

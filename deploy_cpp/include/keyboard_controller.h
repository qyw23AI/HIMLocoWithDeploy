/**
 * @file keyboard_controller.h
 * @brief Keyboard controller for velocity commands and state switching.
 *
 * Runs a listener thread that reads raw terminal key presses.
 *
 * Velocity controls (only effective in RL state):
 *   W / S  – increase / decrease forward velocity (vx)
 *   Q / E  – increase / decrease lateral velocity (vy)
 *   A / D  – increase / decrease yaw rate
 *
 * State switching:
 *   0      – Idle (zero torque)
 *   1      – StandUp (interpolate to standing)
 *   2      – RL (policy control)
 *   3      – JointDamping (passive damping)
 *   4      – ReturnDefault (interpolate back to startup default pose)
 *   5      – SingleStepRL (single-step RL with confirm)
 *   6      – JointSweep (manual joint direction debug)
 *   Space  – Emergency stop (force Idle)
 *   Esc    – Exit program
 *   R      – Reset velocity to zero
 *
 * Debug controls (JointSweep / SingleStepRL):
 *   Enter  – Confirm / execute current step
 *   J / K  – Select next / previous joint  (JointSweep)
 *   + / -  – Increase / decrease offset    (JointSweep)
 */
#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <thread>
#include <optional>
#include <termios.h>

#include "robot_runtime_config.h"
#include "state_machine.h"

namespace deploy {

/// State transition request
struct StateRequest {
    RobotState target;
    bool emergency;
};

class KeyboardController {
public:
    explicit KeyboardController(const RobotRuntimeConfig& config);
    ~KeyboardController();

    // Non-copyable
    KeyboardController(const KeyboardController&) = delete;
    KeyboardController& operator=(const KeyboardController&) = delete;

    /// Get current velocity commands [vx, vy, yaw_rate].
    std::array<float, 3> get_commands() const;

    /// Consume and return pending state transition request (if any).
    std::optional<StateRequest> consume_state_request();

    /// Whether exit has been requested.
    bool is_exit() const { return exit_.load(); }

    /// Restore terminal and stop thread.
    void cleanup();

    // ---- Debug mode interfaces ----

    /// Consume a pending Enter-key confirmation (returns true once per press).
    bool consume_step_confirm();

    /// Get currently selected joint index for sweep mode [0..11].
    int get_sweep_joint_idx() const { return sweep_joint_idx_.load(); }

    /// Get current sweep offset [rad].
    float get_sweep_offset() const { return sweep_offset_.load(); }

    /// Reset sweep state when entering JointSweep mode.
    void reset_sweep() { sweep_joint_idx_.store(0); sweep_offset_.store(0.0f); }

private:
    void run();
    void process_key(char key);
    void zero_commands();
    void restore_terminal();

    std::thread thread_;
    mutable std::mutex mutex_;

    float vx_   = 0.0f;
    float vy_   = 0.0f;
    float yaw_  = 0.0f;

    std::optional<StateRequest> state_request_;
    std::atomic<bool> exit_{false};

    // Debug mode state
    std::atomic<bool> step_confirmed_{false};
    std::atomic<int> sweep_joint_idx_{0};
    std::atomic<float> sweep_offset_{0.0f};

    // Terminal state backup
    int fd_;
    struct termios old_settings_;
    bool terminal_saved_ = false;
    RobotRuntimeConfig config_;
};

}  // namespace deploy

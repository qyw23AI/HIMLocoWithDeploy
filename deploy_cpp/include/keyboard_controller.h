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
 *   Space  – Emergency stop (force Idle)
 *   Esc    – Exit program
 *   R      – Reset velocity to zero
 */
#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <thread>
#include <optional>
#include <termios.h>

#include "state_machine.h"
#include "robot_config.h"

namespace deploy {

/// State transition request
struct StateRequest {
    RobotState target;
    bool emergency;
};

class KeyboardController {
public:
    KeyboardController();
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

    // Terminal state backup
    int fd_;
    struct termios old_settings_;
    bool terminal_saved_ = false;
};

}  // namespace deploy

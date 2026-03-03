/**
 * @file keyboard_controller.cpp
 * @brief Keyboard controller implementation.
 */

#include "keyboard_controller.h"

#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#include <iostream>
#include <algorithm>

namespace deploy {

KeyboardController::KeyboardController()
{
    fd_ = STDIN_FILENO;

    // Save original terminal settings
    if (tcgetattr(fd_, &old_settings_) == 0) {
        terminal_saved_ = true;
    }

    // Start keyboard listener thread
    thread_ = std::thread(&KeyboardController::run, this);
}

KeyboardController::~KeyboardController()
{
    cleanup();
}

void KeyboardController::run()
{
    try {
        // Set terminal to cbreak mode (no buffering, no echo)
        if (terminal_saved_) {
            struct termios raw = old_settings_;
            raw.c_lflag &= ~(ICANON | ECHO);
            raw.c_cc[VMIN]  = 0;
            raw.c_cc[VTIME] = 0;
            tcsetattr(fd_, TCSANOW, &raw);
        }

        while (!exit_.load()) {
            // Poll stdin with 50ms timeout
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(fd_, &readfds);

            struct timeval tv;
            tv.tv_sec  = 0;
            tv.tv_usec = 50000;  // 50 ms

            int ret = select(fd_ + 1, &readfds, nullptr, nullptr, &tv);
            if (ret > 0 && FD_ISSET(fd_, &readfds)) {
                char key;
                if (read(fd_, &key, 1) == 1) {
                    process_key(key);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[Keyboard] Error: " << e.what() << std::endl;
    }

    restore_terminal();
}

void KeyboardController::process_key(char key)
{
    std::lock_guard<std::mutex> lock(mutex_);

    switch (key) {
    // ---- Velocity commands ----
    case 'w': case 'W':
        vx_ = std::min(vx_ + CMD_VX_STEP, CMD_VX_MAX);
        break;
    case 's': case 'S':
        vx_ = std::max(vx_ - CMD_VX_STEP, CMD_VX_MIN);
        break;
    case 'q': case 'Q':
        vy_ = std::min(vy_ + CMD_VY_STEP, CMD_VY_MAX);
        break;
    case 'e': case 'E':
        vy_ = std::max(vy_ - CMD_VY_STEP, CMD_VY_MIN);
        break;
    case 'a': case 'A':
        yaw_ = std::min(yaw_ + CMD_YAW_STEP, CMD_YAW_MAX);
        break;
    case 'd': case 'D':
        yaw_ = std::max(yaw_ - CMD_YAW_STEP, CMD_YAW_MIN);
        break;

    // ---- State transitions ----
    case '0':
        state_request_ = StateRequest{RobotState::IDLE, false};
        zero_commands();
        break;
    case '1':
        state_request_ = StateRequest{RobotState::STAND_UP, false};
        zero_commands();
        break;
    case '2':
        state_request_ = StateRequest{RobotState::RL, false};
        // Keep current velocity commands when entering RL
        break;
    case '3':
        state_request_ = StateRequest{RobotState::JOINT_DAMPING, false};
        zero_commands();
        break;

    // ---- Emergency stop ----
    case ' ':
        state_request_ = StateRequest{RobotState::IDLE, true};
        zero_commands();
        break;

    // ---- Exit ----
    case '\x1b':  // Escape
        exit_.store(true);
        break;

    // ---- Reset velocity to zero ----
    case 'r': case 'R':
        zero_commands();
        break;

    default:
        break;
    }
}

void KeyboardController::zero_commands()
{
    vx_  = 0.0f;
    vy_  = 0.0f;
    yaw_ = 0.0f;
}

std::array<float, 3> KeyboardController::get_commands() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return {vx_, vy_, yaw_};
}

std::optional<StateRequest> KeyboardController::consume_state_request()
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto req = state_request_;
    state_request_.reset();
    return req;
}

void KeyboardController::restore_terminal()
{
    if (terminal_saved_) {
        tcsetattr(fd_, TCSADRAIN, &old_settings_);
        terminal_saved_ = false;
    }
}

void KeyboardController::cleanup()
{
    exit_.store(true);
    if (thread_.joinable()) {
        thread_.join();
    }
    restore_terminal();
}

}  // namespace deploy

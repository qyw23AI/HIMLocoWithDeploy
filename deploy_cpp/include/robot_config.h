/**
 * @file robot_config.h
 * @brief Robot configuration for Mybot quadruped deployment.
 *
 * All parameters match the simulation training config
 * (mybot_config.py + legged_robot_config.py).
 */
#pragma once

#include <array>
#include <string>
#include <cmath>

namespace deploy {

// ========================== Joint Configuration ==========================
constexpr int NUM_JOINTS  = 12;
constexpr int NUM_ACTIONS = 12;

// Joint names in DOF order (IsaacGym body depth-first traversal of mybot.xml)
// Index: 0-2 FR, 3-5 FL, 6-8 RR, 9-11 RL
inline const char* JOINT_NAMES[NUM_JOINTS] = {
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",   // DOF 0-2
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",   // DOF 3-5
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",   // DOF 6-8
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",   // DOF 9-11
};

// Default standing joint angles [rad] (from mybot_config.py init_state)
constexpr std::array<float, NUM_JOINTS> DEFAULT_DOF_POS = {
    -0.1f,  0.8f, -1.5f,  // FR: hip, thigh, calf
     0.1f,  0.8f, -1.5f,  // FL: hip, thigh, calf
    -0.1f,  1.0f, -1.5f,  // RR: hip, thigh, calf
     0.1f,  1.0f, -1.5f,  // RL: hip, thigh, calf
};

// Joint position limits [rad] (from mybot.xml joint range attributes)
constexpr std::array<float, NUM_JOINTS> JOINT_POS_LOWER = {
    -0.48f, -1.44f, -2.70f,  // FR
    -0.48f, -1.44f, -2.70f,  // FL
    -0.48f, -1.44f, -2.70f,  // RR
    -0.48f, -1.44f, -2.70f,  // RL
};

constexpr std::array<float, NUM_JOINTS> JOINT_POS_UPPER = {
    0.48f, 1.44f, -0.60f,  // FR
    0.48f, 1.44f, -0.60f,  // FL
    0.48f, 1.44f, -0.60f,  // RR
    0.48f, 1.44f, -0.60f,  // RL
};

// ========================== Motor Configuration ==========================
// GO-M8010-6 gear ratio
constexpr float GEAR_RATIO = 6.33f;

// Motor mapping: DOF index -> (motor_id, port_idx, is_reversed)
struct MotorMapping {
    int motor_id;
    int port_idx;    // 0 = /dev/ttyUSB0, 1 = /dev/ttyUSB1
    bool is_reversed;
};

constexpr std::array<MotorMapping, NUM_JOINTS> MOTOR_MAP = {{
    {1,  0, false},  // DOF 0:  FR_hip
    {2,  0, true },  // DOF 1:  FR_thigh   (reversed)
    {3,  0, true },  // DOF 2:  FR_calf    (reversed)
    {4,  0, false},  // DOF 3:  FL_hip
    {5,  0, false},  // DOF 4:  FL_thigh
    {6,  0, false},  // DOF 5:  FL_calf
    {10, 1, true },  // DOF 6:  RR_hip     (reversed)
    {11, 1, true },  // DOF 7:  RR_thigh   (reversed)
    {12, 1, true },  // DOF 8:  RR_calf    (reversed)
    {7,  1, true },  // DOF 9:  RL_hip     (reversed)
    {8,  1, false},  // DOF 10: RL_thigh
    {9,  1, false},  // DOF 11: RL_calf
}};

// Serial port device paths
inline const std::string SERIAL_PORTS[2] = {"/dev/ttyUSB0", "/dev/ttyUSB1"};

// ========================== PD Control ==========================
// Joint-side PD gains (from mybot_config.py control class)
constexpr float KP_JOINT = 40.0f;   // [N·m/rad]
constexpr float KD_JOINT = 1.0f;    // [N·m·s/rad]

// Motor-side PD gains (converted through gear ratio)
constexpr float KP_MOTOR = KP_JOINT / (GEAR_RATIO * GEAR_RATIO);  // ≈ 0.999
constexpr float KD_MOTOR = KD_JOINT / (GEAR_RATIO * GEAR_RATIO);  // ≈ 0.025

// Joint damping state motor-side gain (for safe deceleration)
constexpr float KD_DAMP_MOTOR = 0.1f;

// ========================== Policy / Observation ==========================
// Observation scales (from legged_robot_config.py normalization.obs_scales)
constexpr float LIN_VEL_SCALE = 2.0f;
constexpr float ANG_VEL_SCALE = 0.25f;
constexpr float DOF_POS_SCALE = 1.0f;
constexpr float DOF_VEL_SCALE = 0.05f;

// Command scaling: [lin_vel_x, lin_vel_y, ang_vel_yaw]
constexpr std::array<float, 3> COMMANDS_SCALE = {LIN_VEL_SCALE, LIN_VEL_SCALE, ANG_VEL_SCALE};

// Policy action parameters (from mybot_config.py control class)
constexpr float ACTION_SCALE   = 0.25f;
constexpr float HIP_REDUCTION  = 1.0f;
constexpr std::array<int, 4> HIP_INDICES = {0, 3, 6, 9};

// Observation dimensions
constexpr int NUM_ONE_STEP_OBS = 45;
constexpr int HISTORY_LENGTH   = 6;
constexpr int NUM_OBS          = NUM_ONE_STEP_OBS * HISTORY_LENGTH;  // 270

// Clipping (from legged_robot_config.py normalization)
constexpr float CLIP_OBS     = 100.0f;
constexpr float CLIP_ACTIONS = 100.0f;

// Velocity command deadband
constexpr float CMD_DEADBAND = 0.2f;

// ========================== Control Loop ==========================
constexpr float CONTROL_DT = 0.02f;     // 50 Hz policy frequency
constexpr float SIM_DT     = 0.005f;    // sim dt (for reference)
constexpr int   DECIMATION = 4;

// ========================== Command Ranges ==========================
constexpr float CMD_VX_MIN  = -1.0f;   // [m/s]
constexpr float CMD_VX_MAX  =  1.0f;
constexpr float CMD_VY_MIN  = -1.0f;   // [m/s]
constexpr float CMD_VY_MAX  =  1.0f;
constexpr float CMD_YAW_MIN = -3.14f;  // [rad/s]
constexpr float CMD_YAW_MAX =  3.14f;

// Keyboard step sizes
constexpr float CMD_VX_STEP  = 0.1f;
constexpr float CMD_VY_STEP  = 0.1f;
constexpr float CMD_YAW_STEP = 0.3f;

// ========================== State Machine ==========================
constexpr float STANDUP_DURATION = 2.0f;  // seconds to interpolate to standing pose

}  // namespace deploy

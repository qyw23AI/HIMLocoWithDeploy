/**
 * @file policy_runner.h
 * @brief Policy inference runner for HIM locomotion policy (LibTorch JIT).
 *
 * Loads a JIT-exported model (policy.pt), builds the 270-dimensional
 * observation history, and runs inference to produce 12-dim actions.
 *
 * Observation layout (45 dims per step, 6 steps history = 270 total):
 *   [0:3]   commands * commands_scale
 *   [3:6]   base_ang_vel * ang_vel_scale
 *   [6:9]   projected_gravity
 *   [9:21]  (dof_pos - default_dof_pos) * dof_pos_scale
 *   [21:33] dof_vel * dof_vel_scale
 *   [33:45] last_actions
 */
#pragma once

#include <array>
#include <string>

#include <torch/script.h>
#include <torch/torch.h>

#include "robot_runtime_config.h"

namespace deploy {

class PolicyRunner {
public:
    /**
     * @brief Construct policy runner.
     * @param policy_path Path to JIT-exported policy.pt file
     * @param device Torch device string, e.g. "cuda:0" or "cpu"
     */
    PolicyRunner(const std::string& policy_path, const RobotRuntimeConfig& config);

    /// Reset observation history and last actions (call when entering RL state).
    void reset();

    /**
     * @brief Build single-step observation vector (45 dims).
     *
     * Strictly reproduces legged_robot.py compute_observations():
     *   cat([commands*scale, ang_vel*scale, gravity,
     *        (dof_pos-default)*scale, dof_vel*scale, last_actions])
     */
    torch::Tensor compute_obs(const std::array<float, 3>& commands,
                              const std::array<float, 3>& ang_vel,
                              const std::array<float, 3>& projected_gravity,
                              const std::array<float, NUM_JOINTS>& dof_pos,
                              const std::array<float, NUM_JOINTS>& dof_vel);

    /// Update sliding-window observation history.
    void update_obs_history(const torch::Tensor& current_obs);

    /**
     * @brief Run policy inference on current observation history.
     * @return Raw actions as std::array<float, 12>
     */
    std::array<float, NUM_ACTIONS> infer();

    /**
     * @brief Convert raw actions to target joint positions.
     *
     * Reproduces: target = default_dof_pos + actions * action_scale * hip_reduction
     * Clamps to joint limits.
     */
    std::array<float, NUM_JOINTS> get_target_dof_pos(
        const std::array<float, NUM_ACTIONS>& actions);

    /**
     * @brief Full inference step: observe → update history → infer → target.
     * @param[out] target_dof_pos Target joint positions [rad]
     * @param[out] actions Raw policy actions
     */
    void step(const std::array<float, 3>& commands,
              const std::array<float, 3>& ang_vel,
              const std::array<float, 3>& projected_gravity,
              const std::array<float, NUM_JOINTS>& dof_pos,
              const std::array<float, NUM_JOINTS>& dof_vel,
              std::array<float, NUM_JOINTS>& target_dof_pos,
              std::array<float, NUM_ACTIONS>& actions);

private:
    torch::jit::script::Module model_;
    torch::Device device_;
    RobotRuntimeConfig config_;

    // Buffers on device
    torch::Tensor obs_history_;    // (1, 270)
    torch::Tensor last_actions_;   // (1, 12)
    torch::Tensor default_dof_pos_; // (1, 12)
    torch::Tensor commands_scale_;  // (1, 3)
    torch::Tensor dof_pos_scale_;   // (1, 12)
    torch::Tensor dof_vel_scale_;   // (1, 12)
    torch::Tensor action_scale_;    // (1, 12)
};

}  // namespace deploy

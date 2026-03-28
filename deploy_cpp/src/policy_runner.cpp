/**
 * @file policy_runner.cpp
 * @brief Policy inference runner implementation using LibTorch JIT.
 */

#include "policy_runner.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace deploy {

PolicyRunner::PolicyRunner(const std::string &policy_path,
                           const RobotRuntimeConfig &config)
    : device_(config.device), config_(config) {
  std::cout << "[PolicyRunner] Loading policy from " << policy_path
            << std::endl;

  try {
    model_ = torch::jit::load(policy_path, device_);
    model_.eval();
  } catch (const c10::Error &e) {
    std::cerr << "[PolicyRunner] ERROR: Failed to load model: " << e.what()
              << std::endl;
    throw;
  }

  std::cout << "[PolicyRunner] Loaded JIT model on " << config.device
            << std::endl;

  // Initialize buffers
  obs_history_ = torch::zeros(
      {1, NUM_OBS},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  last_actions_ = torch::zeros(
      {1, NUM_ACTIONS},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

  default_dof_pos_ =
      torch::from_blob(const_cast<float *>(config_.policy_dof_pos.data()),
                       {1, NUM_JOINTS}, torch::kFloat32)
          .clone()
          .to(device_);

    std::array<float, 3> command_scale = {
      config_.lin_vel_scale, config_.lin_vel_scale, config_.ang_vel_scale};
    commands_scale_ =
      torch::from_blob(command_scale.data(), {1, 3}, torch::kFloat32)
        .clone()
        .to(device_);

    dof_pos_scale_ =
      torch::from_blob(const_cast<float *>(config_.dof_pos_scale.data()),
               {1, NUM_JOINTS}, torch::kFloat32)
        .clone()
        .to(device_);
    dof_vel_scale_ =
      torch::from_blob(const_cast<float *>(config_.dof_vel_scale.data()),
               {1, NUM_JOINTS}, torch::kFloat32)
        .clone()
        .to(device_);
    action_scale_ =
      torch::from_blob(const_cast<float *>(config_.action_scale.data()),
               {1, NUM_JOINTS}, torch::kFloat32)
        .clone()
        .to(device_);
}

void PolicyRunner::reset() {
  obs_history_.zero_();
  last_actions_.zero_();
}

torch::Tensor
PolicyRunner::compute_obs(const std::array<float, 3> &commands,
                          const std::array<float, 3> &ang_vel,
                          const std::array<float, 3> &projected_gravity,
                          const std::array<float, NUM_JOINTS> &dof_pos,
                          const std::array<float, NUM_JOINTS> &dof_vel) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);

  // Apply command deadband
  std::array<float, 3> cmd = commands;
  float cmd_norm = std::sqrt(cmd[0] * cmd[0] + cmd[1] * cmd[1]);
  if (cmd_norm < config_.cmd_deadband) {
    cmd[0] = 0.0f;
    cmd[1] = 0.0f;
  }

  // Create tensors (1, N) on CPU first, then move to device
  auto t_cmd = torch::from_blob(cmd.data(), {1, 3}, opts).clone().to(device_);
  auto t_angvel =
      torch::from_blob(const_cast<float *>(ang_vel.data()), {1, 3}, opts)
          .clone()
          .to(device_);
  auto t_gravity =
      torch::from_blob(const_cast<float *>(projected_gravity.data()), {1, 3},
                       opts)
          .clone()
          .to(device_);
  auto t_dofpos = torch::from_blob(const_cast<float *>(dof_pos.data()),
                                   {1, NUM_JOINTS}, opts)
                      .clone()
                      .to(device_);
  auto t_dofvel = torch::from_blob(const_cast<float *>(dof_vel.data()),
                                   {1, NUM_JOINTS}, opts)
                      .clone()
                      .to(device_);

  // Build observation: exactly reproduces compute_observations()
  auto current_obs = torch::cat(
      {
        t_cmd * commands_scale_,                    // (1, 3)
        t_angvel * config_.ang_vel_scale,           // (1, 3)
        t_gravity,                                  // (1, 3)
        (t_dofpos - default_dof_pos_) * dof_pos_scale_, // (1, 12)
        t_dofvel * dof_vel_scale_,                  // (1, 12)
        last_actions_,                              // (1, 12)
      },
      /*dim=*/1); // (1, 45)

  return current_obs;
}

void PolicyRunner::update_obs_history(const torch::Tensor &current_obs) {
  // Sliding window: obs_history = cat([current[:, :45], obs_history[:, :-45]])
  obs_history_ = torch::cat(
      {
          current_obs.index({torch::indexing::Slice(),
                             torch::indexing::Slice(0, NUM_ONE_STEP_OBS)}),
          obs_history_.index(
              {torch::indexing::Slice(),
               torch::indexing::Slice(0, NUM_OBS - NUM_ONE_STEP_OBS)}),
      },
      /*dim=*/1);

  // Clip
  obs_history_ = torch::clamp(obs_history_, -config_.clip_obs, config_.clip_obs);
}

std::array<float, NUM_ACTIONS> PolicyRunner::infer() {
  torch::NoGradGuard no_grad;

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(obs_history_);

  auto output = model_.forward(inputs).toTensor(); // (1, 12)
  output = torch::clamp(output, -config_.clip_actions, config_.clip_actions);

  last_actions_ = output.clone();

  // Copy to std::array
  auto output_cpu = output.cpu().contiguous();
  float *data_ptr = output_cpu.data_ptr<float>();
  std::array<float, NUM_ACTIONS> actions;
  std::copy(data_ptr, data_ptr + NUM_ACTIONS, actions.begin());

  static int debug_cnt = 0;
  if (++debug_cnt % 50 == 0) {
    auto obs_cpu = obs_history_.cpu().contiguous();
    float *obs_ptr = obs_cpu.data_ptr<float>();
    std::cout << "\n======= RL STEP DEBUG =======" << std::endl;
    std::cout << "Obs (first 15): ";
    for (int i = 0; i < 15; i++)
      std::cout << obs_ptr[i] << ", ";
    std::cout << "\nActions: ";
    for (int i = 0; i < 12; i++)
      std::cout << actions[i] << ", ";
    std::cout << "\n==============================\n";
  }

  return actions;
}

std::array<float, NUM_JOINTS> PolicyRunner::get_target_dof_pos(
    const std::array<float, NUM_ACTIONS> &actions) {
  std::array<float, NUM_JOINTS> target;

  for (int i = 0; i < NUM_JOINTS; ++i) {
    float scaled = actions[i] * config_.action_scale[i];
    // Apply hip reduction
    for (int h : config_.hip_indices) {
      if (i == h) {
        scaled *= config_.hip_reduction;
        break;
      }
    }
    target[i] = config_.policy_dof_pos[i] + scaled;
    // Clamp to joint limits
    target[i] = std::clamp(target[i], config_.joint_pos_lower[i],
                           config_.joint_pos_upper[i]);
  }

  return target;
}

void PolicyRunner::step(const std::array<float, 3> &commands,
                        const std::array<float, 3> &ang_vel,
                        const std::array<float, 3> &projected_gravity,
                        const std::array<float, NUM_JOINTS> &dof_pos,
                        const std::array<float, NUM_JOINTS> &dof_vel,
                        std::array<float, NUM_JOINTS> &target_dof_pos,
                        std::array<float, NUM_ACTIONS> &actions) {
  auto current_obs =
      compute_obs(commands, ang_vel, projected_gravity, dof_pos, dof_vel);
  update_obs_history(current_obs);
  actions = infer();
  target_dof_pos = get_target_dof_pos(actions);
}

} // namespace deploy

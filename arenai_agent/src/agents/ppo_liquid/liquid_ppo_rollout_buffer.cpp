//
// Created by samuel on 06/09/2026.
//

#include "./liquid_ppo_rollout_buffer.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    namespace {
        TorchState to_cpu(const TorchState &state) {
            return {
                .vision = state.vision.detach().cpu(),
                .proprioception = state.proprioception.detach().cpu()};
        }
    }// namespace

    void LiquidPpoRolloutBuffer::add(const LiquidPpoInputStep &step) {
        const auto nb_tanks = step.state.vision.size(0);

        if (!already_terminated_.defined())
            already_terminated_ = torch::zeros(
                {nb_tanks}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

        // tanks already terminated before this step have no valid transition to store
        const auto valid = already_terminated_.logical_not();
        already_terminated_.logical_or_(
            step.done.detach().cpu().to(torch::kBool).reshape({nb_tanks}));

        steps_.push_back(
            {.step =
                 {.state = to_cpu(step.state),
                  .action =
                      {.continuous_action = step.action.continuous_action.detach().cpu(),
                       .discrete_action = step.action.discrete_action.detach().cpu()},
                  .continuous_log_prob = step.continuous_log_prob.detach().cpu(),
                  .discrete_log_prob = step.discrete_log_prob.detach().cpu(),
                  .reward = step.reward.detach().cpu(),
                  .done = step.done.detach().cpu(),
                  .actor_hidden = step.actor_hidden.detach().cpu(),
                  .episode_start = step.episode_start},
             .valid = valid});

        // the freshly added step is pending: its closing observation is not known yet
        final_state_.reset();
    }

    void LiquidPpoRolloutBuffer::finish_episode(const TorchState &final_state) {
        if (!steps_.empty()) final_state_ = to_cpu(final_state);

        if (already_terminated_.defined()) already_terminated_.fill_(false);
    }

    size_t LiquidPpoRolloutBuffer::nb_complete_steps() const {
        if (steps_.empty()) return 0;
        return final_state_.has_value() ? steps_.size() : steps_.size() - 1;
    }

    LiquidPpoRollout LiquidPpoRolloutBuffer::get_rollout() {
        const auto nb_steps = nb_complete_steps();
        TORCH_CHECK(nb_steps > 0, "LiquidPpoRolloutBuffer: no complete step to train on");

        // the observation closing the last consumed step: the episode's final state,
        // or the pending step's own observation
        const auto bootstrap_state =
            final_state_.has_value() ? *final_state_ : steps_[nb_steps].step.state;

        const auto stack = [&](const auto &get_field) {
            std::vector<torch::Tensor> tensors;
            tensors.reserve(nb_steps);
            for (size_t i = 0; i < nb_steps; i++) tensors.push_back(get_field(steps_[i]));
            return torch::stack(tensors, 0);
        };

        auto episode_starts = torch::zeros({static_cast<long>(nb_steps)}, torch::kBool);
        for (size_t i = 0; i < nb_steps; i++)
            if (steps_[i].step.episode_start) episode_starts[static_cast<long>(i)] = true;

        LiquidPpoRollout rollout{
            .states =
                {.vision = stack([](const StoredStep &s) { return s.step.state.vision; }),
                 .proprioception =
                     stack([](const StoredStep &s) { return s.step.state.proprioception; })},
            .actions =
                {.continuous_action =
                     stack([](const StoredStep &s) { return s.step.action.continuous_action; }),
                 .discrete_action =
                     stack([](const StoredStep &s) { return s.step.action.discrete_action; })},
            .continuous_log_probs =
                stack([](const StoredStep &s) { return s.step.continuous_log_prob; }),
            .discrete_log_probs =
                stack([](const StoredStep &s) { return s.step.discrete_log_prob; }),
            .rewards = stack([](const StoredStep &s) { return s.step.reward; }),
            .dones = stack([](const StoredStep &s) { return s.step.done; }),
            .bootstrap_state = bootstrap_state,
            .valids = stack([](const StoredStep &s) { return s.valid; }).unsqueeze(-1),
            .actor_hiddens = stack([](const StoredStep &s) { return s.step.actor_hidden; }),
            .episode_starts = episode_starts};

        steps_.erase(steps_.begin(), steps_.begin() + static_cast<long>(nb_steps));
        final_state_.reset();

        return rollout;
    }

}// namespace arenai::agent

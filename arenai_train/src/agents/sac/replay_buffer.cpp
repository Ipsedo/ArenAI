//
// Created by samuel on 03/10/2025.
//

#include "./replay_buffer.h"

#include <algorithm>

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

    SacReplayBuffer::SacReplayBuffer(const int memory_size)
        : initialized_(false), memory_size_(memory_size), nb_steps_(0), write_idx_(0), size_(0),
          nb_tanks_(0) {}

    void SacReplayBuffer::initialize(const SacInputStep &first_step) {
        constexpr auto cpu = torch::kCPU;

        nb_tanks_ = first_step.state.vision.size(0);
        nb_steps_ = memory_size_ / static_cast<size_t>(nb_tanks_);
        TORCH_CHECK(
            nb_steps_ > 0, "SacReplayBuffer: memory_size (", memory_size_,
            ") must be >= nb_tanks (", nb_tanks_, ")");

        const auto mem = static_cast<int64_t>(nb_steps_);

        const auto make_storage = [&](const torch::Tensor &ref) {
            auto sizes = ref.sizes().vec();
            sizes.insert(sizes.begin(), mem);
            return torch::empty(sizes, ref.options().device(cpu).requires_grad(false));
        };

        store_vision_ = make_storage(first_step.state.vision);
        store_proprioception_ = make_storage(first_step.state.proprioception);
        store_cont_action_ = make_storage(first_step.action.continuous_action);
        store_disc_action_ = make_storage(first_step.action.discrete_action);
        store_reward_ = make_storage(first_step.reward);
        store_done_ = make_storage(first_step.done);

        const auto bool_cpu = torch::TensorOptions().dtype(torch::kBool).device(cpu);
        store_sampleable_ = torch::zeros({mem, nb_tanks_}, bool_cpu);
        already_terminated_ = torch::zeros({nb_tanks_}, bool_cpu);

        initialized_ = true;
    }

    void SacReplayBuffer::add(const SacInputStep &step) {
        if (!initialized_) initialize(step);

        const auto idx = static_cast<int64_t>(write_idx_);

        store_vision_[idx].copy_(step.state.vision);
        store_proprioception_[idx].copy_(step.state.proprioception);
        store_cont_action_[idx].copy_(step.action.continuous_action.detach());
        store_disc_action_[idx].copy_(step.action.discrete_action.detach());
        store_reward_[idx].copy_(step.reward);
        store_done_[idx].copy_(step.done);

        // tanks already terminated before this step have no valid transition to store
        store_sampleable_[idx].copy_(already_terminated_.logical_not());
        already_terminated_.logical_or_(
            step.done.to(torch::kBool)
                .reshape({nb_tanks_})
                .logical_or(step.truncated.to(torch::kBool).reshape({nb_tanks_})));

        advance_write_idx();
    }

    void SacReplayBuffer::finish_episode(const TorchState &final_step) {
        if (!initialized_) return;

        // the final observation is only read as next_state of the episode's last transitions:
        // stored in the ring but never sampled as a starting state
        const auto idx = static_cast<int64_t>(write_idx_);

        store_vision_[idx].copy_(final_step.vision);
        store_proprioception_[idx].copy_(final_step.proprioception);
        store_cont_action_[idx].zero_();
        store_disc_action_[idx].zero_();
        store_reward_[idx].zero_();
        store_done_[idx].zero_();

        store_sampleable_[idx].fill_(false);
        already_terminated_.fill_(false);

        advance_write_idx();
    }

    void SacReplayBuffer::advance_write_idx() {
        write_idx_ = (write_idx_ + 1) % nb_steps_;
        if (size_ < nb_steps_) size_++;
    }

    SacTrainStep SacReplayBuffer::sample(int batch_size, const torch::Device device) const {
        // valid starting pairs: sampleable, and their following slot is already written
        const auto valid = store_sampleable_.clone();
        const auto last_written = static_cast<int64_t>((write_idx_ + nb_steps_ - 1) % nb_steps_);
        valid[last_written] = false;

        const auto valid_pairs = torch::nonzero(valid);
        const auto nb_transitions = valid_pairs.size(0);
        TORCH_CHECK(nb_transitions > 0, "SacReplayBuffer: no transition to sample");

        batch_size = std::max(1, std::min(batch_size, static_cast<int>(nb_transitions)));

        const auto pick = torch::randint(
            nb_transitions, {batch_size}, torch::TensorOptions().dtype(torch::kInt64));
        const auto chosen = valid_pairs.index_select(0, pick);
        const auto step_idx = chosen.select(1, 0);
        const auto tank_idx = chosen.select(1, 1);
        const auto next_idx = (step_idx + 1).remainder(static_cast<int64_t>(nb_steps_));

        const auto take = [&](const torch::Tensor &store, const torch::Tensor &steps) {
            return store.index({steps, tank_idx}).to(device);
        };

        return {
            .state =
                {.vision = take(store_vision_, step_idx),
                 .proprioception = take(store_proprioception_, step_idx)},
            .action =
                {.continuous_action = take(store_cont_action_, step_idx),
                 .discrete_action = take(store_disc_action_, step_idx)},
            .reward = take(store_reward_, step_idx),
            .done = take(store_done_, step_idx),
            .next_state = {
                .vision = take(store_vision_, next_idx),
                .proprioception = take(store_proprioception_, next_idx)}};
    }

    size_t SacReplayBuffer::size() const { return size_; }

}// namespace arenai::train

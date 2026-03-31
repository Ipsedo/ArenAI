//
// Created by samuel on 03/10/2025.
//

#include <arenai_train/replay_buffer.h>

ReplayBuffer::ReplayBuffer(const int memory_size)
    : memory_size_(memory_size), write_idx_(0), size_(0), initialized_(false) {}

void ReplayBuffer::initialize(const TorchStep &first_step) {
    const auto mem = static_cast<int64_t>(memory_size_);
    constexpr auto cpu = torch::kCPU;

    const auto make_storage = [&](const torch::Tensor &ref) {
        auto sizes = ref.sizes().vec();
        sizes.insert(sizes.begin(), mem);
        return torch::empty(sizes, ref.options().device(cpu).requires_grad(false));
    };

    store_state_vision_ = make_storage(first_step.state.vision);
    store_state_proprioception_ = make_storage(first_step.state.proprioception);
    store_cont_action_ = make_storage(first_step.action.continuous_action);
    store_disc_action_ = make_storage(first_step.action.discrete_action);
    store_reward_ = make_storage(first_step.reward);
    store_done_ = make_storage(first_step.done);
    store_next_vision_ = make_storage(first_step.next_state.vision);
    store_next_proprioception_ = make_storage(first_step.next_state.proprioception);

    initialized_ = true;
}

void ReplayBuffer::add(const TorchStep &step) {
    if (!initialized_) initialize(step);

    const auto idx = static_cast<int64_t>(write_idx_);

    store_state_vision_[idx].copy_(step.state.vision);
    store_state_proprioception_[idx].copy_(step.state.proprioception);
    store_cont_action_[idx].copy_(step.action.continuous_action.detach());
    store_disc_action_[idx].copy_(step.action.discrete_action.detach());
    store_reward_[idx].copy_(step.reward);
    store_done_[idx].copy_(step.done);
    store_next_vision_[idx].copy_(step.next_state.vision);
    store_next_proprioception_[idx].copy_(step.next_state.proprioception);

    write_idx_ = (write_idx_ + 1) % memory_size_;
    if (size_ < memory_size_) size_++;
}

TorchStep ReplayBuffer::sample(int batch_size, torch::Device device) const {
    batch_size = std::max(1, std::min(batch_size, static_cast<int>(size_)));

    const auto idx = torch::randint(
        static_cast<long>(size_), {batch_size},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    return {
        {store_state_vision_.index_select(0, idx).to(device),
         store_state_proprioception_.index_select(0, idx).to(device)},
        {store_cont_action_.index_select(0, idx).to(device),
         store_disc_action_.index_select(0, idx).to(device)},
        store_reward_.index_select(0, idx).to(device),
        store_done_.index_select(0, idx).to(device),
        {store_next_vision_.index_select(0, idx).to(device),
         store_next_proprioception_.index_select(0, idx).to(device)}};
}

int ReplayBuffer::size() const { return size_; }

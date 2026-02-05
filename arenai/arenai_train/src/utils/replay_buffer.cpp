//
// Created by samuel on 03/10/2025.
//

#include "./replay_buffer.h"

ReplayBuffer::ReplayBuffer(const int memory_size)
    : memory_size_(memory_size), write_idx_(0), size_(0) {
    memory.resize(memory_size);
}

TorchStep ReplayBuffer::sample(int batch_size, torch::Device device) {
    batch_size = std::max(1, std::min(batch_size, static_cast<int>(size_)));

    const auto idx = torch::randint(
        static_cast<long>(size_), {batch_size},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    std::vector<torch::Tensor> states_vision, states_proprioception, actions, log_probas, rewards,
        dones, next_states_vision, next_states_proprioception;

    auto idx_acc = idx.accessor<int64_t, 1>();

    for (int i = 0; i < batch_size; i++) {
        auto [state, action, log_proba, reward, done, next_state] =
            memory[static_cast<size_t>(idx_acc[i])];

        states_vision.push_back(state.vision);
        states_proprioception.push_back(state.proprioception);

        actions.push_back(action);
        log_probas.push_back(log_proba);
        rewards.push_back(reward);
        dones.push_back(done);

        next_states_vision.push_back(next_state.vision);
        next_states_proprioception.push_back(next_state.proprioception);
    }

    return {
        {torch::stack(states_vision).to(device), torch::stack(states_proprioception).to(device)},
        torch::stack(actions).to(device),
        torch::stack(log_probas).to(device),
        torch::stack(rewards).to(device),
        torch::stack(dones).to(device),
        {torch::stack(next_states_vision).to(device),
         torch::stack(next_states_proprioception).to(device)}};
}

void ReplayBuffer::add(const TorchStep &step) {
    memory[write_idx_] = clone_step(step);

    write_idx_ = (write_idx_ + 1) % memory_size_;
    if (size_ < memory_size_) size_++;
}

int ReplayBuffer::size() const { return memory.size(); }

TorchStep ReplayBuffer::clone_step(const TorchStep &to_clone) {

    return {
        {to_clone.state.vision.clone(), to_clone.state.proprioception.clone()},
        to_clone.action.detach().clone(),
        to_clone.log_proba.detach().clone(),
        to_clone.reward.clone(),
        to_clone.done.clone(),
        {to_clone.next_state.vision.clone(), to_clone.next_state.proprioception.clone()}};
}

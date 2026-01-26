//
// Created by samuel on 03/10/2025.
//

#include "./replay_buffer.h"

ReplayBuffer::ReplayBuffer(const int memory_size) : memory_size(memory_size) {}

TorchStep ReplayBuffer::sample(int batch_size, torch::Device device) {
    batch_size = std::min(batch_size, static_cast<int>(memory.size()));

    std::vector<torch::Tensor> states_vision, states_proprioception, actions, log_probas, rewards,
        dones, next_states_vision, next_states_proprioception;

    if (batch_size == 0) throw std::invalid_argument("batch size must be greater than 0");

    const auto rand_perm = torch::randperm(memory.size());

    for (int i = 0; i < batch_size; i++) {
        auto [state, action, log_proba, reward, done, next_state] =
            memory[rand_perm[i].item<int>()];

        states_vision.push_back(state.vision);
        states_proprioception.push_back(state.proprioception);

        actions.push_back(action.detach());
        log_probas.push_back(log_proba.detach());
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

void ReplayBuffer::add(TorchStep step) {
    memory.push_back(std::move(step));
    while (memory.size() > memory_size) memory.erase(memory.begin());
}

int ReplayBuffer::size() const { return memory.size(); }

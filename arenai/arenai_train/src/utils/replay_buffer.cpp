//
// Created by samuel on 03/10/2025.
//

#include "./replay_buffer.h"

ReplayBuffer::ReplayBuffer(const int memory_size, const int seed)
    : rng(seed), memory_size(memory_size) {}

TorchStep ReplayBuffer::sample(int batch_size, torch::Device device) {
    batch_size = std::min(batch_size, static_cast<int>(memory.size()));
    std::uniform_int_distribution distribution(0, static_cast<int>(memory.size()) - 1);

    std::vector<torch::Tensor> states_vision, states_proprioception, actions, rewards,
        potential_rewards, dones, next_states_vision, next_states_proprioception,
        next_potential_rewards;

    if (batch_size == 0) throw std::invalid_argument("batch size must be greater than 0");

    for (int i = 0; i < batch_size; i++) {
        const int random_index = distribution(rng);

        auto [state, action, reward, done, next_state] = memory[random_index];

        states_vision.push_back(state.vision);
        states_proprioception.push_back(state.proprioception);
        potential_rewards.push_back(state.potential_reward);

        actions.push_back(action.detach());
        rewards.push_back(reward);
        dones.push_back(done);

        next_states_vision.push_back(next_state.vision);
        next_states_proprioception.push_back(next_state.proprioception);
        next_potential_rewards.push_back(next_state.potential_reward);
    }

    return {
        {torch::stack(states_vision).to(device), torch::stack(states_proprioception).to(device),
         torch::stack(potential_rewards).to(device)},
        torch::stack(actions).to(device),
        torch::stack(rewards).to(device),
        torch::stack(dones).to(device),
        {torch::stack(next_states_vision).to(device),
         torch::stack(next_states_proprioception).to(device),
         torch::stack(next_potential_rewards).to(device)}};
}

void ReplayBuffer::add(TorchStep step) {
    memory.push_back(std::move(step));
    while (memory.size() > memory_size) memory.erase(memory.begin());
}

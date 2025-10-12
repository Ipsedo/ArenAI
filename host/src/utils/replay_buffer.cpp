//
// Created by samuel on 03/10/2025.
//

#include "replay_buffer.h"

ReplayBuffer::ReplayBuffer(const int memory_size, const int seed)
    : rng(seed), memory_size(memory_size) {}

TorchStep ReplayBuffer::sample(int batch_size, torch::Device device) {
    batch_size = std::min(batch_size, static_cast<int>(memory.size()));
    std::uniform_int_distribution<int> distribution(0, static_cast<int>(memory.size()) - 1);

    std::vector<torch::Tensor> states_vision, states_proprioception, actions, rewards, dones,
        next_states_vision, next_states_proprioception;

    for (int i = 0; i < batch_size; i++) {
        const int random_index = distribution(rng);

        auto [state, action, reward, done, next_state] = memory[random_index];

        states_vision.push_back(state.vision);
        states_proprioception.push_back(state.proprioception);
        actions.push_back(action);
        rewards.push_back(reward);
        dones.push_back(done);
        next_states_vision.push_back(next_state.vision);
        next_states_proprioception.push_back(next_state.proprioception);
    }

    return {
        {torch::stack(states_vision).to(device), torch::stack(states_proprioception).to(device)},
        torch::stack(actions).to(device),
        torch::stack(rewards).unsqueeze(-1).to(device),
        torch::stack(dones).unsqueeze(-1).to(device),
        {torch::stack(next_states_vision).to(device),
         torch::stack(next_states_proprioception).to(device)}};
}

void ReplayBuffer::add(const TorchStep &step) {
    memory.push_back(
        {{step.state.vision.clone(), step.state.proprioception.clone()},
         step.action.detach().cpu().clone(),
         step.reward.clone(),
         step.done.clone(),
         {step.next_state.vision.clone(), step.next_state.proprioception.clone()}});
    while (memory.size() > memory_size) memory.erase(memory.begin());
}

//
// Created by claude on 08/11/2026.
//

#include "./sac_replay_buffer.h"

#include <algorithm>

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    namespace {
        const auto CPU = torch::TensorOptions().device(torch::kCPU);

        torch::Tensor to_cpu(const torch::Tensor &tensor) { return tensor.detach().cpu(); }
    }// namespace

    SacReplayBuffer::SacReplayBuffer(const int64_t capacity) : capacity(capacity) {
        TORCH_CHECK(capacity > 0, "SacReplayBuffer: capacity must be positive");
    }

    void SacReplayBuffer::allocate(const SacInputStep &step) {
        nb_tanks = step.state.vision.size(0);
        // a ring needs a slot and its successor
        slots_per_tank = std::max<int64_t>(2, capacity / nb_tanks);

        const auto shape = [&](const torch::Tensor &per_tank) {
            auto sizes = per_tank.sizes().vec();
            sizes[0] = slots_per_tank;
            sizes.insert(sizes.begin(), nb_tanks);
            return sizes;
        };

        visions = torch::zeros(shape(step.state.vision), CPU.dtype(torch::kUInt8));
        proprioceptions = torch::zeros(shape(step.state.proprioception), CPU.dtype(torch::kFloat));
        fire_masks = torch::zeros(shape(step.state.fire_mask), CPU.dtype(torch::kFloat));
        continuous_actions =
            torch::zeros(shape(step.action.continuous_action), CPU.dtype(torch::kFloat));
        discrete_actions =
            torch::zeros(shape(step.action.discrete_action), CPU.dtype(torch::kFloat));
        rewards = torch::zeros({nb_tanks, slots_per_tank, 1}, CPU.dtype(torch::kFloat));
        cuts = torch::zeros({nb_tanks, slots_per_tank, 1}, CPU.dtype(torch::kFloat));
        valids = torch::zeros({nb_tanks, slots_per_tank}, CPU.dtype(torch::kBool));

        already_terminated = torch::zeros({nb_tanks}, CPU.dtype(torch::kBool));
    }

    void SacReplayBuffer::write_observation(const TorchState &state, const torch::Tensor &valid) {
        visions.select(1, write_index).copy_(to_cpu(state.vision));
        proprioceptions.select(1, write_index).copy_(to_cpu(state.proprioception));
        fire_masks.select(1, write_index).copy_(to_cpu(state.fire_mask));
        valids.select(1, write_index).copy_(valid);

        write_index = (write_index + 1) % slots_per_tank;
        nb_written++;
    }

    void SacReplayBuffer::add(const SacInputStep &step) {
        if (nb_tanks == 0) allocate(step);

        const auto done = to_cpu(step.done).to(torch::kBool).reshape({nb_tanks});
        const auto truncated = to_cpu(step.truncated).to(torch::kBool).reshape({nb_tanks});

        // a tank already dead before this step has no live transition to store
        const auto valid = already_terminated.logical_not();
        already_terminated.logical_or_(done);

        continuous_actions.select(1, write_index).copy_(to_cpu(step.action.continuous_action));
        discrete_actions.select(1, write_index).copy_(to_cpu(step.action.discrete_action));
        rewards.select(1, write_index).copy_(to_cpu(step.reward).reshape({nb_tanks, 1}));
        cuts.select(1, write_index)
            .copy_(done.logical_or(truncated).to(torch::kFloat).reshape({nb_tanks, 1}));

        write_observation(step.state, valid);
    }

    void SacReplayBuffer::finish_episode(const TorchState &final_state) {
        if (nb_tanks == 0) return;

        write_observation(final_state, torch::zeros({nb_tanks}, CPU.dtype(torch::kBool)));
        already_terminated.fill_(false);
    }

    torch::Tensor SacReplayBuffer::sampleable() const {
        auto mask = valids.clone();
        // the newest slot's successor is not written yet
        const auto newest = (write_index + slots_per_tank - 1) % slots_per_tank;
        mask.select(1, newest).fill_(false);
        return mask;
    }

    int64_t SacReplayBuffer::nb_transitions() const {
        if (nb_tanks == 0) return 0;
        return sampleable().sum().item<int64_t>();
    }

    SacBatch SacReplayBuffer::sample(const int64_t batch_size, const torch::Device device) const {
        const auto candidates = torch::nonzero(sampleable().flatten()).squeeze(-1);
        TORCH_CHECK(candidates.size(0) > 0, "SacReplayBuffer: no transition to sample");

        const auto picks =
            torch::randint(0, candidates.size(0), {batch_size}, CPU.dtype(torch::kLong));
        const auto flat = candidates.index_select(0, picks);

        const auto tanks = torch::floor_divide(flat, slots_per_tank);
        const auto slots = flat - tanks * slots_per_tank;
        const auto next_flat = tanks * slots_per_tank + (slots + 1) % slots_per_tank;

        const auto gather = [&](const torch::Tensor &store, const torch::Tensor &idx) {
            return store.flatten(0, 1).index_select(0, idx).to(device);
        };

        return {
            .states =
                {.vision = gather(visions, flat),
                 .proprioception = gather(proprioceptions, flat),
                 .fire_mask = gather(fire_masks, flat)},
            .actions =
                {.continuous_action = gather(continuous_actions, flat),
                 .discrete_action = gather(discrete_actions, flat)},
            .rewards = gather(rewards, flat),
            .cuts = gather(cuts, flat),
            .next_states = {
                .vision = gather(visions, next_flat),
                .proprioception = gather(proprioceptions, next_flat),
                .fire_mask = gather(fire_masks, next_flat)}};
    }

}// namespace arenai::agent

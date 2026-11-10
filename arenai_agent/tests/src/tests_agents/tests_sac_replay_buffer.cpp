//
// Created by claude on 08/11/2026.
//

#include <agents/sac/sac_replay_buffer.h>
#include <gtest/gtest.h>

using namespace arenai;
using namespace arenai::agent;

namespace {

    constexpr int NB_TANKS = 2;
    constexpr int H = 4, W = 4, P = 3, A = 2, D = 2;

    // the vision is filled with the step number: a sampled pair (s, s') must read (t, t + 1)
    TorchState make_state(const int step) {
        return {
            .vision = torch::full({NB_TANKS, 3, H, W}, step, torch::kUInt8),
            .proprioception = torch::full({NB_TANKS, P}, static_cast<float>(step)),
            .fire_mask = torch::ones({NB_TANKS, 1})};
    }

    SacInputStep make_step(
        const int step, const torch::Tensor &done = torch::zeros({NB_TANKS, 1}),
        const torch::Tensor &truncated = torch::zeros({NB_TANKS, 1})) {
        return {
            .state = make_state(step),
            .action =
                {.continuous_action = torch::full({NB_TANKS, A}, static_cast<float>(step)),
                 .discrete_action = torch::ones({NB_TANKS, D})},
            .reward = torch::full({NB_TANKS, 1}, static_cast<float>(step)),
            .done = done,
            .truncated = truncated};
    }

    // every sampled transition reads its next observation one step later
    void assert_next_is_successor(const SacBatch &batch) {
        const auto steps = batch.states.vision.flatten(1).select(1, 0).to(torch::kFloat);
        const auto next_steps = batch.next_states.vision.flatten(1).select(1, 0).to(torch::kFloat);

        ASSERT_TRUE(torch::equal(next_steps, steps + 1.f));
        ASSERT_TRUE(torch::equal(batch.rewards.squeeze(-1), steps));
        ASSERT_TRUE(torch::equal(batch.actions.continuous_action.select(1, 0), steps));
    }

}// namespace

TEST(SacReplayBuffer, EmptyBufferHasNoTransition) {
    const SacReplayBuffer buffer(16);
    ASSERT_EQ(buffer.nb_transitions(), 0);
}

TEST(SacReplayBuffer, NewestStepIsNotSampleable) {
    SacReplayBuffer buffer(16);

    buffer.add(make_step(0));
    // its successor is not known yet
    ASSERT_EQ(buffer.nb_transitions(), 0);

    buffer.add(make_step(1));
    ASSERT_EQ(buffer.nb_transitions(), NB_TANKS);
}

TEST(SacReplayBuffer, NextStateIsTheFollowingSlot) {
    SacReplayBuffer buffer(16);
    for (int t = 0; t < 5; t++) buffer.add(make_step(t));

    const auto batch = buffer.sample(32, torch::kCPU);

    ASSERT_EQ(batch.states.vision.sizes(), (std::vector<int64_t>{32, 3, H, W}));
    ASSERT_EQ(batch.states.vision.dtype(), torch::kUInt8);
    ASSERT_EQ(batch.next_states.proprioception.sizes(), (std::vector<int64_t>{32, P}));
    ASSERT_EQ(batch.cuts.sizes(), (std::vector<int64_t>{32, 1}));
    ASSERT_TRUE(torch::all(batch.cuts == 0.f).item<bool>());

    assert_next_is_successor(batch);
}

TEST(SacReplayBuffer, FinishEpisodeClosesTheLastStep) {
    SacReplayBuffer buffer(16);
    buffer.add(make_step(0));
    buffer.finish_episode(make_state(1));

    // the closing observation makes the last step complete without being a transition
    ASSERT_EQ(buffer.nb_transitions(), NB_TANKS);

    const auto batch = buffer.sample(8, torch::kCPU);
    assert_next_is_successor(batch);
}

TEST(SacReplayBuffer, StepsAfterADeathAreNotTransitions) {
    SacReplayBuffer buffer(16);

    // tank 0 dies on step 0
    buffer.add(make_step(0, torch::tensor({{1.f}, {0.f}})));
    buffer.add(make_step(1));
    buffer.add(make_step(2));

    // step 0: both tanks; step 1: tank 1 only; step 2: newest
    ASSERT_EQ(buffer.nb_transitions(), 3);

    const auto batch = buffer.sample(64, torch::kCPU);
    const auto steps = batch.states.vision.flatten(1).select(1, 0).to(torch::kFloat);

    // only the death transition is cut, the others bootstrap
    for (int64_t i = 0; i < 64; i++)
        if (batch.cuts[i].item<float>() == 1.f) ASSERT_EQ(steps[i].item<float>(), 0.f);
    ASSERT_TRUE(torch::any(batch.cuts == 1.f).item<bool>());
}

TEST(SacReplayBuffer, DeathCountResetsAtEpisodeEnd) {
    SacReplayBuffer buffer(32);

    buffer.add(make_step(0, torch::tensor({{1.f}, {1.f}})));
    buffer.finish_episode(make_state(1));
    ASSERT_EQ(buffer.nb_transitions(), NB_TANKS);

    // new episode: every tank is alive again
    buffer.add(make_step(5));
    buffer.add(make_step(6));
    ASSERT_EQ(buffer.nb_transitions(), 2 * NB_TANKS);
}

TEST(SacReplayBuffer, TruncationCutsTheTarget) {
    SacReplayBuffer buffer(16);

    buffer.add(make_step(0, torch::zeros({NB_TANKS, 1}), torch::ones({NB_TANKS, 1})));
    buffer.add(make_step(1));

    const auto batch = buffer.sample(8, torch::kCPU);
    ASSERT_TRUE(torch::all(batch.cuts == 1.f).item<bool>());
}

TEST(SacReplayBuffer, RingWrapsAndKeepsSuccessors) {
    // 3 slots per tank
    SacReplayBuffer buffer(3 * NB_TANKS);
    for (int t = 0; t < 8; t++) buffer.add(make_step(t));

    // the ring holds steps 5, 6, 7 ; 7 is the newest
    ASSERT_EQ(buffer.nb_transitions(), 2 * NB_TANKS);

    const auto batch = buffer.sample(64, torch::kCPU);
    const auto steps = batch.states.vision.flatten(1).select(1, 0).to(torch::kFloat);
    ASSERT_TRUE(torch::all(steps >= 5.f).item<bool>());
    ASSERT_TRUE(torch::all(steps <= 6.f).item<bool>());

    assert_next_is_successor(batch);
}

TEST(SacReplayBuffer, CapacityBelowTwoSlotsPerTankStillRings) {
    SacReplayBuffer buffer(1);
    for (int t = 0; t < 4; t++) buffer.add(make_step(t));

    ASSERT_EQ(buffer.nb_transitions(), NB_TANKS);
    assert_next_is_successor(buffer.sample(8, torch::kCPU));
}

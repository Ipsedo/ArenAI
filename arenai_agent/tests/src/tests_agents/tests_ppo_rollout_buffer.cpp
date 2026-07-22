//
// Created by claude on 22/07/2026.
//

#include <arenai_agent_tests/tests_agents/tests_ppo_rollout_buffer.h>

using namespace arenai;
using namespace arenai::agent;

// ========================================================================
// Fixture helpers
// ========================================================================

TorchState PpoRolloutBufferTest::make_state() {
    return {
        torch::randn({NB_TANKS, 3, VISION_SIZE, VISION_SIZE}),
        torch::randn({NB_TANKS, NB_SENSORS})};
}

PpoInputStep PpoRolloutBufferTest::make_step(const TorchState &state, const torch::Tensor &done) {
    return {
        .state = state,
        .action =
            {.continuous_action = torch::randn({NB_TANKS, NB_CONTINUOUS_ACTIONS}),
             .discrete_action = torch::eye(NB_DISCRETE_ACTIONS)
                                    .index_select(
                                        0, torch::randint(
                                               NB_DISCRETE_ACTIONS, {NB_TANKS},
                                               torch::TensorOptions().dtype(torch::kInt64)))},
        .continuous_log_prob = torch::randn({NB_TANKS, 1}),
        .discrete_log_prob = torch::randn({NB_TANKS, 1}),
        .reward = torch::randn({NB_TANKS, 1}),
        .done = done,
        .truncated = torch::zeros({NB_TANKS, 1})};
}

PpoInputStep PpoRolloutBufferTest::make_step(const TorchState &state) {
    return make_step(state, torch::zeros({NB_TANKS, 1}));
}

// ========================================================================
// Completion counting
// ========================================================================

TEST_F(PpoRolloutBufferTest, EmptyBufferHasNoCompleteStep) {
    PpoRolloutBuffer buffer;

    ASSERT_EQ(buffer.nb_complete_steps(), 0);
    ASSERT_THROW(buffer.get_rollout(), c10::Error);
}

TEST_F(PpoRolloutBufferTest, LastAddedStepStaysPending) {
    PpoRolloutBuffer buffer;

    buffer.add(make_step(make_state()));
    ASSERT_EQ(buffer.nb_complete_steps(), 0);

    buffer.add(make_step(make_state()));
    ASSERT_EQ(buffer.nb_complete_steps(), 1);
}

TEST_F(PpoRolloutBufferTest, FinishEpisodeCompletesPendingStep) {
    PpoRolloutBuffer buffer;

    buffer.add(make_step(make_state()));
    buffer.add(make_step(make_state()));
    buffer.finish_episode(make_state());

    ASSERT_EQ(buffer.nb_complete_steps(), 2);
}

TEST_F(PpoRolloutBufferTest, GetRolloutKeepsPendingStep) {
    PpoRolloutBuffer buffer;

    buffer.add(make_step(make_state()));
    buffer.add(make_step(make_state()));
    buffer.add(make_step(make_state()));

    const auto rollout = buffer.get_rollout();
    ASSERT_EQ(rollout.rewards.size(0), 2);

    // the pending step stays and is closed by the next observation
    ASSERT_EQ(buffer.nb_complete_steps(), 0);
    buffer.add(make_step(make_state()));
    ASSERT_EQ(buffer.nb_complete_steps(), 1);
}

// ========================================================================
// Rollout content
// ========================================================================

TEST_F(PpoRolloutBufferTest, RolloutShapes) {
    PpoRolloutBuffer buffer;

    constexpr int nb_steps = 3;
    for (int t = 0; t < nb_steps; t++) buffer.add(make_step(make_state()));
    buffer.finish_episode(make_state());

    const auto rollout = buffer.get_rollout();

    const auto expected_vision =
        std::vector<int64_t>{nb_steps, NB_TANKS, 3, VISION_SIZE, VISION_SIZE};
    ASSERT_EQ(rollout.states.vision.sizes().vec(), expected_vision);
    ASSERT_EQ(rollout.next_states.vision.sizes().vec(), expected_vision);

    const auto expected_scalar = std::vector<int64_t>{nb_steps, NB_TANKS, 1};
    ASSERT_EQ(rollout.rewards.sizes().vec(), expected_scalar);
    ASSERT_EQ(rollout.continuous_log_probs.sizes().vec(), expected_scalar);
    ASSERT_EQ(rollout.discrete_log_probs.sizes().vec(), expected_scalar);
    ASSERT_EQ(rollout.valids.sizes().vec(), expected_scalar);

    ASSERT_EQ(
        rollout.actions.continuous_action.sizes().vec(),
        (std::vector<int64_t>{nb_steps, NB_TANKS, NB_CONTINUOUS_ACTIONS}));
}

TEST_F(PpoRolloutBufferTest, NextStateIsFollowingObservation) {
    PpoRolloutBuffer buffer;

    const auto state_0 = make_state();
    const auto state_1 = make_state();
    const auto final_state = make_state();

    buffer.add(make_step(state_0));
    buffer.add(make_step(state_1));
    buffer.finish_episode(final_state);

    const auto rollout = buffer.get_rollout();

    ASSERT_TRUE(torch::allclose(rollout.next_states.vision[0], state_1.vision));
    ASSERT_TRUE(torch::allclose(rollout.next_states.vision[1], final_state.vision));
}

// ========================================================================
// Validity mask
// ========================================================================

TEST_F(PpoRolloutBufferTest, TerminatedTankInvalidatesFollowingSteps) {
    PpoRolloutBuffer buffer;

    // tank 0 dies at the first step
    const auto done = torch::cat({torch::ones({1, 1}), torch::zeros({1, 1})}, 0);

    buffer.add(make_step(make_state(), done));
    buffer.add(make_step(make_state()));
    buffer.add(make_step(make_state()));
    buffer.finish_episode(make_state());

    const auto valids = buffer.get_rollout().valids.squeeze(-1);

    // the dying transition itself is valid, the following ones are not for tank 0
    ASSERT_TRUE(valids[0][0].item<bool>());
    ASSERT_FALSE(valids[1][0].item<bool>());
    ASSERT_FALSE(valids[2][0].item<bool>());

    // tank 1 stays valid the whole rollout
    ASSERT_TRUE(valids[0][1].item<bool>());
    ASSERT_TRUE(valids[1][1].item<bool>());
    ASSERT_TRUE(valids[2][1].item<bool>());
}

TEST_F(PpoRolloutBufferTest, FinishEpisodeResetsTermination) {
    PpoRolloutBuffer buffer;

    buffer.add(make_step(make_state(), torch::ones({NB_TANKS, 1})));
    buffer.finish_episode(make_state());

    // new episode: every tank is alive again
    buffer.add(make_step(make_state()));
    buffer.finish_episode(make_state());

    const auto valids = buffer.get_rollout().valids.squeeze(-1);

    ASSERT_TRUE(torch::all(valids[0]).item<bool>());
    ASSERT_TRUE(torch::all(valids[1]).item<bool>());
}

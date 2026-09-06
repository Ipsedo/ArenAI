//
// Created by samuel on 06/09/2026.
//

#include <arenai_agent_tests/tests_agents/tests_liquid_ppo_rollout_buffer.h>

using namespace arenai;
using namespace arenai::agent;

// ========================================================================
// Fixture helpers
// ========================================================================

TorchState LiquidPpoRolloutBufferTest::make_state() {
    return {
        .vision = torch::randn({NB_TANKS, 3, VISION_SIZE, VISION_SIZE}),
        .proprioception = torch::randn({NB_TANKS, NB_SENSORS})};
}

LiquidPpoInputStep LiquidPpoRolloutBufferTest::make_step(
    const TorchState &state, const torch::Tensor &done, const bool episode_start) {
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
        .actor_hidden = torch::randn({NB_TANKS, NEURON_NUMBER}),
        .episode_start = episode_start};
}

LiquidPpoInputStep LiquidPpoRolloutBufferTest::make_step(const TorchState &state) {
    return make_step(state, torch::zeros({NB_TANKS, 1}), false);
}

// ========================================================================
// Completion counting
// ========================================================================

TEST_F(LiquidPpoRolloutBufferTest, EmptyBufferHasNoCompleteStep) {
    LiquidPpoRolloutBuffer buffer;

    ASSERT_EQ(buffer.nb_complete_steps(), 0);
    ASSERT_THROW(buffer.get_rollout(), c10::Error);
}

TEST_F(LiquidPpoRolloutBufferTest, LastAddedStepStaysPending) {
    LiquidPpoRolloutBuffer buffer;

    buffer.add(make_step(make_state()));
    ASSERT_EQ(buffer.nb_complete_steps(), 0);

    buffer.add(make_step(make_state()));
    ASSERT_EQ(buffer.nb_complete_steps(), 1);
}

TEST_F(LiquidPpoRolloutBufferTest, FinishEpisodeCompletesPendingStep) {
    LiquidPpoRolloutBuffer buffer;

    buffer.add(make_step(make_state()));
    buffer.add(make_step(make_state()));
    buffer.finish_episode(make_state());

    ASSERT_EQ(buffer.nb_complete_steps(), 2);
}

TEST_F(LiquidPpoRolloutBufferTest, GetRolloutKeepsPendingStep) {
    LiquidPpoRolloutBuffer buffer;

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

TEST_F(LiquidPpoRolloutBufferTest, RolloutShapes) {
    LiquidPpoRolloutBuffer buffer;

    constexpr int nb_steps = 3;
    for (int t = 0; t < nb_steps; t++) buffer.add(make_step(make_state()));
    buffer.finish_episode(make_state());

    const auto rollout = buffer.get_rollout();

    const auto expected_vision =
        std::vector<int64_t>{nb_steps, NB_TANKS, 3, VISION_SIZE, VISION_SIZE};
    ASSERT_EQ(rollout.states.vision.sizes().vec(), expected_vision);

    // the bootstrap state has no time dimension
    ASSERT_EQ(
        rollout.bootstrap_state.vision.sizes().vec(),
        (std::vector<int64_t>{NB_TANKS, 3, VISION_SIZE, VISION_SIZE}));

    const auto expected_scalar = std::vector<int64_t>{nb_steps, NB_TANKS, 1};
    ASSERT_EQ(rollout.rewards.sizes().vec(), expected_scalar);
    ASSERT_EQ(rollout.continuous_log_probs.sizes().vec(), expected_scalar);
    ASSERT_EQ(rollout.discrete_log_probs.sizes().vec(), expected_scalar);
    ASSERT_EQ(rollout.valids.sizes().vec(), expected_scalar);

    ASSERT_EQ(
        rollout.actions.continuous_action.sizes().vec(),
        (std::vector<int64_t>{nb_steps, NB_TANKS, NB_CONTINUOUS_ACTIONS}));

    // liquid extras: the per-step actor states and the episode-start flags
    ASSERT_EQ(
        rollout.actor_hiddens.sizes().vec(),
        (std::vector<int64_t>{nb_steps, NB_TANKS, NEURON_NUMBER}));
    ASSERT_EQ(rollout.episode_starts.sizes().vec(), (std::vector<int64_t>{nb_steps}));
    ASSERT_EQ(rollout.episode_starts.dtype(), torch::kBool);
}

TEST_F(LiquidPpoRolloutBufferTest, ActorHiddenRoundTrip) {
    LiquidPpoRolloutBuffer buffer;

    const auto step_0 = make_step(make_state());
    const auto step_1 = make_step(make_state());

    buffer.add(step_0);
    buffer.add(step_1);
    buffer.finish_episode(make_state());

    const auto rollout = buffer.get_rollout();

    ASSERT_TRUE(torch::allclose(rollout.actor_hiddens[0], step_0.actor_hidden));
    ASSERT_TRUE(torch::allclose(rollout.actor_hiddens[1], step_1.actor_hidden));
}

TEST_F(LiquidPpoRolloutBufferTest, EpisodeStartFlagsRoundTrip) {
    LiquidPpoRolloutBuffer buffer;

    const auto zeros = torch::zeros({NB_TANKS, 1});
    buffer.add(make_step(make_state(), zeros, true));
    buffer.add(make_step(make_state(), zeros, false));
    buffer.add(make_step(make_state(), zeros, true));
    buffer.finish_episode(make_state());

    const auto episode_starts = buffer.get_rollout().episode_starts;

    ASSERT_TRUE(episode_starts[0].item<bool>());
    ASSERT_FALSE(episode_starts[1].item<bool>());
    ASSERT_TRUE(episode_starts[2].item<bool>());
}

TEST_F(LiquidPpoRolloutBufferTest, BootstrapStateIsEpisodeFinalObservation) {
    LiquidPpoRolloutBuffer buffer;

    const auto state_0 = make_state();
    const auto state_1 = make_state();
    const auto final_state = make_state();

    buffer.add(make_step(state_0));
    buffer.add(make_step(state_1));
    buffer.finish_episode(final_state);

    const auto rollout = buffer.get_rollout();

    ASSERT_EQ(rollout.rewards.size(0), 2);
    ASSERT_TRUE(torch::allclose(rollout.bootstrap_state.vision, final_state.vision));
}

TEST_F(LiquidPpoRolloutBufferTest, BootstrapStateIsPendingObservationMidEpisode) {
    LiquidPpoRolloutBuffer buffer;

    const auto state_0 = make_state();
    const auto state_1 = make_state();

    buffer.add(make_step(state_0));
    buffer.add(make_step(state_1));

    const auto rollout = buffer.get_rollout();

    // only the first step is complete; the pending step's own observation closes it
    ASSERT_EQ(rollout.rewards.size(0), 1);
    ASSERT_TRUE(torch::allclose(rollout.bootstrap_state.vision, state_1.vision));
}

// ========================================================================
// Validity mask
// ========================================================================

TEST_F(LiquidPpoRolloutBufferTest, TerminatedTankInvalidatesFollowingSteps) {
    LiquidPpoRolloutBuffer buffer;

    // tank 0 dies at the first step
    const auto done = torch::cat({torch::ones({1, 1}), torch::zeros({1, 1})}, 0);

    buffer.add(make_step(make_state(), done, false));
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

TEST_F(LiquidPpoRolloutBufferTest, FinishEpisodeResetsTermination) {
    LiquidPpoRolloutBuffer buffer;

    buffer.add(make_step(make_state(), torch::ones({NB_TANKS, 1}), false));
    buffer.finish_episode(make_state());

    // new episode: every tank is alive again
    buffer.add(make_step(make_state()));
    buffer.finish_episode(make_state());

    const auto valids = buffer.get_rollout().valids.squeeze(-1);

    ASSERT_TRUE(torch::all(valids[0]).item<bool>());
    ASSERT_TRUE(torch::all(valids[1]).item<bool>());
}

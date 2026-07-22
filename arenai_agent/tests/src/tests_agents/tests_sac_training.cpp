//
// Created by claude on 01/07/2026.
//

#include <arenai_agent_tests/tests_agents/tests_sac_training.h>

using namespace arenai;
using namespace arenai::agent;

std::unique_ptr<SacTorchAgentFactory>
SacTrainingTest::make_factory(const SacTrainingTestConfig &cfg) const {
    const SacHyperParams params{
        .actor_learning_rate = 1e-3f,
        .critic_learning_rate = 1e-3f,
        .alpha_learning_rate = 1e-3f,
        .hidden_size_sensors = 8,
        .hidden_size_actions = 8,
        .actor_hidden_sizes = {16},
        .critic_hidden_sizes = {16},
        .vision_channels = {{3, 4}},
        .group_norm_nums = {2},
        .metric_window_size = 10,
        .tau = 0.005f,
        .gamma = 0.99f,
        .replay_buffer_size = 10,
        .train_every = 1,
        .epochs = 1,
        .batch_size = 1};

    return std::make_unique<SacTorchAgentFactory>(
        cfg.vision_height, cfg.vision_width, cfg.nb_sensors, cfg.nb_continuous_actions,
        cfg.nb_discrete_actions, device, params);
}

TorchState SacTrainingTest::make_state(const SacTrainingTestConfig &cfg) {
    return {
        torch::randint(0, 255, {1, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8),
        torch::randn({1, cfg.nb_sensors})};
}

TEST_F(SacTrainingTest, ActProducesValidOutput) {
    constexpr SacTrainingTestConfig cfg{8, 8, 3, 2, 3};
    const auto factory = make_factory(cfg);

    const auto [continuous_action, discrete_action] = factory->get_agent()->act(make_state(cfg));

    ASSERT_EQ(continuous_action.size(0), 1);
    ASSERT_EQ(continuous_action.size(1), 2);
    ASSERT_EQ(discrete_action.size(0), 1);
    ASSERT_EQ(discrete_action.size(1), 3);

    ASSERT_TRUE(torch::all(torch::isfinite(continuous_action)).item<bool>());
    ASSERT_TRUE(torch::all(torch::isfinite(discrete_action)).item<bool>());
}

TEST_F(SacTrainingTest, CountParametersPositive) {
    constexpr SacTrainingTestConfig cfg{8, 8, 3, 2, 3};
    const auto factory = make_factory(cfg);

    ASSERT_GT(factory->get_trainer()->count_parameters(), 0)
        << "Agent should have a positive number of parameters";
}

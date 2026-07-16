//
// Created by claude on 01/07/2026.
//

#include <agents/sac.h>

#include <arenai_train_tests/tests_agents/tests_sac_training.h>

#include "../tests_replay_buffer/create_random_step.h"

using namespace arenai;
using namespace arenai::train;

std::unique_ptr<SacAgent> SacTrainingTest::make_agent(const SacTrainingTestConfig &cfg) const {
    return std::make_unique<SacAgent>(
        cfg.vision_height, cfg.vision_width, cfg.nb_sensors, cfg.nb_continuous_actions,
        cfg.nb_discrete_actions, 1e-3f, 1e-3f, 1e-3f, 8, 8, std::vector<int>{16},
        std::vector<int>{16}, std::vector<std::tuple<int, int>>{{3, 4}}, std::vector<int>{2},
        device, 10, 0.005f, 0.99f);
}

std::unique_ptr<ReplayBuffer>
SacTrainingTest::make_filled_buffer(const SacTrainingTestConfig &cfg, const int n_steps) {
    auto buffer = std::make_unique<ReplayBuffer>(n_steps);
    for (int i = 0; i < n_steps; i++) {
        buffer->add(create_random_step(
            cfg.vision_width, cfg.vision_height, cfg.nb_continuous_actions, cfg.nb_discrete_actions,
            cfg.nb_sensors, i == n_steps - 1));
    }
    return buffer;
}

TEST_F(SacTrainingTest, TrainStepDoesNotCrash) {
    constexpr SacTrainingTestConfig cfg{8, 8, 3, 2, 3};
    const auto agent = make_agent(cfg);
    const auto buffer = make_filled_buffer(cfg, 32);

    ASSERT_NO_THROW(agent->train(buffer, 1, 8)) << "Single training step should not crash";
}

TEST_F(SacTrainingTest, MultipleTrainStepsDoNotCrash) {
    constexpr SacTrainingTestConfig cfg{8, 8, 3, 2, 3};
    const auto agent = make_agent(cfg);
    const auto buffer = make_filled_buffer(cfg, 64);

    ASSERT_NO_THROW({
        agent->train(buffer, 3, 16);
        agent->train(buffer, 3, 16);
    }) << "Multiple training steps should not crash";
}

TEST_F(SacTrainingTest, MetricsReturnedNonEmpty) {
    constexpr SacTrainingTestConfig cfg{8, 8, 3, 2, 3};
    const auto agent = make_agent(cfg);
    const auto buffer = make_filled_buffer(cfg, 32);

    agent->train(buffer, 2, 8);

    const auto metrics = agent->get_metrics();

    ASSERT_FALSE(metrics.empty()) << "Agent should expose metrics after training";

    for (const auto &m: metrics) {
        ASSERT_NE(m, nullptr);
        ASSERT_FALSE(m->get_name().empty());
    }
}

TEST_F(SacTrainingTest, ActProducesValidOutput) {
    constexpr SacTrainingTestConfig cfg{8, 8, 3, 2, 3};
    const auto agent = make_agent(cfg);

    const auto vision = torch::randint(0, 255, {1, 3, 8, 8}, torch::kUInt8);
    const auto sensors = torch::randn({1, 3});

    agent->set_train(false);
    const auto [continuous_action, discrete_action] = agent->act(vision, sensors);

    ASSERT_EQ(continuous_action.size(0), 1);
    ASSERT_EQ(continuous_action.size(1), 2);
    ASSERT_EQ(discrete_action.size(0), 1);
    ASSERT_EQ(discrete_action.size(1), 3);

    ASSERT_TRUE(torch::all(torch::isfinite(continuous_action)).item<bool>());
    ASSERT_TRUE(torch::all(torch::isfinite(discrete_action)).item<bool>());
}

TEST_F(SacTrainingTest, MetricsHaveValuesAfterTraining) {
    constexpr SacTrainingTestConfig cfg{8, 8, 3, 2, 3};
    const auto agent = make_agent(cfg);
    const auto buffer = make_filled_buffer(cfg, 32);

    agent->train(buffer, 3, 8);

    for (const auto metrics = agent->get_metrics(); const auto &m: metrics) {
        const auto val = m->compute_metric();
        ASSERT_TRUE(std::isfinite(val))
            << "Metric '" << m->get_name() << "' should be finite after training";
    }
}

TEST_F(SacTrainingTest, CountParametersPositive) {
    constexpr SacTrainingTestConfig cfg{8, 8, 3, 2, 3};
    const auto agent = make_agent(cfg);

    ASSERT_GT(agent->count_parameters(), 0) << "Agent should have a positive number of parameters";
}

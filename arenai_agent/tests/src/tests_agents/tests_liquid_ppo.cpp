//
// Created by samuel on 06/09/2026.
//

#include <arenai_agent_tests/tests_agents/tests_liquid_ppo.h>

using namespace arenai;
using namespace arenai::agent;

// ========================================================================
// Fixture helpers
// ========================================================================

void LiquidPpoAgentTest::SetUp() {
    tmp_dir = std::filesystem::temp_directory_path() / "arenai_test_liquid_ppo";
    std::filesystem::create_directories(tmp_dir);
}

void LiquidPpoAgentTest::TearDown() { std::filesystem::remove_all(tmp_dir); }

std::unique_ptr<LiquidPpoTorchAgentFactory>
LiquidPpoAgentTest::make_factory(const LiquidPpoTestConfig &cfg) const {
    const LiquidPpoHyperParams params{
        .actor_learning_rate = 1e-3f,
        .critic_learning_rate = 1e-3f,
        .hidden_size_sensors = 16,
        .vision_channels = {{3, 8}},
        .group_norm_nums = {4},
        .neuron_number = 16,
        .unfolding_steps = 2,
        .chunk_size = 2,
        .metric_window_size = 10,
        .gamma = 0.99f,
        .gae_lambda = 0.95f,
        .clip_epsilon = 0.2f,
        .grad_norm_max = 1.f,
        .epochs = 1,
        .rollout_size = 8,
        .minibatch_size = 10};

    return std::make_unique<LiquidPpoTorchAgentFactory>(
        cfg.vision_height, cfg.vision_width, cfg.nb_sensors, cfg.nb_continuous_actions,
        cfg.nb_discrete_actions, device, params);
}

TorchState LiquidPpoAgentTest::make_state(const LiquidPpoTestConfig &cfg, const int batch) {
    return {
        .vision =
            torch::randint(0, 255, {batch, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8),
        .proprioception = torch::randn({batch, cfg.nb_sensors})};
}

// ========================================================================
// Fixed tests
// ========================================================================

TEST_F(LiquidPpoAgentTest, ParameterCountPositive) {
    constexpr LiquidPpoTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 10,
        .nb_continuous_actions = 4,
        .nb_discrete_actions = 2};
    const auto factory = make_factory(cfg);

    ASSERT_GT(factory->get_trainer()->count_parameters(), 0);
}

TEST_F(LiquidPpoAgentTest, MetricsNotEmpty) {
    constexpr LiquidPpoTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 10,
        .nb_continuous_actions = 4,
        .nb_discrete_actions = 2};
    const auto factory = make_factory(cfg);

    const auto metrics = factory->get_trainer()->get_metrics();

    ASSERT_EQ(metrics.size(), 10);
}

// ========================================================================
// Parameterized: act shape tests
// ========================================================================

TEST_P(LiquidPpoActShapeParamTest, ActOutputShapes) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    constexpr int batch = 4;
    const auto [continuous_action, discrete_action] =
        factory->get_agent()->act(make_state(cfg, batch), true);

    ASSERT_EQ(continuous_action.size(0), batch);
    ASSERT_EQ(continuous_action.size(1), cfg.nb_continuous_actions);

    ASSERT_EQ(discrete_action.size(0), batch);
    ASSERT_EQ(discrete_action.size(1), cfg.nb_discrete_actions);
}

TEST_P(LiquidPpoActShapeParamTest, ActContinuousFinite) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    constexpr int batch = 4;
    const auto [continuous_action, discrete_action] =
        factory->get_agent()->act(make_state(cfg, batch), true);

    ASSERT_TRUE(torch::all(torch::isfinite(continuous_action)).item<bool>());
}

TEST_P(LiquidPpoActShapeParamTest, ActDiscreteIsOneHot) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    constexpr int batch = 4;
    const auto [continuous_action, discrete_action] =
        factory->get_agent()->act(make_state(cfg, batch), true);

    const auto row_sums = torch::sum(discrete_action, -1);
    ASSERT_TRUE(torch::allclose(row_sums, torch::ones({batch})));

    const auto is_binary =
        torch::logical_or(torch::eq(discrete_action, 0.0f), torch::eq(discrete_action, 1.0f));
    ASSERT_TRUE(torch::all(is_binary).item<bool>());
}

TEST_P(LiquidPpoActShapeParamTest, ActKeepsShapesOverConsecutiveSteps) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    // the liquid state is carried across the calls: every step must stay consistent
    constexpr int batch = 4;
    for (int t = 0; t < 3; t++) {
        const auto [continuous_action, discrete_action] =
            factory->get_agent()->act(make_state(cfg, batch), true);

        ASSERT_EQ(continuous_action.size(0), batch);
        ASSERT_EQ(continuous_action.size(1), cfg.nb_continuous_actions);
        ASSERT_TRUE(torch::all(torch::isfinite(continuous_action)).item<bool>());

        ASSERT_EQ(discrete_action.size(0), batch);
        ASSERT_EQ(discrete_action.size(1), cfg.nb_discrete_actions);
    }
}

INSTANTIATE_TEST_SUITE_P(
    LiquidPpoAgent, LiquidPpoActShapeParamTest,
    testing::Values(
        LiquidPpoTestConfig{8, 8, 10, 4, 2}, LiquidPpoTestConfig{8, 8, 5, 2, 3},
        LiquidPpoTestConfig{16, 16, 20, 6, 4}, LiquidPpoTestConfig{8, 12, 10, 4, 2}));

// ========================================================================
// Parameterized: save / load tests
// ========================================================================

TEST_P(LiquidPpoSaveLoadParamTest, SaveCreatesExpectedFiles) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    const auto save_dir = tmp_dir / "save_test";
    std::filesystem::create_directories(save_dir);

    factory->get_trainer()->save(save_dir);

    const std::vector<std::string> expected_files = {
        "actor.pt",        "critic.pt",      "actor_optim.pt",
        "critic_optim.pt", "actor_repr.txt", "critic_repr.txt",
    };

    for (const auto &f: expected_files)
        ASSERT_TRUE(std::filesystem::exists(save_dir / f)) << "Missing file: " << f;
}

TEST_P(LiquidPpoSaveLoadParamTest, SavedFilesNonEmpty) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    const auto save_dir = tmp_dir / "save_nonempty";
    std::filesystem::create_directories(save_dir);

    factory->get_trainer()->save(save_dir);

    for (const auto &entry: std::filesystem::directory_iterator(save_dir)) {
        if (entry.is_regular_file())
            ASSERT_GT(entry.file_size(), 0u) << "Empty file: " << entry.path().filename();
    }
}

INSTANTIATE_TEST_SUITE_P(
    LiquidPpoAgent, LiquidPpoSaveLoadParamTest,
    testing::Values(
        LiquidPpoTestConfig{8, 8, 10, 4, 2}, LiquidPpoTestConfig{8, 8, 5, 2, 3},
        LiquidPpoTestConfig{16, 16, 20, 6, 4}));

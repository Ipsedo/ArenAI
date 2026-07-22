//
// Created by claude on 22/07/2026.
//

#include <arenai_agent_tests/tests_agents/tests_ppo.h>

using namespace arenai;
using namespace arenai::agent;

// ========================================================================
// Fixture helpers
// ========================================================================

void PpoAgentTest::SetUp() {
    tmp_dir = std::filesystem::temp_directory_path() / "arenai_test_ppo";
    std::filesystem::create_directories(tmp_dir);
}

void PpoAgentTest::TearDown() { std::filesystem::remove_all(tmp_dir); }

std::unique_ptr<PpoTorchAgentFactory> PpoAgentTest::make_factory(const PpoTestConfig &cfg) const {
    const PpoHyperParams params{
        .actor_learning_rate = 1e-3f,
        .critic_learning_rate = 1e-3f,
        .hidden_size_sensors = 16,
        .hidden_size_actions = 16,
        .actor_hidden_sizes = {32},
        .critic_hidden_sizes = {32},
        .vision_channels = {{3, 8}},
        .group_norm_nums = {4},
        .metric_window_size = 10,
        .gamma = 0.99f,
        .gae_lambda = 0.95f,
        .clip_epsilon = 0.2f,
        .continuous_entropy_coef = 0.01f,
        .discrete_entropy_coef = 0.01f,
        .epochs = 1,
        .rollout_size = 8};

    return std::make_unique<PpoTorchAgentFactory>(
        cfg.vision_height, cfg.vision_width, cfg.nb_sensors, cfg.nb_continuous_actions,
        cfg.nb_discrete_actions, device, params);
}

TorchState PpoAgentTest::make_state(const PpoTestConfig &cfg, const int batch) {
    return {
        torch::randint(0, 255, {batch, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8),
        torch::randn({batch, cfg.nb_sensors})};
}

// ========================================================================
// Fixed tests
// ========================================================================

TEST_F(PpoAgentTest, ParameterCountPositive) {
    constexpr PpoTestConfig cfg{8, 8, 10, 4, 2};
    const auto factory = make_factory(cfg);

    ASSERT_GT(factory->get_trainer()->count_parameters(), 0);
}

TEST_F(PpoAgentTest, MetricsNotEmpty) {
    constexpr PpoTestConfig cfg{8, 8, 10, 4, 2};
    const auto factory = make_factory(cfg);

    const auto metrics = factory->get_trainer()->get_metrics();

    ASSERT_EQ(metrics.size(), 7);
}

// ========================================================================
// Parameterized: act shape tests
// ========================================================================

TEST_P(PpoActShapeParamTest, ActOutputShapes) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    constexpr int batch = 4;
    const auto [continuous_action, discrete_action] =
        factory->get_agent()->act(make_state(cfg, batch));

    ASSERT_EQ(continuous_action.size(0), batch);
    ASSERT_EQ(continuous_action.size(1), cfg.nb_continuous_actions);

    ASSERT_EQ(discrete_action.size(0), batch);
    ASSERT_EQ(discrete_action.size(1), cfg.nb_discrete_actions);
}

TEST_P(PpoActShapeParamTest, ActContinuousFinite) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    constexpr int batch = 4;
    const auto [continuous_action, discrete_action] =
        factory->get_agent()->act(make_state(cfg, batch));

    ASSERT_TRUE(torch::all(torch::isfinite(continuous_action)).item<bool>());
}

TEST_P(PpoActShapeParamTest, ActDiscreteIsOneHot) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    constexpr int batch = 4;
    const auto [continuous_action, discrete_action] =
        factory->get_agent()->act(make_state(cfg, batch));

    const auto row_sums = torch::sum(discrete_action, -1);
    ASSERT_TRUE(torch::allclose(row_sums, torch::ones({batch})));

    const auto is_binary =
        torch::logical_or(torch::eq(discrete_action, 0.0f), torch::eq(discrete_action, 1.0f));
    ASSERT_TRUE(torch::all(is_binary).item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    PpoAgent, PpoActShapeParamTest,
    testing::Values(
        PpoTestConfig{8, 8, 10, 4, 2}, PpoTestConfig{8, 8, 5, 2, 3},
        PpoTestConfig{16, 16, 20, 6, 4}, PpoTestConfig{8, 12, 10, 4, 2}));

// ========================================================================
// Parameterized: save / load tests
// ========================================================================

TEST_P(PpoSaveLoadParamTest, SaveCreatesExpectedFiles) {
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

TEST_P(PpoSaveLoadParamTest, SavedFilesNonEmpty) {
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
    PpoAgent, PpoSaveLoadParamTest,
    testing::Values(
        PpoTestConfig{8, 8, 10, 4, 2}, PpoTestConfig{8, 8, 5, 2, 3},
        PpoTestConfig{16, 16, 20, 6, 4}));

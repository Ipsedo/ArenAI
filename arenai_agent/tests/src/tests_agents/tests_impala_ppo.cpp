//
// Created by claude on 26/10/2026.
//

#include <arenai_agent_tests/tests_agents/tests_impala_ppo.h>

using namespace arenai;
using namespace arenai::agent;

// ========================================================================
// Fixture helpers
// ========================================================================

void ImpalaPpoAgentTest::SetUp() {
    tmp_dir = std::filesystem::temp_directory_path() / "arenai_test_impala_ppo";
    std::filesystem::create_directories(tmp_dir);
}

void ImpalaPpoAgentTest::TearDown() { std::filesystem::remove_all(tmp_dir); }

std::unique_ptr<ImpalaPpoTorchAgentFactory>
ImpalaPpoAgentTest::make_factory(const ImpalaPpoTestConfig &cfg) const {
    const ImpalaPpoHyperParams params{
        .actor_learning_rate = 1e-3f,
        .critic_learning_rate = 1e-3f,
        .hidden_size_sensors = 16,
        .actor_hidden_sizes = {32},
        .critic_hidden_sizes = {32},
        .vision_channels = {{3, 8}},
        .metric_window_size = 10,
        .gamma = 0.99f,
        .gae_lambda = 0.95f,
        .clip_epsilon = 0.2f,
        .grad_norm_max = 1.f,
        .epochs = 1,
        .rollout_size = 8,
        .minibatch_size = 10};

    return std::make_unique<ImpalaPpoTorchAgentFactory>(
        cfg.vision_height, cfg.vision_width, cfg.nb_sensors, cfg.nb_continuous_actions,
        cfg.nb_discrete_actions, device, params);
}

TorchState ImpalaPpoAgentTest::make_state(const ImpalaPpoTestConfig &cfg, const int batch) {
    return {
        .vision =
            torch::randint(0, 255, {batch, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8),
        .proprioception = torch::randn({batch, cfg.nb_sensors}),
        .fire_mask = torch::ones({batch, 1})};
}

// ========================================================================
// Fixed tests
// ========================================================================

TEST_F(ImpalaPpoAgentTest, ParameterCountPositive) {
    constexpr ImpalaPpoTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 10,
        .nb_continuous_actions = 4,
        .nb_discrete_actions = 2};
    const auto factory = make_factory(cfg);

    ASSERT_GT(factory->get_trainer()->count_parameters(), 0);
}

TEST_F(ImpalaPpoAgentTest, MetricsNotEmpty) {
    constexpr ImpalaPpoTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 10,
        .nb_continuous_actions = 4,
        .nb_discrete_actions = 2};
    const auto factory = make_factory(cfg);

    const auto metrics = factory->get_trainer()->get_metrics();

    // 8 trainer metrics + one entropy per action dimension
    ASSERT_EQ(metrics.size(), 8);
}

// ========================================================================
// Parameterized: act shape tests
// ========================================================================

TEST_P(ImpalaPpoActShapeParamTest, ActOutputShapes) {
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

TEST_P(ImpalaPpoActShapeParamTest, ActContinuousFinite) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    constexpr int batch = 4;
    const auto [continuous_action, discrete_action] =
        factory->get_agent()->act(make_state(cfg, batch), true);

    ASSERT_TRUE(torch::all(torch::isfinite(continuous_action)).item<bool>());
}

TEST_P(ImpalaPpoActShapeParamTest, ActDiscreteIsBinary) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    constexpr int batch = 4;
    const auto [continuous_action, discrete_action] =
        factory->get_agent()->act(make_state(cfg, batch), true);

    // independent Bernoulli actions: each entry is 0 or 1, no one-hot constraint
    const auto is_binary =
        torch::logical_or(torch::eq(discrete_action, 0.0f), torch::eq(discrete_action, 1.0f));
    ASSERT_TRUE(torch::all(is_binary).item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    ImpalaPpoAgent, ImpalaPpoActShapeParamTest,
    testing::Values(
        ImpalaPpoTestConfig{8, 8, 10, 4, 2}, ImpalaPpoTestConfig{8, 8, 5, 2, 2},
        ImpalaPpoTestConfig{16, 16, 20, 6, 2}, ImpalaPpoTestConfig{8, 12, 10, 4, 2}));

// ========================================================================
// Parameterized: save / load tests
// ========================================================================

TEST_P(ImpalaPpoSaveLoadParamTest, SaveCreatesExpectedFiles) {
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

TEST_P(ImpalaPpoSaveLoadParamTest, SavedFilesNonEmpty) {
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
    ImpalaPpoAgent, ImpalaPpoSaveLoadParamTest,
    testing::Values(
        ImpalaPpoTestConfig{8, 8, 10, 4, 2}, ImpalaPpoTestConfig{8, 8, 5, 2, 2},
        ImpalaPpoTestConfig{16, 16, 20, 6, 2}));

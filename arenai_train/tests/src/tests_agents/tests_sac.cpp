//
// Created by samuel on 30/06/2026.
//

#include <arenai_train_tests/tests_agents/tests_sac.h>

using namespace arenai;
using namespace arenai::train;

// ========================================================================
// Fixture helpers
// ========================================================================

void SacAgentTest::SetUp() {
    tmp_dir = std::filesystem::temp_directory_path() / "arenai_test_sac";
    std::filesystem::create_directories(tmp_dir);
}

void SacAgentTest::TearDown() { std::filesystem::remove_all(tmp_dir); }

std::unique_ptr<SacTorchAgentFactory> SacAgentTest::make_factory(const SacTestConfig &cfg) const {
    const std::vector<int> actor_hidden{32};
    const std::vector<int> critic_hidden{32};
    const std::vector<std::tuple<int, int>> vision_channels{{3, 8}};
    const std::vector<int> group_norm_nums{4};

    return std::make_unique<SacTorchAgentFactory>(
        cfg.vision_height, cfg.vision_width, cfg.nb_sensors, cfg.nb_continuous_actions,
        cfg.nb_discrete_actions, 1e-3f, 1e-3f, 1e-3f, 16, 16, actor_hidden, critic_hidden,
        vision_channels, group_norm_nums, device, 10, 0.005f, 0.99f, 10, 1, 1, 1);
}

TorchState SacAgentTest::make_state(const SacTestConfig &cfg, const int batch) {
    return {
        torch::randint(0, 255, {batch, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8),
        torch::randn({batch, cfg.nb_sensors})};
}

// ========================================================================
// Fixed tests
// ========================================================================

TEST_F(SacAgentTest, ParameterCountPositive) {
    constexpr SacTestConfig cfg{8, 8, 10, 4, 2};
    const auto factory = make_factory(cfg);

    ASSERT_GT(factory->get_trainer()->count_parameters(), 0);
}

TEST_F(SacAgentTest, MetricsNotEmpty) {
    constexpr SacTestConfig cfg{8, 8, 10, 4, 2};
    const auto factory = make_factory(cfg);

    const auto metrics = factory->get_trainer()->get_metrics();

    ASSERT_EQ(metrics.size(), 12);
}

// ========================================================================
// Parameterized: act shape tests
// ========================================================================

TEST_P(SacActShapeParamTest, ActOutputShapes) {
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

TEST_P(SacActShapeParamTest, ActContinuousFinite) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    constexpr int batch = 4;
    const auto [continuous_action, discrete_action] =
        factory->get_agent()->act(make_state(cfg, batch));

    ASSERT_TRUE(torch::all(torch::isfinite(continuous_action)).item<bool>());
}

TEST_P(SacActShapeParamTest, ActDiscreteIsOneHot) {
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
    SacAgent, SacActShapeParamTest,
    testing::Values(
        SacTestConfig{8, 8, 10, 4, 2}, SacTestConfig{8, 8, 5, 2, 3},
        SacTestConfig{16, 16, 20, 6, 4}, SacTestConfig{8, 12, 10, 4, 2}));

// ========================================================================
// Parameterized: save / load tests
// ========================================================================

TEST_P(SacSaveLoadParamTest, SaveCreatesExpectedFiles) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    const auto save_dir = tmp_dir / "save_test";
    std::filesystem::create_directories(save_dir);

    factory->get_trainer()->save(save_dir);

    const std::vector<std::string> expected_files = {
        "actor.pt",
        "critic_1.pt",
        "critic_2.pt",
        "target_critic_1.pt",
        "target_critic_2.pt",
        "alpha_continuous.pt",
        "alpha_discrete.pt",
        "actor_optim.pt",
        "critic_1_optim.pt",
        "critic_2_optim.pt",
        "alpha_continuous_optim.pt",
        "alpha_discrete_optim.pt",
        "actor_repr.txt",
        "critic_repr.txt",
    };

    for (const auto &f: expected_files)
        ASSERT_TRUE(std::filesystem::exists(save_dir / f)) << "Missing file: " << f;
}

TEST_P(SacSaveLoadParamTest, SavedFilesNonEmpty) {
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
    SacAgent, SacSaveLoadParamTest,
    testing::Values(
        SacTestConfig{8, 8, 10, 4, 2}, SacTestConfig{8, 8, 5, 2, 3},
        SacTestConfig{16, 16, 20, 6, 4}));

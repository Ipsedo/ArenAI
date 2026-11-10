#include <arenai_agent_tests/tests_agents/tests_sac.h>

using namespace arenai;
using namespace arenai::agent;

void SacAgentTest::SetUp() {
    tmp_dir = std::filesystem::temp_directory_path() / "arenai_test_sac";
    std::filesystem::create_directories(tmp_dir);
}

void SacAgentTest::TearDown() { std::filesystem::remove_all(tmp_dir); }

std::unique_ptr<SacTorchAgentFactory> SacAgentTest::make_factory(const SacTestConfig &cfg) const {
    const SacHyperParams params{
        .hidden_size_sensors = 16,
        .actor_hidden_sizes = {32},
        .critic_hidden_sizes = {32},
        .vision_channels = {{3, 8}},
        .group_norm_nums = {4},
        .metric_window_size = 10,
        .buffer_size = 64,
        .learning_starts = 8,
        .batch_size = 8,
        .gpu_batch_size = 4};

    return std::make_unique<SacTorchAgentFactory>(
        cfg.vision_height, cfg.vision_width, cfg.nb_sensors, cfg.nb_continuous_actions,
        cfg.nb_discrete_actions, device, params);
}

TorchState SacAgentTest::make_state(const SacTestConfig &cfg, const int batch) {
    return TorchState(
        torch::randint(0, 255, {batch, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8),
        torch::randn({batch, cfg.nb_sensors}), torch::ones({batch, 1}));
}

TEST_F(SacAgentTest, ParameterCountPositive) {
    constexpr SacTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 10,
        .nb_continuous_actions = 4,
        .nb_discrete_actions = 2};
    const auto factory = make_factory(cfg);

    ASSERT_GT(factory->get_trainer()->count_parameters(), 0);
}

TEST_F(SacAgentTest, MetricsNotEmpty) {
    constexpr SacTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 10,
        .nb_continuous_actions = 4,
        .nb_discrete_actions = 2};
    const auto factory = make_factory(cfg);

    ASSERT_EQ(factory->get_trainer()->get_metrics().size(), 5);
    ASSERT_EQ(
        factory->get_trainer()->get_csv_only_metrics().size(),
        cfg.nb_continuous_actions + cfg.nb_discrete_actions);
}

TEST_F(SacAgentTest, ConfigCarriesGamma) {
    constexpr SacTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 10,
        .nb_continuous_actions = 4,
        .nb_discrete_actions = 2};
    const auto factory = make_factory(cfg);

    ASSERT_TRUE(factory->get_config().contains("gamma"));
    ASSERT_TRUE(factory->get_config().contains("buffer_size"));
}

TEST_F(SacAgentTest, MaskedFireIsNeverEngaged) {
    constexpr SacTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 10,
        .nb_continuous_actions = 4,
        .nb_discrete_actions = 2};
    const auto factory = make_factory(cfg);

    constexpr int batch = 64;
    auto state = make_state(cfg, batch);
    state = TorchState(state.get_vision(), state.get_state(), torch::zeros({batch, 1}));

    const auto [continuous_action, discrete_action] = factory->get_agent()->act(state, true);

    // column 0 = fire
    ASSERT_TRUE(torch::all(discrete_action.select(1, 0) == 0.f).item<bool>());
}

TEST_P(SacActShapeParamTest, ActOutputShapes) {
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

TEST_P(SacActShapeParamTest, ActContinuousInRange) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    constexpr int batch = 4;
    for (const bool sample: {true, false}) {
        const auto [continuous_action, discrete_action] =
            factory->get_agent()->act(make_state(cfg, batch), sample);

        ASSERT_TRUE(torch::all(torch::isfinite(continuous_action)).item<bool>());
        ASSERT_TRUE(torch::all(continuous_action.abs() <= 1.f).item<bool>());
    }
}

TEST_P(SacActShapeParamTest, ActDiscreteIsBinary) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    constexpr int batch = 4;
    const auto [continuous_action, discrete_action] =
        factory->get_agent()->act(make_state(cfg, batch), true);

    const auto is_binary =
        torch::logical_or(torch::eq(discrete_action, 0.0f), torch::eq(discrete_action, 1.0f));
    ASSERT_TRUE(torch::all(is_binary).item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    SacAgent, SacActShapeParamTest,
    testing::Values(
        SacTestConfig{8, 8, 10, 4, 2}, SacTestConfig{8, 8, 5, 2, 2},
        SacTestConfig{16, 16, 20, 6, 2}, SacTestConfig{8, 12, 10, 4, 2}));

TEST_P(SacSaveLoadParamTest, SaveCreatesExpectedFiles) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    const auto save_dir = tmp_dir / "save_test";
    std::filesystem::create_directories(save_dir);

    factory->get_trainer()->save(save_dir);

    const std::vector<std::string> expected_files = {
        "actor.pt",       "critic.pt",      "target_critic.pt",
        "alphas.pt",      "actor_optim.pt", "critic_optim.pt",
        "alpha_optim.pt", "actor_repr.txt", "critic_repr.txt",
    };

    for (const auto &f: expected_files)
        ASSERT_TRUE(std::filesystem::exists(save_dir / f)) << "Missing file: " << f;

    for (const auto &entry: std::filesystem::directory_iterator(save_dir))
        if (entry.is_regular_file())
            ASSERT_GT(entry.file_size(), 0u) << "Empty file: " << entry.path().filename();
}

TEST_P(SacSaveLoadParamTest, LoadedActorActsLikeSaved) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    const auto save_dir = tmp_dir / "save_load";
    std::filesystem::create_directories(save_dir);
    factory->get_trainer()->save(save_dir);

    const auto other = make_factory(cfg);
    const auto loaded = std::dynamic_pointer_cast<TorchSacAgent>(other->get_agent());
    ASSERT_NE(loaded, nullptr);
    loaded->load(save_dir);

    const auto state = make_state(cfg, 3);
    const auto [expected_continuous, expected_discrete] = factory->get_agent()->act(state, false);
    const auto [continuous, discrete] = loaded->act(state, false);

    ASSERT_TRUE(torch::allclose(continuous, expected_continuous));
    ASSERT_TRUE(torch::equal(discrete, expected_discrete));
}

INSTANTIATE_TEST_SUITE_P(
    SacAgent, SacSaveLoadParamTest,
    testing::Values(SacTestConfig{8, 8, 10, 4, 2}, SacTestConfig{16, 16, 20, 6, 2}));

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

std::unique_ptr<SacAgent> SacAgentTest::make_agent(const SacTestConfig &cfg) const {
    const std::vector<int> actor_hidden{32};
    const std::vector<int> critic_hidden{32};
    const std::vector<std::tuple<int, int>> vision_channels{{3, 8}};
    const std::vector<int> group_norm_nums{4};

    return std::make_unique<SacAgent>(
        cfg.vision_height, cfg.vision_width, cfg.nb_sensors, cfg.nb_continuous_actions,
        cfg.nb_discrete_actions, 1e-3f, 1e-3f, 1e-3f, 16, 16, actor_hidden, critic_hidden,
        vision_channels, group_norm_nums, device, 10, 0.005f, 0.99f);
}

std::unique_ptr<ReplayBuffer>
SacAgentTest::make_filled_buffer(const SacTestConfig &cfg, const int n_steps) {
    auto buffer = std::make_unique<ReplayBuffer>(n_steps * 2);

    for (int i = 0; i < n_steps; ++i) {
        TorchStep step;
        step.state.vision =
            torch::randint(0, 255, {3, cfg.vision_height, cfg.vision_width}, torch::kUInt8);
        step.state.proprioception = torch::randn({cfg.nb_sensors});
        step.action.continuous_action = torch::randn({cfg.nb_continuous_actions});

        auto disc = torch::zeros({cfg.nb_discrete_actions});
        disc[0] = 1.0f;
        step.action.discrete_action = disc;

        step.reward = torch::randn({1});
        step.done = torch::zeros({1});
        step.next_state.vision =
            torch::randint(0, 255, {3, cfg.vision_height, cfg.vision_width}, torch::kUInt8);
        step.next_state.proprioception = torch::randn({cfg.nb_sensors});

        buffer->add(step);
    }

    return buffer;
}

// ========================================================================
// Fixed tests
// ========================================================================

TEST_F(SacAgentTest, ParameterCountPositive) {
    constexpr SacTestConfig cfg{8, 8, 10, 4, 2};
    const auto agent = make_agent(cfg);

    ASSERT_GT(agent->count_parameters(), 0);
}

TEST_F(SacAgentTest, MetricsNotEmpty) {
    constexpr SacTestConfig cfg{8, 8, 10, 4, 2};
    const auto agent = make_agent(cfg);

    const auto metrics = agent->get_metrics();

    ASSERT_EQ(metrics.size(), 12);
}

// ========================================================================
// Train test: metrics no longer zero after training
// ========================================================================

TEST_F(SacAgentTest, TrainUpdatesMetrics) {
    constexpr SacTestConfig cfg{8, 8, 10, 4, 2};
    const auto agent = make_agent(cfg);

    const auto buffer = make_filled_buffer(cfg, 32);

    agent->train(buffer, 3, 8);

    const auto metrics = agent->get_metrics();

    int non_zero_count = 0;
    for (const auto &m: metrics)
        if (m->compute_metric() != 0.0f) non_zero_count++;

    ASSERT_GT(non_zero_count, 0) << "At least some metrics should be non-zero after training";
}

TEST_F(SacAgentTest, TrainMultipleEpochsNoThrow) {
    constexpr SacTestConfig cfg{8, 8, 10, 4, 2};
    const auto agent = make_agent(cfg);

    const auto buffer = make_filled_buffer(cfg, 32);

    ASSERT_NO_THROW(agent->train(buffer, 5, 8));
}

// ========================================================================
// Parameterized: act shape tests
// ========================================================================

TEST_P(SacActShapeParamTest, ActOutputShapes) {
    const auto cfg = GetParam();
    const auto agent = make_agent(cfg);

    constexpr int batch = 4;
    const auto vision =
        torch::randint(0, 255, {batch, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8);
    const auto sensors = torch::randn({batch, cfg.nb_sensors});

    agent->set_train(false);
    const auto [continuous_action, discrete_action] = agent->act(vision, sensors);

    ASSERT_EQ(continuous_action.size(0), batch);
    ASSERT_EQ(continuous_action.size(1), cfg.nb_continuous_actions);

    ASSERT_EQ(discrete_action.size(0), batch);
    ASSERT_EQ(discrete_action.size(1), cfg.nb_discrete_actions);
}

TEST_P(SacActShapeParamTest, ActContinuousFinite) {
    const auto cfg = GetParam();
    const auto agent = make_agent(cfg);

    constexpr int batch = 4;
    const auto vision =
        torch::randint(0, 255, {batch, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8);
    const auto sensors = torch::randn({batch, cfg.nb_sensors});

    agent->set_train(false);
    const auto [continuous_action, discrete_action] = agent->act(vision, sensors);

    ASSERT_TRUE(torch::all(torch::isfinite(continuous_action)).item<bool>());
}

TEST_P(SacActShapeParamTest, ActDiscreteIsOneHot) {
    const auto cfg = GetParam();
    const auto agent = make_agent(cfg);

    constexpr int batch = 4;
    const auto vision =
        torch::randint(0, 255, {batch, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8);
    const auto sensors = torch::randn({batch, cfg.nb_sensors});

    agent->set_train(false);
    const auto [continuous_action, discrete_action] = agent->act(vision, sensors);

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
    const auto agent = make_agent(cfg);

    const auto save_dir = tmp_dir / "save_test";
    std::filesystem::create_directories(save_dir);

    agent->save(save_dir);

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

    ASSERT_TRUE(std::filesystem::is_directory(save_dir / "actor_state_dict"));
}

TEST_P(SacSaveLoadParamTest, SavedFilesNonEmpty) {
    const auto cfg = GetParam();
    const auto agent = make_agent(cfg);

    const auto save_dir = tmp_dir / "save_nonempty";
    std::filesystem::create_directories(save_dir);

    agent->save(save_dir);

    for (const auto &entry: std::filesystem::directory_iterator(save_dir)) {
        if (entry.is_regular_file())
            ASSERT_GT(entry.file_size(), 0u) << "Empty file: " << entry.path().filename();
    }
}

TEST_P(SacSaveLoadParamTest, LoadRestoresWeights) {
    const auto cfg = GetParam();
    const auto agent1 = make_agent(cfg);

    const auto save_dir = tmp_dir / "save_restore";
    std::filesystem::create_directories(save_dir);

    agent1->save(save_dir);

    const auto agent2 = make_agent(cfg);

    agent2->load(save_dir);

    constexpr int batch = 2;
    const auto vision =
        torch::randint(0, 255, {batch, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8);
    const auto sensors = torch::randn({batch, cfg.nb_sensors});

    torch::manual_seed(42);
    agent1->set_train(false);
    const auto [cont1, disc1] = agent1->act(vision, sensors);

    torch::manual_seed(42);
    agent2->set_train(false);
    const auto [cont2, disc2] = agent2->act(vision, sensors);

    ASSERT_TRUE(torch::allclose(cont1, cont2, 1e-4, 1e-4))
        << "Continuous actions should match after load";
    ASSERT_TRUE(torch::equal(disc1, disc2)) << "Discrete actions should match after load";
}

TEST_P(SacSaveLoadParamTest, LoadAfterTrainPreservesState) {
    const auto cfg = GetParam();
    const auto agent = make_agent(cfg);

    const auto buffer = make_filled_buffer(cfg, 32);
    agent->train(buffer, 2, 8);

    const auto save_dir = tmp_dir / "save_after_train";
    std::filesystem::create_directories(save_dir);

    agent->save(save_dir);

    const auto agent2 = make_agent(cfg);
    agent2->load(save_dir);

    constexpr int batch = 2;
    const auto vision =
        torch::randint(0, 255, {batch, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8);
    const auto sensors = torch::randn({batch, cfg.nb_sensors});

    torch::manual_seed(123);
    agent->set_train(false);
    const auto [cont1, disc1] = agent->act(vision, sensors);

    torch::manual_seed(123);
    agent2->set_train(false);
    const auto [cont2, disc2] = agent2->act(vision, sensors);

    ASSERT_TRUE(torch::allclose(cont1, cont2, 1e-4, 1e-4));
    ASSERT_TRUE(torch::equal(disc1, disc2));
}

INSTANTIATE_TEST_SUITE_P(
    SacAgent, SacSaveLoadParamTest,
    testing::Values(
        SacTestConfig{8, 8, 10, 4, 2}, SacTestConfig{8, 8, 5, 2, 3},
        SacTestConfig{16, 16, 20, 6, 4}));

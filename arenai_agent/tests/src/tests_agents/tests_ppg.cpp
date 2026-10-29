//
// Created by claude on 29/10/2026.
//

#include <arenai_agent_tests/tests_agents/tests_ppg.h>

using namespace arenai;
using namespace arenai::agent;

// ========================================================================
// Fixture helpers
// ========================================================================

void PpgAgentTest::SetUp() {
    tmp_dir = std::filesystem::temp_directory_path() / "arenai_test_ppg";
    std::filesystem::create_directories(tmp_dir);
}

void PpgAgentTest::TearDown() { std::filesystem::remove_all(tmp_dir); }

std::unique_ptr<PpgTorchAgentFactory> PpgAgentTest::make_factory(const PpgTestConfig &cfg) const {
    const PpgHyperParams params{
        .actor_learning_rate = 1e-3f,
        .critic_learning_rate = 1e-3f,
        .hidden_size_sensors = 16,
        .actor_hidden_sizes = {32},
        .critic_hidden_sizes = {32},
        .vision_channels = {{3, 8}},
        .group_norm_nums = {4},
        .metric_window_size = 10,
        .gamma = 0.99f,
        .gae_lambda = 0.95f,
        .clip_epsilon = 0.2f,
        .grad_norm_max = 1.f,
        .actor_epochs = 1,
        .critic_epochs = 1,
        .rollout_size = 8,
        .actor_batch_size = 10,
        .actor_gpu_batch_size = 10,
        .critic_batch_size = 10,
        .critic_gpu_batch_size = 10,
        .actor_aux_epochs = 1,
        .critic_aux_epochs = 1};

    return std::make_unique<PpgTorchAgentFactory>(
        cfg.vision_height, cfg.vision_width, cfg.nb_sensors, cfg.nb_continuous_actions,
        cfg.nb_discrete_actions, device, params);
}

TorchState PpgAgentTest::make_state(const PpgTestConfig &cfg, const int batch) {
    return {
        .vision =
            torch::randint(0, 255, {batch, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8),
        .proprioception = torch::randn({batch, cfg.nb_sensors}),
        .fire_mask = torch::ones({batch, 1})};
}

// ========================================================================
// Fixed tests
// ========================================================================

TEST_F(PpgAgentTest, ParameterCountPositive) {
    constexpr PpgTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 10,
        .nb_continuous_actions = 4,
        .nb_discrete_actions = 2};
    const auto factory = make_factory(cfg);

    ASSERT_GT(factory->get_trainer()->count_parameters(), 0);
}

TEST_F(PpgAgentTest, MetricsNotEmpty) {
    constexpr PpgTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 10,
        .nb_continuous_actions = 4,
        .nb_discrete_actions = 2};
    const auto factory = make_factory(cfg);

    const auto metrics = factory->get_trainer()->get_metrics();

    ASSERT_EQ(metrics.size(), 8);
}

TEST_F(PpgAgentTest, DistillationValueShape) {
    constexpr PpgTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 10,
        .nb_continuous_actions = 4,
        .nb_discrete_actions = 2};

    DistillationActor actor(
        cfg.nb_sensors, cfg.nb_continuous_actions, cfg.nb_discrete_actions, 16, std::vector{32},
        {{3, 8}}, {4}, 0.5f, std::vector(cfg.nb_discrete_actions, 0.5f));

    constexpr int batch = 4;
    const auto state = make_state(cfg, batch);
    const auto value = actor.value(state.vision, state.proprioception);

    ASSERT_EQ(value.sizes().vec(), (std::vector<int64_t>{batch, 1}));
    ASSERT_TRUE(torch::all(torch::isfinite(value)).item<bool>());
}

TEST_F(PpgAgentTest, DistillationValueHeadIsRegistered) {
    // the auxiliary value head must be part of the actor's parameters: saved with
    // it and stepped by the actor optimizer during the auxiliary phase
    DistillationActor actor(10, 4, 2, 16, std::vector{32}, {{3, 8}}, {4}, 0.5f, {0.5f, 0.5f});
    const Actor plain_actor(10, 4, 2, 16, std::vector{32}, {{3, 8}}, {4}, 0.5f, {0.5f, 0.5f});

    auto named_parameters = actor.named_parameters();
    ASSERT_TRUE(named_parameters.contains("value.weight"));
    ASSERT_TRUE(named_parameters.contains("value.bias"));

    ASSERT_EQ(actor.parameters().size(), plain_actor.parameters().size() + 2);
}

// ========================================================================
// Parameterized: act shape tests
// ========================================================================

TEST_P(PpgActShapeParamTest, ActOutputShapes) {
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

TEST_P(PpgActShapeParamTest, ActContinuousFinite) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    constexpr int batch = 4;
    const auto [continuous_action, discrete_action] =
        factory->get_agent()->act(make_state(cfg, batch), true);

    ASSERT_TRUE(torch::all(torch::isfinite(continuous_action)).item<bool>());
}

TEST_P(PpgActShapeParamTest, ActDiscreteIsBinary) {
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
    PpgAgent, PpgActShapeParamTest,
    testing::Values(
        PpgTestConfig{8, 8, 10, 4, 2}, PpgTestConfig{8, 8, 5, 2, 2},
        PpgTestConfig{16, 16, 20, 6, 2}, PpgTestConfig{8, 12, 10, 4, 2}));

// ========================================================================
// Parameterized: save / load tests
// ========================================================================

TEST_P(PpgSaveLoadParamTest, SaveCreatesExpectedFiles) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    const auto save_dir = tmp_dir / "save_test";
    std::filesystem::create_directories(save_dir);

    factory->get_trainer()->save(save_dir);

    const std::vector<std::string> expected_files = {
        "actor.pt",         "critic.pt",      "actor_optim.pt",  "critic_optim.pt",
        "return_scaler.pt", "actor_repr.txt", "critic_repr.txt",
    };

    for (const auto &f: expected_files)
        ASSERT_TRUE(std::filesystem::exists(save_dir / f)) << "Missing file: " << f;
}

TEST_P(PpgSaveLoadParamTest, SavedFilesNonEmpty) {
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

TEST_P(PpgSaveLoadParamTest, AgentLoadsSavedActor) {
    const auto cfg = GetParam();
    const auto factory = make_factory(cfg);

    const auto save_dir = tmp_dir / "save_load";
    std::filesystem::create_directories(save_dir);

    factory->get_trainer()->save(save_dir);

    // a fresh agent reloads the saved actor and acts deterministically like it
    const auto other_factory = make_factory(cfg);
    const auto other_agent = std::dynamic_pointer_cast<AbstractAgent>(other_factory->get_agent());
    ASSERT_NE(other_agent, nullptr);
    ASSERT_NO_THROW(other_agent->load(save_dir));

    const auto state = make_state(cfg, 4);
    const auto [continuous_action, discrete_action] = factory->get_agent()->act(state, false);
    const auto [other_continuous_action, other_discrete_action] =
        other_factory->get_agent()->act(state, false);

    ASSERT_TRUE(torch::allclose(continuous_action, other_continuous_action));
    ASSERT_TRUE(torch::equal(discrete_action, other_discrete_action));
}

INSTANTIATE_TEST_SUITE_P(
    PpgAgent, PpgSaveLoadParamTest,
    testing::Values(
        PpgTestConfig{8, 8, 10, 4, 2}, PpgTestConfig{8, 8, 5, 2, 2},
        PpgTestConfig{16, 16, 20, 6, 2}));

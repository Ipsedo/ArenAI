//
// Created by claude on 08/11/2026.
//

#include <cmath>

#include <arenai_agent_tests/tests_agents/tests_sac_training.h>

using namespace arenai;
using namespace arenai::agent;

SacTrainingTriad SacTrainingTest::make_triad(const SacTrainingTestConfig &cfg) const {
    const std::vector<std::tuple<int, int>> vision_channels{{3, 4}};
    const std::vector group_norm_nums{2};

    SacTrainingTriad triad;
    triad.actor = std::make_shared<SacActor>(
        cfg.nb_sensors, cfg.nb_continuous_actions, cfg.nb_discrete_actions, 8, std::vector{16},
        vision_channels, group_norm_nums, 0.3f, std::vector(cfg.nb_discrete_actions, 0.2f));
    triad.replay_buffer = std::make_shared<SacReplayBuffer>(64);
    triad.collector = std::make_shared<SacStepCollector>(triad.replay_buffer);
    triad.agent = std::make_shared<TorchSacAgent>(triad.actor, device, triad.collector);
    triad.trainer = std::make_shared<SacTrainer>(
        triad.actor, triad.replay_buffer, cfg.nb_sensors, cfg.nb_continuous_actions,
        cfg.nb_discrete_actions, 1e-3f, 1e-3f, 1e-3f, 8, std::vector{16}, vision_channels,
        group_norm_nums, device, 10, 0.99f, 0.05f, 0.1f, 0.3f, 0.3f, LEARNING_STARTS, 1, BATCH_SIZE,
        GPU_BATCH_SIZE);
    return triad;
}

TorchState SacTrainingTest::make_state(const SacTrainingTestConfig &cfg, const int nb_tanks) {
    return {
        .vision = torch::randint(
            0, 255, {nb_tanks, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8),
        .proprioception = torch::randn({nb_tanks, cfg.nb_sensors}),
        .fire_mask = torch::randint(0, 2, {nb_tanks, 1}).to(torch::kFloat)};
}

TEST_F(SacTrainingTest, NoUpdateBeforeLearningStarts) {
    constexpr SacTrainingTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 3,
        .nb_continuous_actions = 2,
        .nb_discrete_actions = 2};
    const auto triad = make_triad(cfg);

    std::vector<torch::Tensor> initial_parameters;
    for (const auto &parameter: triad.actor->parameters())
        initial_parameters.push_back(parameter.detach().clone());

    // 2 steps -> 2 sampleable transitions, under the threshold
    for (int t = 0; t < 2; t++) {
        triad.agent->act(make_state(cfg, NB_TANKS), true);
        triad.collector->on_transition(
            torch::randn({NB_TANKS, 1}), torch::zeros({NB_TANKS, 1}), torch::zeros({NB_TANKS, 1}));
        triad.trainer->step();
    }

    const auto parameters = triad.actor->parameters();
    for (size_t i = 0; i < parameters.size(); i++)
        ASSERT_TRUE(torch::equal(parameters[i], initial_parameters[i]));
}

TEST_F(SacTrainingTest, TrainingUpdatesActorParameters) {
    constexpr SacTrainingTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 3,
        .nb_continuous_actions = 2,
        .nb_discrete_actions = 3};
    const auto triad = make_triad(cfg);

    std::vector<torch::Tensor> initial_parameters;
    for (const auto &parameter: triad.actor->parameters())
        initial_parameters.push_back(parameter.detach().clone());

    for (int t = 0; t < 6; t++) {
        triad.agent->act(make_state(cfg, NB_TANKS), true);
        triad.collector->on_transition(
            torch::randn({NB_TANKS, 1}), torch::zeros({NB_TANKS, 1}), torch::zeros({NB_TANKS, 1}));
        triad.trainer->step();
    }

    ASSERT_GE(triad.replay_buffer->nb_transitions(), LEARNING_STARTS);

    const auto parameters = triad.actor->parameters();
    ASSERT_EQ(parameters.size(), initial_parameters.size());

    bool any_changed = false;
    for (size_t i = 0; i < parameters.size(); i++)
        if (!torch::allclose(parameters[i], initial_parameters[i])) {
            any_changed = true;
            break;
        }
    ASSERT_TRUE(any_changed) << "Training should update the actor's parameters";

    for (const auto &metric: triad.trainer->get_metrics())
        ASSERT_TRUE(std::isfinite(metric->compute_metric())) << metric->get_name();
}

TEST_F(SacTrainingTest, MaskedFireReceivesNoUpdate) {
    // fire never possible: its head row must stay untouched while zoom keeps learning
    constexpr SacTrainingTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 3,
        .nb_continuous_actions = 2,
        .nb_discrete_actions = 2};
    const auto triad = make_triad(cfg);

    auto named_parameters = triad.actor->named_parameters();
    const auto weight = named_parameters["discrete.0.weight"];
    const auto bias = named_parameters["discrete.0.bias"];
    const auto initial_weight = weight.detach().clone();
    const auto initial_bias = bias.detach().clone();

    for (int t = 0; t < 6; t++) {
        auto state = make_state(cfg, NB_TANKS);
        state.fire_mask = torch::zeros({NB_TANKS, 1});

        triad.agent->act(state, true);
        triad.collector->on_transition(
            torch::randn({NB_TANKS, 1}), torch::zeros({NB_TANKS, 1}), torch::zeros({NB_TANKS, 1}));
        triad.trainer->step();
    }

    // row 0 = fire, row 1 = zoom
    ASSERT_TRUE(torch::equal(weight[0], initial_weight[0]));
    ASSERT_TRUE(torch::equal(bias[0], initial_bias[0]));
    ASSERT_FALSE(torch::equal(weight[1], initial_weight[1]));
}

TEST_F(SacTrainingTest, EpisodeEndKeepsTraining) {
    constexpr SacTrainingTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 3,
        .nb_continuous_actions = 2,
        .nb_discrete_actions = 2};
    const auto triad = make_triad(cfg);

    for (int episode = 0; episode < 2; episode++) {
        for (int t = 0; t < 4; t++) {
            triad.agent->act(make_state(cfg, NB_TANKS), true);
            // the last step kills one tank
            const auto done = t == 3 ? torch::tensor({{1.f}, {0.f}}) : torch::zeros({NB_TANKS, 1});
            triad.collector->on_transition(
                torch::randn({NB_TANKS, 1}), done, torch::zeros({NB_TANKS, 1}));
            triad.trainer->step();
        }
        triad.collector->on_episode_end(make_state(cfg, NB_TANKS));
    }

    // 2 episodes x 4 steps x 2 tanks, every step closed by its successor
    ASSERT_EQ(triad.replay_buffer->nb_transitions(), 16);

    for (const auto &metric: triad.trainer->get_metrics())
        ASSERT_TRUE(std::isfinite(metric->compute_metric())) << metric->get_name();
}

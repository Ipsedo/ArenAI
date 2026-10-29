//
// Created by claude on 29/10/2026.
//

#include <arenai_agent_tests/tests_agents/tests_ppg_training.h>

using namespace arenai;
using namespace arenai::agent;

std::unique_ptr<PpgTorchAgentFactory>
PpgTrainingTest::make_factory(const PpgTrainingTestConfig &cfg) const {
    const PpgHyperParams params{
        .actor_learning_rate = 1e-3f,
        .critic_learning_rate = 1e-3f,
        .hidden_size_sensors = 8,
        .actor_hidden_sizes = {16},
        .critic_hidden_sizes = {16},
        .vision_channels = {{3, 4}},
        .group_norm_nums = {2},
        .metric_window_size = 10,
        .gamma = 0.99f,
        .gae_lambda = 0.95f,
        .clip_epsilon = 0.2f,
        .actor_epochs = 2,
        .critic_epochs = 2,
        .rollout_size = ROLLOUT_SIZE,
        .actor_batch_size = MINIBATCH_SIZE,
        .actor_gpu_batch_size = MINIBATCH_SIZE,
        .critic_batch_size = MINIBATCH_SIZE,
        .critic_gpu_batch_size = MINIBATCH_SIZE,
        .actor_aux_epochs = 2,
        .critic_aux_epochs = 2};

    return std::make_unique<PpgTorchAgentFactory>(
        cfg.vision_height, cfg.vision_width, cfg.nb_sensors, cfg.nb_continuous_actions,
        cfg.nb_discrete_actions, device, params);
}

TorchState PpgTrainingTest::make_state(const PpgTrainingTestConfig &cfg, const int nb_tanks) {
    return {
        .vision = torch::randint(
            0, 255, {nb_tanks, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8),
        .proprioception = torch::randn({nb_tanks, cfg.nb_sensors}),
        .fire_mask = torch::randint(0, 2, {nb_tanks, 1}).to(torch::kFloat)};
}

PpgTriad PpgTrainingTest::make_triad(const PpgTrainingTestConfig &cfg) const {
    const std::vector<std::tuple<int, int>> vision_channels{{3, 4}};
    const std::vector group_norm_nums{2};

    PpgTriad triad;
    triad.actor = std::make_shared<DistillationActor>(
        cfg.nb_sensors, cfg.nb_continuous_actions, cfg.nb_discrete_actions, 8, std::vector{16},
        vision_channels, group_norm_nums, 0.1f, std::vector(cfg.nb_discrete_actions, 0.2f));
    triad.rollout_buffer = std::make_shared<PpgRolloutBuffer>();
    triad.collector = std::make_shared<PpgStepCollector>(triad.rollout_buffer);
    triad.agent = std::make_shared<TorchPpgAgent>(triad.actor, device, triad.collector);
    triad.trainer = std::make_shared<PpgTrainer>(
        triad.actor, triad.rollout_buffer, cfg.nb_sensors, 1e-3f, 1e-3f, 8, std::vector{16},
        vision_channels, group_norm_nums, device, 10, 0.99f, 0.95f, 0.2f, 0.f, 1.f, 0.001f, 0.001f,
        2, 2, ROLLOUT_SIZE, MINIBATCH_SIZE, MINIBATCH_SIZE, MINIBATCH_SIZE, MINIBATCH_SIZE, 2, 2,
        1.f);
    return triad;
}

void PpgTrainingTest::run_env_loop(
    const PpgTriad &triad, const PpgTrainingTestConfig &cfg, const int nb_steps,
    const bool fire_possible) {
    for (int t = 0; t < nb_steps; t++) {
        constexpr int nb_tanks = 2;

        auto state = make_state(cfg, nb_tanks);
        if (!fire_possible) state.fire_mask = torch::zeros({nb_tanks, 1});

        triad.agent->act(state, true);
        triad.collector->on_transition(
            torch::randn({nb_tanks, 1}), torch::zeros({nb_tanks, 1}), torch::zeros({nb_tanks, 1}));
        triad.trainer->step();
    }
}

TEST_F(PpgTrainingTest, ActProducesValidOutput) {
    constexpr PpgTrainingTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 3,
        .nb_continuous_actions = 2,
        .nb_discrete_actions = 2};
    const auto factory = make_factory(cfg);

    const auto [continuous_action, discrete_action] =
        factory->get_agent()->act(make_state(cfg, 1), true);

    ASSERT_EQ(continuous_action.size(0), 1);
    ASSERT_EQ(continuous_action.size(1), 2);
    ASSERT_EQ(discrete_action.size(0), 1);
    ASSERT_EQ(discrete_action.size(1), 2);

    ASSERT_TRUE(torch::all(torch::isfinite(continuous_action)).item<bool>());
    ASSERT_TRUE(torch::all(torch::isfinite(discrete_action)).item<bool>());
}

TEST_F(PpgTrainingTest, CountParametersPositive) {
    constexpr PpgTrainingTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 3,
        .nb_continuous_actions = 2,
        .nb_discrete_actions = 2};
    const auto factory = make_factory(cfg);

    ASSERT_GT(factory->get_trainer()->count_parameters(), 0)
        << "Agent should have a positive number of parameters";
}

TEST_F(PpgTrainingTest, FactoryTrainingRunsWithoutError) {
    constexpr PpgTrainingTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 3,
        .nb_continuous_actions = 2,
        .nb_discrete_actions = 2};
    const auto factory = make_factory(cfg);

    for (int t = 0; t < ROLLOUT_SIZE + 2; t++) {
        constexpr int nb_tanks = 2;

        factory->get_agent()->act(make_state(cfg, nb_tanks), true);
        factory->get_collector()->on_transition(
            torch::randn({nb_tanks, 1}), torch::zeros({nb_tanks, 1}), torch::zeros({nb_tanks, 1}));
        ASSERT_NO_THROW(factory->get_trainer()->step());
    }
}

TEST_F(PpgTrainingTest, TrainingUpdatesActorParameters) {
    constexpr PpgTrainingTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 3,
        .nb_continuous_actions = 2,
        .nb_discrete_actions = 3};
    const auto triad = make_triad(cfg);

    std::vector<torch::Tensor> initial_parameters;
    for (const auto &parameter: triad.actor->parameters())
        initial_parameters.push_back(parameter.detach().clone());

    // one more step than the rollout horizon so that the batch is complete when the
    // trainer checks
    run_env_loop(triad, cfg, ROLLOUT_SIZE + 2, true);

    // the rollout has been consumed by the training
    ASSERT_LT(triad.rollout_buffer->nb_complete_steps(), static_cast<size_t>(ROLLOUT_SIZE));

    const auto parameters = triad.actor->parameters();
    ASSERT_EQ(parameters.size(), initial_parameters.size());

    bool any_changed = false;
    for (size_t i = 0; i < parameters.size(); i++)
        if (!torch::allclose(parameters[i], initial_parameters[i])) {
            any_changed = true;
            break;
        }

    ASSERT_TRUE(any_changed) << "Training should update the actor's parameters";
}

TEST_F(PpgTrainingTest, AuxiliaryPhaseUpdatesValueHead) {
    // the actor's value head only receives a gradient from the auxiliary phase
    constexpr PpgTrainingTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 3,
        .nb_continuous_actions = 2,
        .nb_discrete_actions = 2};
    const auto triad = make_triad(cfg);

    auto named_parameters = triad.actor->named_parameters();
    const auto weight = named_parameters["value.weight"];
    const auto initial_weight = weight.detach().clone();

    run_env_loop(triad, cfg, ROLLOUT_SIZE + 2, true);

    ASSERT_LT(triad.rollout_buffer->nb_complete_steps(), static_cast<size_t>(ROLLOUT_SIZE));
    ASSERT_FALSE(torch::equal(weight, initial_weight))
        << "The auxiliary phase should distill the value into the actor";
}

TEST_F(PpgTrainingTest, TrainedActorStaysFinite) {
    constexpr PpgTrainingTestConfig cfg{
        .vision_height = 8,
        .vision_width = 8,
        .nb_sensors = 3,
        .nb_continuous_actions = 2,
        .nb_discrete_actions = 2};
    const auto triad = make_triad(cfg);

    // several rollouts so that the auxiliary phase runs more than once
    run_env_loop(triad, cfg, 3 * ROLLOUT_SIZE + 2, true);

    for (const auto &item: triad.actor->named_parameters())
        ASSERT_TRUE(torch::all(torch::isfinite(item.value())).item<bool>())
            << "Non-finite parameter: " << item.key();
}

TEST_F(PpgTrainingTest, MaskedFireReceivesNoUpdate) {
    // fire never possible: its head row must stay untouched while zoom keeps learning
    constexpr PpgTrainingTestConfig cfg{
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

    run_env_loop(triad, cfg, ROLLOUT_SIZE + 2, false);

    // the rollout has been consumed by the training
    ASSERT_LT(triad.rollout_buffer->nb_complete_steps(), static_cast<size_t>(ROLLOUT_SIZE));

    // row 0 = fire, row 1 = zoom
    ASSERT_TRUE(torch::equal(weight[0], initial_weight[0]));
    ASSERT_TRUE(torch::equal(bias[0], initial_bias[0]));
    ASSERT_FALSE(torch::equal(weight[1], initial_weight[1]));
}

//
// Created by claude on 22/07/2026.
//

#include <arenai_agent_tests/tests_agents/tests_ppo_training.h>

using namespace arenai;
using namespace arenai::agent;

std::unique_ptr<PpoTorchAgentFactory>
PpoTrainingTest::make_factory(const PpoTrainingTestConfig &cfg) const {
    const PpoHyperParams params{
        .actor_learning_rate = 1e-3f,
        .critic_learning_rate = 1e-3f,
        .hidden_size_sensors = 8,
        .hidden_size_actions = 8,
        .actor_hidden_sizes = {16},
        .critic_hidden_sizes = {16},
        .vision_channels = {{3, 4}},
        .group_norm_nums = {2},
        .metric_window_size = 10,
        .gamma = 0.99f,
        .gae_lambda = 0.95f,
        .clip_epsilon = 0.2f,
        .continuous_entropy_coef = 0.01f,
        .discrete_entropy_coef = 0.01f,
        .epochs = 2,
        .rollout_size = ROLLOUT_SIZE};

    return std::make_unique<PpoTorchAgentFactory>(
        cfg.vision_height, cfg.vision_width, cfg.nb_sensors, cfg.nb_continuous_actions,
        cfg.nb_discrete_actions, device, params);
}

TorchState PpoTrainingTest::make_state(const PpoTrainingTestConfig &cfg, const int nb_tanks) {
    return {
        torch::randint(0, 255, {nb_tanks, 3, cfg.vision_height, cfg.vision_width}, torch::kUInt8),
        torch::randn({nb_tanks, cfg.nb_sensors})};
}

TEST_F(PpoTrainingTest, ActProducesValidOutput) {
    constexpr PpoTrainingTestConfig cfg{8, 8, 3, 2, 3};
    const auto factory = make_factory(cfg);

    const auto [continuous_action, discrete_action] = factory->get_agent()->act(make_state(cfg, 1));

    ASSERT_EQ(continuous_action.size(0), 1);
    ASSERT_EQ(continuous_action.size(1), 2);
    ASSERT_EQ(discrete_action.size(0), 1);
    ASSERT_EQ(discrete_action.size(1), 3);

    ASSERT_TRUE(torch::all(torch::isfinite(continuous_action)).item<bool>());
    ASSERT_TRUE(torch::all(torch::isfinite(discrete_action)).item<bool>());
}

TEST_F(PpoTrainingTest, CountParametersPositive) {
    constexpr PpoTrainingTestConfig cfg{8, 8, 3, 2, 3};
    const auto factory = make_factory(cfg);

    ASSERT_GT(factory->get_trainer()->count_parameters(), 0)
        << "Agent should have a positive number of parameters";
}

TEST_F(PpoTrainingTest, TrainingUpdatesActorParameters) {
    // build the triad by hand to keep a handle on the actor's parameters
    constexpr PpoTrainingTestConfig cfg{8, 8, 3, 2, 3};
    constexpr int nb_tanks = 2;

    const std::vector<std::tuple<int, int>> vision_channels{{3, 4}};
    const std::vector<int> group_norm_nums{2};

    const auto actor = std::make_shared<Actor>(
        cfg.vision_height, cfg.vision_width, cfg.nb_sensors, cfg.nb_continuous_actions,
        cfg.nb_discrete_actions, 8, std::vector<int>{16}, vision_channels, group_norm_nums);
    const auto rollout_buffer = std::make_shared<PpoRolloutBuffer>();
    const auto collector = std::make_shared<PpoStepCollector>(rollout_buffer);
    const auto agent = std::make_shared<TorchPpoAgent>(actor, device, collector);
    const auto trainer = std::make_shared<PpoTrainer>(
        actor, rollout_buffer, cfg.vision_height, cfg.vision_width, cfg.nb_sensors, 1e-3f, 1e-3f, 8,
        8, std::vector<int>{16}, vision_channels, group_norm_nums, device, 10, 0.99f, 0.95f, 0.2f,
        0.01f, 0.01f, 2, ROLLOUT_SIZE);

    std::vector<torch::Tensor> initial_parameters;
    for (const auto &parameter: actor->parameters())
        initial_parameters.push_back(parameter.detach().clone());

    // env loop: act -> transition -> maybe train, one more step than the rollout
    // horizon so that the batch is complete when the trainer checks
    for (int t = 0; t < ROLLOUT_SIZE + 2; t++) {
        agent->act(make_state(cfg, nb_tanks));
        collector->on_transition(
            torch::randn({nb_tanks, 1}), torch::zeros({nb_tanks, 1}), torch::zeros({nb_tanks, 1}));
        trainer->step();
    }

    // the rollout has been consumed by the training
    ASSERT_LT(rollout_buffer->nb_complete_steps(), static_cast<size_t>(ROLLOUT_SIZE));

    const auto parameters = actor->parameters();
    ASSERT_EQ(parameters.size(), initial_parameters.size());

    bool any_changed = false;
    for (size_t i = 0; i < parameters.size(); i++)
        if (!torch::allclose(parameters[i], initial_parameters[i])) {
            any_changed = true;
            break;
        }

    ASSERT_TRUE(any_changed) << "Training should update the actor's parameters";
}

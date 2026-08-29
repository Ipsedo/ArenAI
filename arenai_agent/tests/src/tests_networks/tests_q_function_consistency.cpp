//
// Created by claude on 01/07/2026.
//

#include <networks/actor.h>
#include <networks/q_function.h>

#include <arenai_agent_tests/tests_networks/tests_q_function_consistency.h>

using namespace arenai;
using namespace arenai::agent;

// ========================================================================
// value_per_discrete_action must match value_ohe action by action
// ========================================================================

TEST_F(QFunctionConsistencyTest, PerDiscreteActionMatchesOhe) {
    constexpr int height = 8, width = 8;
    constexpr int nb_sensors = 5, nb_cont = 3, nb_disc = 4;
    constexpr int batch = 4;

    QFunction q(height, width, nb_sensors, nb_cont, nb_disc, 8, 8, {16}, {{3, 4}}, {2});

    const auto vision = torch::randint(0, 255, {batch, 3, height, width}, torch::kUInt8);
    const auto sensors = torch::randn({batch, nb_sensors});
    const auto cont_actions = torch::randn({batch, nb_cont});

    torch::NoGradGuard no_grad;

    const auto q_per_action = q.value_per_discrete_action(vision, sensors, cont_actions);

    ASSERT_EQ(q_per_action.size(0), batch);
    ASSERT_EQ(q_per_action.size(1), nb_disc);

    const auto one_hots = torch::eye(nb_disc);
    for (int a = 0; a < nb_disc; a++) {
        const auto ohe = one_hots[a].unsqueeze(0).expand({batch, -1});
        const auto q_a = q.value_ohe(vision, sensors, cont_actions, ohe);

        ASSERT_TRUE(torch::allclose(q_per_action.select(1, a).unsqueeze(1), q_a, 1e-4, 1e-4))
            << "value_per_discrete_action column " << a << " should equal value_ohe(one_hot[" << a
            << "])";
    }
}

TEST_F(QFunctionConsistencyTest, ValueOheOutputFinite) {
    constexpr int height = 8, width = 8;
    constexpr int nb_sensors = 5, nb_cont = 3, nb_disc = 2;
    constexpr int batch = 4;

    QFunction q(height, width, nb_sensors, nb_cont, nb_disc, 8, 8, {16}, {{3, 4}}, {2});

    const auto vision = torch::randint(0, 255, {batch, 3, height, width}, torch::kUInt8);
    const auto sensors = torch::randn({batch, nb_sensors});
    const auto cont_actions = torch::randn({batch, nb_cont});
    const auto disc_ohe = torch::zeros({batch, nb_disc});
    disc_ohe.select(1, 0).fill_(1.0f);

    const auto value = q.value_ohe(vision, sensors, cont_actions, disc_ohe);

    ASSERT_TRUE(torch::all(torch::isfinite(value)).item<bool>()) << "Q-value should be finite";
}

// ========================================================================
// Gradient flow tests for networks
// ========================================================================

TEST_F(QFunctionGradientTest, GradientFlowsThroughQFunction) {
    constexpr int height = 8, width = 8;
    constexpr int nb_sensors = 3, nb_cont = 2, nb_disc = 2;
    constexpr int batch = 2;

    QFunction q(height, width, nb_sensors, nb_cont, nb_disc, 8, 8, {16}, {{3, 4}}, {2});

    const auto vision = torch::randint(0, 255, {batch, 3, height, width}, torch::kUInt8);
    const auto sensors = torch::randn({batch, nb_sensors});
    const auto cont_actions = torch::randn({batch, nb_cont});

    const auto value = q.value_per_discrete_action(vision, sensors, cont_actions);
    const auto loss = value.sum();

    loss.backward();

    bool any_grad = false;
    for (const auto &p: q.parameters()) {
        if (p.grad().defined() && p.grad().abs().sum().item<float>() > 0) {
            any_grad = true;
            ASSERT_TRUE(torch::all(torch::isfinite(p.grad())).item<bool>())
                << "All gradients should be finite";
        }
    }
    ASSERT_TRUE(any_grad) << "At least some parameters should receive gradients";
}

TEST_F(ActorGradientTest, GradientFlowsThroughActor) {
    constexpr int height = 8, width = 8;
    constexpr int nb_sensors = 3, nb_cont = 2, nb_disc = 2;
    constexpr int batch = 2;

    Actor actor(height, width, nb_sensors, nb_cont, nb_disc, 8, {16}, {{3, 4}}, {2}, 0.1f, 0.2f);

    const auto vision = torch::randint(0, 255, {batch, 3, height, width}, torch::kUInt8);
    const auto sensors = torch::randn({batch, nb_sensors});

    const auto [alpha, beta, discrete] = actor.act(vision, sensors);
    const auto loss = alpha.sum() + beta.sum() + discrete.sum();

    loss.backward();

    bool any_grad = false;
    for (const auto &p: actor.parameters()) {
        if (p.grad().defined() && p.grad().abs().sum().item<float>() > 0) {
            any_grad = true;
            ASSERT_TRUE(torch::all(torch::isfinite(p.grad())).item<bool>())
                << "All actor gradients should be finite";
        }
    }
    ASSERT_TRUE(any_grad) << "At least some actor parameters should receive gradients";
}

TEST_F(ActorGradientTest, ActorWeightsChangeAfterOptimStep) {
    constexpr int height = 8, width = 8;
    constexpr int nb_sensors = 3, nb_cont = 2, nb_disc = 2;
    constexpr int batch = 2;

    Actor actor(height, width, nb_sensors, nb_cont, nb_disc, 8, {16}, {{3, 4}}, {2}, 0.1f, 0.2f);

    auto optimizer = torch::optim::Adam(actor.parameters(), 1e-3);

    auto params_before = std::vector<torch::Tensor>();
    for (const auto &p: actor.parameters()) params_before.push_back(p.clone());

    const auto vision = torch::randint(0, 255, {batch, 3, height, width}, torch::kUInt8);
    const auto sensors = torch::randn({batch, nb_sensors});

    const auto [alpha, beta, discrete] = actor.act(vision, sensors);
    const auto loss = -(alpha.sum() + beta.log().sum());

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    bool any_changed = false;
    int i = 0;
    for (const auto &p: actor.parameters()) {
        if (!torch::equal(p, params_before[i])) any_changed = true;
        i++;
    }
    ASSERT_TRUE(any_changed) << "Some actor parameters should change after optimizer step";
}

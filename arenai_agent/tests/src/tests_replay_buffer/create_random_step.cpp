//
// Created by samuel on 30/06/2026.
//

#include "./create_random_step.h"

using namespace arenai;
using namespace arenai::agent;

// single-tank state: every tensor carries a leading nb_tanks dimension of 1
TorchState create_random_state(const int width, const int height, const int nb_sensors) {
    return {
        .vision = torch::randint(255, {1, 3, height, width}, torch::kUInt8),
        .proprioception = torch::randn({1, nb_sensors})};
}

SacInputStep create_random_step(
    const int width, const int height, const int nb_cont_actions, const int nb_discrete_actions,
    const int nb_sensors, const bool done) {
    return {
        .state = create_random_state(width, height, nb_sensors),
        .action =
            {.continuous_action = torch::rand({1, nb_cont_actions}) * 2.f - 1.f,
             .discrete_action = torch::softmax(torch::randn({1, nb_discrete_actions}), -1)},
        .reward = torch::randn({1, 1}),
        .done = torch::full({1, 1}, done, torch::kBool)};
}

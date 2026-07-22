//
// Created by samuel on 30/06/2026.
//

#include "./create_random_step.h"

using namespace arenai;
using namespace arenai::agent;

// single-tank state: every tensor carries a leading nb_tanks dimension of 1
arenai::agent::TorchState
create_random_state(const int width, const int height, const int nb_sensors) {
    return {
        torch::randint(255, {1, 3, height, width}, torch::kUInt8), torch::randn({1, nb_sensors})};
}

arenai::agent::SacInputStep create_random_step(
    const int width, const int height, const int nb_cont_actions, const int nb_discrete_actions,
    const int nb_sensors, const bool done) {
    return {
        create_random_state(width, height, nb_sensors),
        {torch::rand({1, nb_cont_actions}) * 2.f - 1.f,
         torch::softmax(torch::randn({1, nb_discrete_actions}), -1)},
        torch::randn({1, 1}),
        torch::full({1, 1}, done, torch::kBool),
        torch::full({1, 1}, false, torch::kBool)};
}

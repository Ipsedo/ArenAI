//
// Created by samuel on 30/06/2026.
//

#include "./create_random_step.h"

TorchInputStep create_random_step(
    int width, int height, int nb_cont_actions, int nb_discrete_actions, int nb_sensors,
    bool done) {
    return {
        {torch::randint(255, {3, height, width}, torch::kUInt8), torch::randn({nb_sensors})},
        {torch::rand({nb_cont_actions}) * 2.f - 1.f,
         torch::softmax(torch::randn({nb_discrete_actions}), -1)},
        torch::randn({1}),
        torch::randn({1}),
        torch::tensor({done}),
        {torch::randint(255, {3, height, width}, torch::kUInt8), torch::randn({nb_sensors})}};
}

#include "./aim_head.h"

#include <cmath>

#include "./constants.h"

using namespace arenai;
using namespace arenai::agent;

namespace {
    // per cell: logit, mu_x, mu_y, log sigma_x, log sigma_y
    constexpr int NB_CELL_OUTPUTS = 5;
}// namespace

namespace arenai::agent {

    AimHead::AimHead(const int map_channels, const int state_size, const float initial_sigma)
        : cells(register_module(
            "cells", torch::nn::Conv2d(
                         torch::nn::Conv2dOptions(map_channels + state_size, NB_CELL_OUTPUTS, 1)))),
          log_sigma(register_module(
              "log_sigma", std::make_shared<SoftClamp>(LOG_SIGMA_MIN, LOG_SIGMA_MAX))) {
        torch::NoGradGuard no_grad;

        // uniform cells and centred offsets at the start
        torch::nn::init::orthogonal_(cells->weight, 0.01f);
        cells->bias.zero_();
        cells->bias.slice(0, 3, NB_CELL_OUTPUTS)
            .fill_(SoftClamp::inverse(std::log(initial_sigma), LOG_SIGMA_MIN, LOG_SIGMA_MAX));
    }

    SpatialAimLaw AimHead::forward(const torch::Tensor &map, const torch::Tensor &state) {
        const auto rows = map.size(2), cols = map.size(3);
        const auto broadcast_state =
            state.unsqueeze(-1).unsqueeze(-1).expand({state.size(0), state.size(1), rows, cols});

        // [N, R*C, 5]
        const auto outputs =
            cells->forward(torch::cat({map, broadcast_state}, 1)).flatten(2).transpose(1, 2);

        return {
            outputs.select(-1, 0), outputs.slice(-1, 1, 3),
            torch::exp(log_sigma->forward(outputs.slice(-1, 3, NB_CELL_OUTPUTS))),
            static_cast<int>(rows), static_cast<int>(cols)};
    }

}// namespace arenai::agent

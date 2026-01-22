//
// Created by samuel on 21/01/2026.
//

#include "./agent.h"

int AbstractAgent::count_parameters_impl(const std::vector<torch::Tensor> &params) {
    return std::accumulate(params.begin(), params.end(), 0, [](int c, auto tensor) {
        auto s = tensor.sizes();
        return c + std::accumulate(s.begin(), s.end(), 1, std::multiplies<int>());
    });
}

//
// Created by claude on 22/07/2026.
//

#include "./trainer.h"

#include <numeric>

using namespace arenai;
using namespace arenai::agent;

int AbstractTrainer::count_parameters_impl(const std::vector<torch::Tensor> &params) {
    return std::accumulate(params.begin(), params.end(), 0, [](long c, const auto &tensor) {
        auto s = tensor.sizes();
        return c + std::accumulate(s.begin(), s.end(), 1L, std::multiplies<long>());
    });
}

//
// Created by claude on 27/10/2026.
//

#include "./ppo_update.h"

#include <algorithm>

namespace arenai::agent {
    namespace {
        constexpr float KL_TRIM_FRACTION = 0.01f;
    }// namespace

    float trimmed_kl(const std::vector<torch::Tensor> &kl_per_rows) {
        const auto kl_per_row = torch::cat(kl_per_rows);

        const auto nb_kept = std::max<int64_t>(
            1, static_cast<int64_t>(
                   static_cast<float>(kl_per_row.size(0)) * (1.f - KL_TRIM_FRACTION)));

        // per-row KL is non-negative: sorting ascending puts the outliers past nb_kept
        return std::get<0>(torch::sort(kl_per_row)).slice(0, 0, nb_kept).mean().item<float>();
    }

    void clipped_step(
        torch::optim::Optimizer &optim, const std::vector<torch::Tensor> &parameters,
        const float grad_norm_max) {
        torch::nn::utils::clip_grad_norm_(parameters, grad_norm_max);
        optim.step();
    }

}// namespace arenai::agent

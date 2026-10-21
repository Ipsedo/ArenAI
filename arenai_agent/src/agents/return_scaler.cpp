//
// Created by samuel on 21/10/2026.
//

#include "./return_scaler.h"

#include "../networks/constants.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    ReturnScaler::ReturnScaler(const float gamma)
        : gamma(gamma), count(register_buffer("count", torch::full({1}, 1e-4, torch::kDouble))),
          mean(register_buffer("mean", torch::zeros({1}, torch::kDouble))),
          var(register_buffer("var", torch::ones({1}, torch::kDouble))) {}

    torch::Tensor ReturnScaler::scale(
        const torch::Tensor &rewards, const torch::Tensor &dones, const torch::Tensor &valids) {
        torch::NoGradGuard no_grad;

        const auto nb_steps = rewards.size(0);
        const auto nb_tanks = rewards.size(1);

        const auto float_rewards = rewards.to(torch::kFloat);
        const auto float_dones = dones.to(torch::kFloat);
        const auto float_valids = valids.to(torch::kFloat);

        if (!discounted_returns.defined() || discounted_returns.size(0) != nb_tanks)
            discounted_returns = torch::zeros({nb_tanks, 1});

        std::vector<torch::Tensor> returns_list;
        returns_list.reserve(nb_steps);

        for (int64_t t = 0; t < nb_steps; t++) {
            discounted_returns = gamma * discounted_returns + float_rewards[t] * float_valids[t];
            returns_list.push_back(discounted_returns);
            discounted_returns = discounted_returns * (1.f - float_dones[t]);
        }

        // live transitions only: the post-mortem steps carry no return
        const auto valid_idx = torch::nonzero(float_valids.flatten() > 0.f).squeeze(-1);
        const auto returns =
            torch::stack(returns_list, 0).flatten().index_select(0, valid_idx).to(torch::kDouble);

        if (const auto batch_count = returns.size(0); batch_count > 0) {
            // parallel variance update (Chan et al.)
            const auto batch_mean = returns.mean();
            const auto batch_var = returns.var(false);

            const auto delta = batch_mean - mean;
            const auto total = count + static_cast<double>(batch_count);

            const auto m2 = var * count + batch_var * static_cast<double>(batch_count)
                            + delta.square() * count * static_cast<double>(batch_count) / total;

            mean.copy_(mean + delta * static_cast<double>(batch_count) / total);
            var.copy_(m2 / total);
            count.copy_(total);
        }

        return float_rewards / torch::sqrt(var + EPSILON).to(torch::kFloat);
    }

}// namespace arenai::agent

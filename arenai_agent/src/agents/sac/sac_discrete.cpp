//
// Created by claude on 08/11/2026.
//

#include "./sac_discrete.h"

#include "../../distributions/bernoulli.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    torch::Tensor
    discrete_combinations(const int64_t nb_actions, const torch::TensorOptions &options) {
        const auto long_options = options.dtype(torch::kLong);

        const auto indices = torch::arange(int64_t{1} << nb_actions, long_options).unsqueeze(1);
        const auto bits = torch::arange(nb_actions, long_options).unsqueeze(0);

        return torch::bitwise_and(torch::bitwise_right_shift(indices, bits), 1).to(options.dtype());
    }

    torch::Tensor combination_index(const torch::Tensor &discrete_actions) {
        const auto weights = torch::pow(
            2.f, torch::arange(
                     discrete_actions.size(-1), discrete_actions.options().dtype(torch::kFloat)));
        return (discrete_actions.to(torch::kFloat) * weights).sum(-1).round().to(torch::kLong);
    }

    DiscreteJoint discrete_joint(const torch::Tensor &probas, const torch::Tensor &mask) {
        // [1, K, nb_actions] against [N, 1, nb_actions]
        const auto combinations =
            discrete_combinations(probas.size(-1), probas.options()).unsqueeze(0);

        const auto effective_probas = (probas * mask).unsqueeze(1);
        const auto log_effective = bernoulli_log_proba(combinations, effective_probas).sum(-1);

        const auto action_log_probas =
            bernoulli_log_proba(combinations, probas.unsqueeze(1)) * mask.unsqueeze(1);

        return {
            .probas = torch::exp(log_effective),
            .action_log_probas = action_log_probas,
            .log_probas = action_log_probas.sum(-1)};
    }

    torch::Tensor discrete_joint_entropy(const DiscreteJoint &joint) {
        return -(joint.probas * joint.log_probas).sum(-1, true);
    }

    torch::Tensor discrete_action_entropies(const DiscreteJoint &joint) {
        return -(joint.probas.unsqueeze(-1) * joint.action_log_probas).sum(1);
    }

}// namespace arenai::agent

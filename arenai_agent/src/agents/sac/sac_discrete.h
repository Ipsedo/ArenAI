//
// Created by claude on 08/11/2026.
//

#ifndef ARENAI_SAC_DISCRETE_H
#define ARENAI_SAC_DISCRETE_H

#include <torch/torch.h>

namespace arenai::agent {

    // Joint law of the independent binary discrete actions, enumerated over their
    // 2^nb_actions combinations: SAC takes exact expectations over it instead of sampling.

    // [2^nb_actions, nb_actions]: action j of combination k is bit j of k
    torch::Tensor discrete_combinations(int64_t nb_actions, const torch::TensorOptions &options);

    // [N, nb_actions] binary actions -> [N] index of their combination
    torch::Tensor combination_index(const torch::Tensor &discrete_actions);

    struct DiscreteJoint {
        // [N, K] probability of each combination; an action without effect (mask 0) is
        // forced to 0, so the combinations engaging it get no weight
        torch::Tensor probas;
        // [N, K, nb_actions] log-probability of each action in each combination, counting
        // only the actions that have an effect: a fire that cannot shoot is not a decision,
        // it brings no log-probability
        torch::Tensor action_log_probas;
        // [N, K] log-probability of each combination: the sum over its actions
        torch::Tensor log_probas;
    };

    // probas / mask: [N, nb_actions]
    DiscreteJoint discrete_joint(const torch::Tensor &probas, const torch::Tensor &mask);

    // [N, 1] entropy of the joint law over the actions that have an effect
    torch::Tensor discrete_joint_entropy(const DiscreteJoint &joint);

    // [N, nb_actions] entropy of each action's marginal law (0 for an action without
    // effect); the actions being independent, they sum to the joint entropy
    torch::Tensor discrete_action_entropies(const DiscreteJoint &joint);

}// namespace arenai::agent

#endif//ARENAI_SAC_DISCRETE_H

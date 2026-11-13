#ifndef ARENAI_SAC_DISCRETE_H
#define ARENAI_SAC_DISCRETE_H

#include <torch/torch.h>

namespace arenai::agent {

    // [2^nb_actions, nb_actions]: action j of combination k is bit j of k
    torch::Tensor discrete_combinations(int64_t nb_actions, const torch::TensorOptions &options);

    // [N, nb_actions] binary actions -> [N] index of their combination
    torch::Tensor combination_index(const torch::Tensor &discrete_actions);

    struct DiscreteJoint {
        torch::Tensor probas;
        torch::Tensor action_log_probas;
        // [N, K] log-probability of each combination: the sum over its actions
        torch::Tensor log_probas;
    };

    // probas / mask: [N, nb_actions]
    DiscreteJoint discrete_joint(const torch::Tensor &probas, const torch::Tensor &mask);

    // [N, 1] entropy of the joint law over the actions that have an effect
    torch::Tensor discrete_joint_entropy(const DiscreteJoint &joint);

    torch::Tensor discrete_action_entropies(const DiscreteJoint &joint);

}// namespace arenai::agent

#endif//ARENAI_SAC_DISCRETE_H

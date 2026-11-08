//
// Created by claude on 08/11/2026.
//

#ifndef ARENAI_ENTROPY_ALPHAS_H
#define ARENAI_ENTROPY_ALPHAS_H

#include <filesystem>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "../metrics/metric.h"
#include "../networks/entropy.h"

namespace arenai::agent {

    // Learned entropy temperatures of a PPO-like actor, one per action: each alpha follows
    // its own dual ascent toward H_action >= target, a constant target per action kind
    // (an entropy per continuous action, a Bernoulli probability per discrete action).
    class EntropyAlphas {
    public:
        EntropyAlphas(
            int nb_continuous_actions, int nb_discrete_actions, float initial_alpha,
            float learning_rate, float continuous_target_entropy, float discrete_target_proba);

        // [nb_continuous_actions] and [nb_discrete_actions], attached to log_alpha
        torch::Tensor continuous() const;
        torch::Tensor discrete() const;

        void zero_grad() const;

        // accumulates a device block's share of the dual gradient: entropies are
        // [N, nb_actions], the discrete ones weighted by their [N, nb_discrete_actions] mask
        void backward(
            const torch::Tensor &continuous_entropy, const torch::Tensor &discrete_entropy,
            const torch::Tensor &discrete_mask, float batch_fraction) const;

        void step() const;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() const;

        void save(const std::filesystem::path &output_folder) const;
        void to(torch::Device device) const;
        int count_parameters() const;

    private:
        std::shared_ptr<AlphaParameters> continuous_alphas;
        std::shared_ptr<AlphaParameters> discrete_alphas;
        std::unique_ptr<torch::optim::Adam> optim;

        float continuous_target_entropy;
        float discrete_target_entropy;

        std::vector<std::shared_ptr<AbstractMetric>> continuous_metrics;
        std::vector<std::shared_ptr<AbstractMetric>> discrete_metrics;
    };

}// namespace arenai::agent

#endif//ARENAI_ENTROPY_ALPHAS_H

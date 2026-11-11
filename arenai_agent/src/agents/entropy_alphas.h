#ifndef ARENAI_ENTROPY_ALPHAS_H
#define ARENAI_ENTROPY_ALPHAS_H

#include <filesystem>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "../metrics/metric.h"
#include "../networks/entropy.h"

namespace arenai::agent {

    class EntropyAlphas {
    public:
        EntropyAlphas(
            int nb_continuous_actions, int nb_discrete_actions, float initial_alpha,
            float learning_rate, const std::vector<float> &continuous_target_entropy,
            float discrete_target_proba);

        // [nb_continuous_actions] and [nb_discrete_actions], attached to log_alpha
        torch::Tensor continuous() const;
        torch::Tensor discrete() const;

        void zero_grad() const;

        void backward(
            const torch::Tensor &continuous_entropy, const torch::Tensor &discrete_entropy,
            const torch::Tensor &discrete_mask, float batch_fraction) const;

        void step() const;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() const;

        void save(const std::filesystem::path &output_folder) const;
        void load(const std::filesystem::path &input_folder) const;
        void to(torch::Device device) const;
        int count_parameters() const;

    private:
        std::shared_ptr<AlphaParameters> continuous_alphas;
        std::shared_ptr<AlphaParameters> discrete_alphas;
        std::unique_ptr<torch::optim::Adam> optim;

        // [nb_continuous_actions]
        mutable torch::Tensor continuous_target_entropy;
        float discrete_target_entropy;

        std::vector<std::shared_ptr<AbstractMetric>> continuous_metrics;
        std::vector<std::shared_ptr<AbstractMetric>> discrete_metrics;
    };

}// namespace arenai::agent

#endif//ARENAI_ENTROPY_ALPHAS_H

#ifndef ARENAI_AIM_ENTROPY_ALPHAS_H
#define ARENAI_AIM_ENTROPY_ALPHAS_H

#include <filesystem>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "../metrics/metric.h"
#include "../networks/entropy.h"

namespace arenai::agent {

    class AimEntropyAlphas {
    public:
        AimEntropyAlphas(
            float initial_alpha, float learning_rate, float cell_target_entropy,
            const std::vector<float> &offset_target_entropy);

        // [1] and [2], attached to log_alpha
        torch::Tensor cell() const;
        torch::Tensor offset() const;

        void zero_grad() const;

        // cell [N, 1], offset [N, 2]
        void backward(
            const torch::Tensor &cell_entropy, const torch::Tensor &offset_entropy,
            float batch_fraction) const;

        void step() const;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() const;

        void save(const std::filesystem::path &output_folder) const;
        void load(const std::filesystem::path &input_folder) const;
        void to(torch::Device device) const;
        int count_parameters() const;

    private:
        std::shared_ptr<AlphaParameters> cell_alpha;
        std::shared_ptr<AlphaParameters> offset_alphas;
        std::unique_ptr<torch::optim::Adam> optim;

        float cell_target_entropy;
        // [2]
        mutable torch::Tensor offset_target_entropy;

        std::vector<std::shared_ptr<AbstractMetric>> metrics;
    };

}// namespace arenai::agent

#endif//ARENAI_AIM_ENTROPY_ALPHAS_H

#ifndef ARENAI_PANORAMA_MASK_LEARNER_H
#define ARENAI_PANORAMA_MASK_LEARNER_H

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include <torch/torch.h>

#include "../core/train_environment.h"
#include "../metrics/metric.h"
#include "../networks/panorama_mask.h"

namespace arenai::agent {

    // trains the panorama encoder alone, on the mask of the enemies visible per bearing column
    class PanoramaMaskLearner {
    public:
        PanoramaMaskLearner(
            const std::shared_ptr<PanoramaMaskNetwork> &encoder, float learning_rate,
            int batch_size, int epochs, torch::Device device);

        // panorama [nb_tanks, 3, PH, PW] uint8
        void observe(
            const torch::Tensor &panorama,
            const std::vector<std::optional<std::vector<EnemySighting>>> &sightings);

        // epochs passes over the observations since the last fit
        void fit();

        void save(const std::filesystem::path &output_folder) const;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() const;

    private:
        std::shared_ptr<PanoramaMaskNetwork> encoder;
        std::unique_ptr<torch::optim::Adam> optim;
        int batch_size;
        int epochs;
        torch::Device device;

        std::shared_ptr<AbstractMetric> loss_metric;

        // one entry per live tank observed: [3, PH, PW] uint8, [columns], [columns]
        std::vector<torch::Tensor> panoramas;
        std::vector<torch::Tensor> labels;
        std::vector<torch::Tensor> weights;
    };

}// namespace arenai::agent

#endif//ARENAI_PANORAMA_MASK_LEARNER_H

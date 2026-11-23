#ifndef ARENAI_PRESENCE_LEARNER_H
#define ARENAI_PRESENCE_LEARNER_H

#include <filesystem>
#include <memory>
#include <string>

#include <torch/torch.h>

#include "../metrics/metric.h"
#include "../networks/presence_model.h"

namespace arenai::agent {

    // supervised enemy presence map, trained through the presence parameters alone
    class PresenceLearner {
    public:
        PresenceLearner(
            const std::shared_ptr<PresenceModel> &model, float learning_rate,
            const std::shared_ptr<AbstractMetric> &loss_metric);

        // one pass: vision [rows, C, H, W], presence [rows, grid rows, grid cols]
        void
        fit(const torch::Tensor &vision, const torch::Tensor &presence, const torch::Tensor &rows,
            int64_t gpu_batch_size) const;

        void save(const std::filesystem::path &output_folder, const std::string &file_name) const;

    private:
        std::shared_ptr<PresenceModel> model;
        std::unique_ptr<torch::optim::Adam> optim;
        std::shared_ptr<AbstractMetric> loss_metric;
    };

}// namespace arenai::agent

#endif//ARENAI_PRESENCE_LEARNER_H

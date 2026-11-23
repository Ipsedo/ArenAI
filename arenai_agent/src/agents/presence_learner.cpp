#include "./presence_learner.h"

#include "../networks_utils/torch_saver.h"

namespace arenai::agent {

    PresenceLearner::PresenceLearner(
        const std::shared_ptr<PresenceModel> &model, const float learning_rate,
        const std::shared_ptr<AbstractMetric> &loss_metric)
        : model(model),
          optim(std::make_unique<torch::optim::Adam>(model->presence_parameters(), learning_rate)),
          loss_metric(loss_metric) {}

    void PresenceLearner::fit(
        const torch::Tensor &vision_rows, const torch::Tensor &presence, const torch::Tensor &rows,
        const int64_t gpu_batch_size) const {
        const auto shuffled = rows.index_select(
            0, torch::randperm(rows.size(0), torch::TensorOptions().device(rows.device())));

        for (const auto &idx: shuffled.split(gpu_batch_size)) {
            optim->zero_grad();

            const auto loss = torch::binary_cross_entropy_with_logits(
                model->presence_logits(vision_rows.index_select(0, idx)),
                presence.index_select(0, idx));

            loss.backward();
            optim->step();

            loss_metric->add(loss.item<float>());
        }
    }

    void PresenceLearner::save(
        const std::filesystem::path &output_folder, const std::string &file_name) const {
        save_torch(output_folder, optim, file_name);
    }

}// namespace arenai::agent

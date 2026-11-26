#include "./panorama_mask_learner.h"

#include "../core/panorama_mask_labels.h"
#include "../metrics/mean_metric.h"
#include "../networks_utils/torch_saver.h"

namespace arenai::agent {

    namespace {
        constexpr int LOSS_WINDOW_SIZE = 64;
    }// namespace

    PanoramaMaskLearner::PanoramaMaskLearner(
        const std::shared_ptr<PanoramaMaskNetwork> &encoder, const float learning_rate,
        const int batch_size, const int epochs, const torch::Device device)
        : encoder(encoder),
          optim(std::make_unique<torch::optim::Adam>(encoder->parameters(), learning_rate)),
          batch_size(batch_size), epochs(epochs), device(device),
          loss_metric(std::make_shared<MeanMetric>("mask_loss", LOSS_WINDOW_SIZE)) {}

    void PanoramaMaskLearner::observe(
        const torch::Tensor &panorama,
        const std::vector<std::optional<std::vector<EnemySighting>>> &sightings) {
        const auto cpu_panorama = panorama.detach().cpu();

        for (size_t tank = 0; tank < sightings.size(); tank++) {
            if (!sightings[tank].has_value()) continue;

            const auto mask = panorama_mask_labels(*sightings[tank], encoder->get_output_size());

            panoramas.push_back(cpu_panorama[static_cast<int64_t>(tank)]);
            labels.push_back(mask.labels);
            weights.push_back(mask.weights);
        }
    }

    void PanoramaMaskLearner::fit() {
        if (panoramas.empty()) return;

        const auto all_panoramas = torch::stack(panoramas);
        const auto all_labels = torch::stack(labels);
        const auto all_weights = torch::stack(weights);

        for (int epoch = 0; epoch < epochs; epoch++)
            for (const auto &idx: torch::randperm(all_panoramas.size(0)).split(batch_size)) {
                const auto batch_weights = all_weights.index_select(0, idx).to(device);

                const auto loss =
                    (torch::binary_cross_entropy_with_logits(
                         encoder->logits(all_panoramas.index_select(0, idx).to(device)),
                         all_labels.index_select(0, idx).to(device), {}, {}, at::Reduction::None)
                     * batch_weights)
                        .sum()
                    / batch_weights.sum().clamp_min(1.f);

                optim->zero_grad();
                loss.backward();
                optim->step();

                loss_metric->add(loss.item<float>());
            }

        panoramas.clear();
        labels.clear();
        weights.clear();
    }

    void PanoramaMaskLearner::save(const std::filesystem::path &output_folder) const {
        save_torch(output_folder, optim, "panorama_mask_optim.pt");
    }

    std::vector<std::shared_ptr<AbstractMetric>> PanoramaMaskLearner::get_metrics() const {
        return {loss_metric};
    }

}// namespace arenai::agent

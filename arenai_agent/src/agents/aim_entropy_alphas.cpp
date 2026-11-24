#include "./aim_entropy_alphas.h"

#include "../metrics/last_metric.h"
#include "../networks_utils/torch_loader.h"
#include "../networks_utils/torch_saver.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    namespace {
        constexpr int NB_OFFSET_DIMS = 2;

        std::vector<torch::Tensor> all_parameters(
            const std::shared_ptr<AlphaParameters> &cell_alpha,
            const std::shared_ptr<AlphaParameters> &offset_alphas) {
            auto parameters = cell_alpha->parameters();
            const auto offset_parameters = offset_alphas->parameters();
            parameters.insert(parameters.end(), offset_parameters.begin(), offset_parameters.end());
            return parameters;
        }

        void record(
            const std::vector<std::shared_ptr<AbstractMetric>> &metrics,
            const torch::Tensor &alphas) {
            const auto values = alphas.detach().cpu();
            for (int i = 0; i < static_cast<int>(metrics.size()); i++)
                metrics[i]->add(values[i].item<float>());
        }
    }// namespace

    AimEntropyAlphas::AimEntropyAlphas(
        const float initial_alpha, const float learning_rate, const float cell_target_entropy,
        const std::vector<float> &offset_target_entropy)
        : cell_alpha(std::make_shared<AlphaParameters>(initial_alpha, 1)),
          offset_alphas(std::make_shared<AlphaParameters>(initial_alpha, NB_OFFSET_DIMS)),
          optim(std::make_unique<torch::optim::Adam>(
              all_parameters(cell_alpha, offset_alphas), learning_rate)),
          cell_target_entropy(cell_target_entropy),
          offset_target_entropy(torch::tensor(offset_target_entropy)),
          metrics{
              std::make_shared<LastMetric>("alpha_aim_cell", 2, true),
              std::make_shared<LastMetric>("alpha_aim_dx", 2, true),
              std::make_shared<LastMetric>("alpha_aim_dy", 2, true)} {
        if (static_cast<int>(offset_target_entropy.size()) != NB_OFFSET_DIMS)
            throw std::invalid_argument(
                "aim_offset_target_entropy needs " + std::to_string(NB_OFFSET_DIMS)
                + " values, got " + std::to_string(offset_target_entropy.size()));
        record(metrics, torch::cat({cell(), offset()}));
    }

    torch::Tensor AimEntropyAlphas::cell() const { return cell_alpha->alpha(); }

    torch::Tensor AimEntropyAlphas::offset() const { return offset_alphas->alpha(); }

    void AimEntropyAlphas::zero_grad() const { optim->zero_grad(); }

    void AimEntropyAlphas::backward(
        const torch::Tensor &cell_entropy, const torch::Tensor &offset_entropy,
        const float batch_fraction) const {
        // entropy errors, detached from the actor's graph
        const auto loss =
            cell_alpha->loss((cell_entropy.detach() - cell_target_entropy).mean(0))
            + offset_alphas->loss((offset_entropy.detach() - offset_target_entropy).mean(0));

        // weighted so that the accumulated gradient is the one of the whole batch's mean
        (loss * batch_fraction).backward();
    }

    void AimEntropyAlphas::step() const {
        optim->step();
        cell_alpha->clamp();
        offset_alphas->clamp();

        record(metrics, torch::cat({cell(), offset()}));
    }

    std::vector<std::shared_ptr<AbstractMetric>> AimEntropyAlphas::get_metrics() const {
        return metrics;
    }

    void AimEntropyAlphas::save(const std::filesystem::path &output_folder) const {
        save_torch(output_folder, cell_alpha, "aim_cell_alpha.pt");
        save_torch(output_folder, offset_alphas, "aim_offset_alphas.pt");
        save_torch(output_folder, optim, "aim_alpha_optim.pt");
    }

    void AimEntropyAlphas::load(const std::filesystem::path &input_folder) const {
        load_torch(input_folder, cell_alpha, "aim_cell_alpha.pt");
        load_torch(input_folder, offset_alphas, "aim_offset_alphas.pt");
    }

    void AimEntropyAlphas::to(const torch::Device device) const {
        cell_alpha->to(device);
        offset_alphas->to(device);
        offset_target_entropy = offset_target_entropy.to(device);
    }

    int AimEntropyAlphas::count_parameters() const {
        int count = 0;
        for (const auto &parameter: all_parameters(cell_alpha, offset_alphas))
            count += static_cast<int>(parameter.numel());
        return count;
    }

}// namespace arenai::agent

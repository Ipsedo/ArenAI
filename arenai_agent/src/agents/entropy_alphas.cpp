#include "./entropy_alphas.h"

#include "../distributions/bernoulli.h"
#include "../metrics/last_metric.h"
#include "../networks_utils/torch_saver.h"
#include "./torch_types.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    namespace {
        std::vector<std::shared_ptr<AbstractMetric>>
        make_metrics(const std::vector<std::string> &action_names, const int nb_actions) {
            std::vector<std::shared_ptr<AbstractMetric>> metrics;
            metrics.reserve(nb_actions);
            for (int i = 0; i < nb_actions; i++) {
                const auto name =
                    i < static_cast<int>(action_names.size()) ? action_names[i] : std::to_string(i);
                metrics.push_back(std::make_shared<LastMetric>("alpha_" + name, 2, true));
            }
            return metrics;
        }

        std::vector<torch::Tensor> all_parameters(
            const std::shared_ptr<AlphaParameters> &continuous_alphas,
            const std::shared_ptr<AlphaParameters> &discrete_alphas) {
            auto parameters = continuous_alphas->parameters();
            const auto discrete_parameters = discrete_alphas->parameters();
            parameters.insert(
                parameters.end(), discrete_parameters.begin(), discrete_parameters.end());
            return parameters;
        }

        void record(
            const std::vector<std::shared_ptr<AbstractMetric>> &metrics,
            const torch::Tensor &alpha) {
            const auto values = alpha.detach().cpu();
            for (int64_t i = 0; i < values.size(0); i++) metrics[i]->add(values[i].item<float>());
        }
    }// namespace

    EntropyAlphas::EntropyAlphas(
        const int nb_continuous_actions, const int nb_discrete_actions, const float initial_alpha,
        const float learning_rate, const float continuous_target_entropy,
        const float discrete_target_proba)
        : continuous_alphas(
            std::make_shared<AlphaParameters>(initial_alpha, nb_continuous_actions)),
          discrete_alphas(std::make_shared<AlphaParameters>(initial_alpha, nb_discrete_actions)),
          optim(std::make_unique<torch::optim::Adam>(
              all_parameters(continuous_alphas, discrete_alphas), learning_rate)),
          continuous_target_entropy(continuous_target_entropy),
          discrete_target_entropy(
              bernoulli_entropy(torch::tensor(discrete_target_proba)).item<float>()),
          continuous_metrics(
              make_metrics(TorchAction::continuous_action_names(), nb_continuous_actions)),
          discrete_metrics(
              make_metrics(TorchAction::discrete_action_names(), nb_discrete_actions)) {
        record(continuous_metrics, continuous());
        record(discrete_metrics, discrete());
    }

    torch::Tensor EntropyAlphas::continuous() const { return continuous_alphas->alpha(); }

    torch::Tensor EntropyAlphas::discrete() const { return discrete_alphas->alpha(); }

    void EntropyAlphas::zero_grad() const { optim->zero_grad(); }

    void EntropyAlphas::backward(
        const torch::Tensor &continuous_entropy, const torch::Tensor &discrete_entropy,
        const torch::Tensor &discrete_mask, const float batch_fraction) const {
        const auto continuous_error =
            (continuous_entropy.detach() - continuous_target_entropy).mean(0);

        const auto discrete_error =
            ((discrete_entropy.detach() - discrete_target_entropy) * discrete_mask).sum(0)
            / discrete_mask.sum(0).clamp_min(1.f);

        const auto loss =
            continuous_alphas->loss(continuous_error) + discrete_alphas->loss(discrete_error);

        (loss * batch_fraction).backward();
    }

    void EntropyAlphas::step() const {
        optim->step();
        continuous_alphas->clamp();
        discrete_alphas->clamp();

        record(continuous_metrics, continuous());
        record(discrete_metrics, discrete());
    }

    std::vector<std::shared_ptr<AbstractMetric>> EntropyAlphas::get_metrics() const {
        auto metrics = continuous_metrics;
        metrics.insert(metrics.end(), discrete_metrics.begin(), discrete_metrics.end());
        return metrics;
    }

    void EntropyAlphas::save(const std::filesystem::path &output_folder) const {
        save_torch(output_folder, continuous_alphas, "continuous_alphas.pt");
        save_torch(output_folder, discrete_alphas, "discrete_alphas.pt");
        save_torch(output_folder, optim, "alpha_optim.pt");
    }

    void EntropyAlphas::to(const torch::Device device) const {
        continuous_alphas->to(device);
        discrete_alphas->to(device);
    }

    int EntropyAlphas::count_parameters() const {
        int count = 0;
        for (const auto &parameter: all_parameters(continuous_alphas, discrete_alphas))
            count += static_cast<int>(parameter.numel());
        return count;
    }

}// namespace arenai::agent

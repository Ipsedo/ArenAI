//
// Created by samuel on 24/10/2026.
//

#include "./per_action_metrics.h"

#include <stdexcept>

#include "./mean_metric.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    std::vector<std::shared_ptr<AbstractMetric>> make_per_action_metrics(
        const std::string &prefix, const std::vector<std::string> &action_names,
        const int nb_actions, const int window_size) {
        std::vector<std::shared_ptr<AbstractMetric>> metrics;
        metrics.reserve(nb_actions);

        for (int i = 0; i < nb_actions; i++) {
            const auto suffix = i < action_names.size() ? action_names[i] : std::to_string(i);
            metrics.push_back(
                std::make_shared<MeanMetric>(prefix + "_" + suffix, window_size, 3, false, false));
        }

        return metrics;
    }

    void add_per_action_values(
        const std::vector<std::shared_ptr<AbstractMetric>> &metrics, const torch::Tensor &values) {
        if (values.dim() != 1 || values.size(0) != static_cast<int64_t>(metrics.size()))
            throw std::invalid_argument(
                "add_per_action_values: expected a 1-D tensor of size "
                + std::to_string(metrics.size()));

        const auto cpu_values = values.detach().cpu().to(torch::kFloat).contiguous();
        const auto accessor = cpu_values.accessor<float, 1>();

        for (int i = 0; i < metrics.size(); i++) metrics[i]->add(accessor[i]);
    }

}// namespace arenai::agent

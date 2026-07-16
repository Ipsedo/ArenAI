//
// Created by samuel on 11/06/2026.
//

#include "./std_metric.h"

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

    StdMetric::StdMetric(const std::string &name, const int window_size)
        : AbstractMetric(name, window_size, 2, true), mean_metric(name + "_mean", window_size) {}

    float StdMetric::compute_metric_impl(const std::vector<float> &curr_values) {
        const auto curr_mean = mean_metric.compute_metric();

        return std::sqrt(
            std::accumulate(
                curr_values.begin(), curr_values.end(), 0.0f,
                [curr_mean](const float a, const float b) {
                    return a + std::pow(b - curr_mean, 2.0);
                })
            / std::max(static_cast<float>(curr_values.size()), 1.f));
    }

    float StdMetric::to_stored_value(const float value) {
        mean_metric.add(value);
        return AbstractMetric::to_stored_value(value);
    }

}// namespace arenai::train

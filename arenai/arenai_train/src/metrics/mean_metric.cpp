//
// Created by samuel on 11/06/2026.
//

#include "./mean_metric.h"

MeanMetric::MeanMetric(
    const std::string &name, const int window_size, const int precision, const bool scientific)
    : AbstractMetric(name, window_size, precision, scientific) {}

float MeanMetric::compute_metric_impl(const std::vector<float> &curr_values) {
    return std::reduce(curr_values.begin(), curr_values.end(), 0.0f, std::plus())
           / std::max(static_cast<float>(curr_values.size()), 1.f);
}

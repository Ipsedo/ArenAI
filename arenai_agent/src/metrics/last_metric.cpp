//
// Created by samuel on 11/07/2026.
//

#include "./last_metric.h"

using namespace arenai;
using namespace arenai::agent;

LastMetric::LastMetric(const std::string &name, const int precision, const bool scientific)
    : AbstractMetric(name, 1, precision, scientific) {}

float LastMetric::compute_metric_impl(const std::vector<float> &curr_values) {
    return curr_values.size() > 0 ? curr_values.back() : 0.f;
}

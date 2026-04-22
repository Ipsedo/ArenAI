//
// Created by samuel on 12/10/2025.
//

#include <numeric>

#include <arenai_train/metric.h>

Metric::Metric(
    const std::string &name, const int window_size, const int precision, const bool scientific)
    : name(name), window_size(window_size), float_display_scientific(scientific),
      float_display_precision(precision) {}

float Metric::last_value() const { return values.back(); }

float Metric::average_value() {
    return std::reduce(values.begin(), values.end(), 0.0f, std::plus())
           / std::max(static_cast<float>(values.size()), 1.f);
}

void Metric::add(const float value) {
    values.push_back(value);

    while (values.size() > window_size) values.erase(values.begin());
}

std::string Metric::to_string() {
    const auto float_display_format = float_display_scientific ? std::scientific : std::fixed;
    std::stringstream stream;
    stream << name << " = " << std::setprecision(float_display_precision) << float_display_format
           << average_value();

    return stream.str();
}

std::string Metric::metrics_to_string(const std::vector<std::shared_ptr<Metric>> &metrics) {
    std::stringstream stream;

    for (int i = 0; i < metrics.size(); i++) {
        if (i > 0) stream << ", ";
        stream << metrics[i]->to_string();
    }

    stream << " ";

    return stream.str();
}

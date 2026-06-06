//
// Created by samuel on 12/10/2025.
//

#include <numeric>

#include <arenai_train/metric.h>

Metric::Metric(
    const std::string &name, const int window_size, const int precision, const bool scientific,
    const bool print_std)
    : name(name), window_size(window_size), float_display_scientific(scientific),
      float_display_precision(precision), print_std(print_std) {}

float Metric::last_value() const { return values.back(); }

float Metric::mean() {
    return std::reduce(values.begin(), values.end(), 0.0f, std::plus())
           / std::max(static_cast<float>(values.size()), 1.f);
}

float Metric::std() {
    const float curr_mean = mean();

    return std::sqrt(
        std::accumulate(
            values.begin(), values.end(), 0.0f,
            [curr_mean](const float a, const float b) { return a + std::pow(b - curr_mean, 2.0); })
        / std::max(static_cast<float>(values.size()), 1.f));
}

void Metric::add(const float value) {
    values.push_back(value);

    while (values.size() > window_size) values.erase(values.begin());
}

std::string Metric::to_string() {
    const auto float_display_format = float_display_scientific ? std::scientific : std::fixed;
    std::stringstream stream;
    stream << name << " = " << std::setprecision(float_display_precision) << float_display_format
           << mean();

    if (print_std) stream << " ±" << std::setprecision(1) << std::scientific << std();

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

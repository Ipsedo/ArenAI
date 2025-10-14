//
// Created by samuel on 12/10/2025.
//

#include "./metric.h"

#include <numeric>

Metric::Metric(const std::string &name, const int window_size)
    : name(name), window_size(window_size) {}

float Metric::last_value() const { return values.back(); }

float Metric::average_value() {
    return std::reduce(values.begin(), values.end(), 0.0f, std::plus<>())
           / static_cast<float>(values.size());
}

void Metric::add(const float value) {
    values.push_back(value);

    while (values.size() > window_size) values.erase(values.begin());
}

std::string Metric::to_string() {
    std::stringstream stream;
    stream << name << " = " << std::setprecision(6) << std::fixed << average_value();
    return stream.str();
}

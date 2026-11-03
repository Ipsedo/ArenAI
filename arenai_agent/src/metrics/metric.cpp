//
// Created by samuel on 12/10/2025.
//

#include "./metric.h"

#include <iomanip>
#include <sstream>
#include <stdexcept>

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    AbstractMetric::AbstractMetric(
        const std::string &name, const int window_size, const int precision, const bool scientific,
        const bool print_in_progress_bar)
        : name(name), window_size(window_size), float_display_scientific(scientific),
          float_display_precision(precision), print_in_progress_bar(print_in_progress_bar) {}

    void AbstractMetric::add(const float value) {
        values.push_back(to_stored_value(value));
        while (values.size() > window_size) values.erase(values.begin());
    }

    float AbstractMetric::compute_metric() { return compute_metric_impl(values); }

    float AbstractMetric::to_stored_value(const float value) { return value; }

    float AbstractMetric::last_value() const {
        if (values.empty()) throw std::runtime_error("last_value() called on empty metric");
        return values.back();
    }

    std::string AbstractMetric::get_name() const { return name; }

    bool AbstractMetric::is_printed_in_progress_bar() const { return print_in_progress_bar; }

    std::string AbstractMetric::to_string() {
        const auto float_display_format = float_display_scientific ? std::scientific : std::fixed;
        std::stringstream stream;
        stream << get_name() << " = " << std::setprecision(float_display_precision)
               << float_display_format << compute_metric();

        return stream.str();
    }

    std::string
    AbstractMetric::metrics_to_string(const std::vector<std::shared_ptr<AbstractMetric>> &metrics) {
        std::stringstream stream;

        bool first = true;
        for (const auto &metric: metrics) {
            if (!metric->is_printed_in_progress_bar()) continue;
            if (!first) stream << ", ";
            stream << metric->to_string();
            first = false;
        }

        stream << " ";

        return stream.str();
    }

}// namespace arenai::agent

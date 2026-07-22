//
// Created by samuel on 11/07/2026.
//

#ifndef ARENAI_LAST_METRIC_H
#define ARENAI_LAST_METRIC_H

#include "./metric.h"

namespace arenai::agent {
    class LastMetric final : public AbstractMetric {
    public:
        explicit LastMetric(const std::string &name, int precision = 4, bool scientific = false);
        ~LastMetric() override = default;

    protected:
        float compute_metric_impl(const std::vector<float> &curr_values) override;
    };
}// namespace arenai::agent

#endif//ARENAI_LAST_METRIC_H

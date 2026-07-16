//
// Created by samuel on 11/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_MEAN_METRIC_H
#define ARENAI_TRAIN_HOST_MEAN_METRIC_H

#include <arenai_train/metric.h>

namespace arenai::train {

    class MeanMetric : public AbstractMetric {
    public:
        MeanMetric(
            const std::string &name, int window_size, int precision = 3, bool scientific = false);

    protected:
        float compute_metric_impl(const std::vector<float> &curr_values) override;
    };

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_MEAN_METRIC_H

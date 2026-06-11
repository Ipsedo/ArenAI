//
// Created by samuel on 11/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_MEAN_METRIC_H
#define ARENAI_TRAIN_HOST_MEAN_METRIC_H

#include <arenai_train/metric.h>

class MeanMetric : public AbstractMetric {
public:
    MeanMetric(
        const std::string &name, int window_size, int precision = 4, bool scientific = false);

protected:
    float compute_metric_impl(const std::vector<float> &curr_values) override;
};

#endif//ARENAI_TRAIN_HOST_MEAN_METRIC_H

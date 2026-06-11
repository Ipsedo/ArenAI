//
// Created by samuel on 11/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_STD_METRIC_H
#define ARENAI_TRAIN_HOST_STD_METRIC_H

#include <arenai_train/metric.h>

#include "./mean_metric.h"

class StdMetric : public AbstractMetric {
public:
    StdMetric(const std::string &name, int window_size);

protected:
    float to_stored_value(float value) override;
    float compute_metric_impl(const std::vector<float> &curr_values) override;

private:
    MeanMetric mean_metric;
};

#endif//ARENAI_TRAIN_HOST_STD_METRIC_H

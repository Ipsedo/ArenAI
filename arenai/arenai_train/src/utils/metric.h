//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_METRIC_H
#define ARENAI_TRAIN_HOST_METRIC_H

#include <string>
#include <vector>

#include <torch/torch.h>

class Metric {
public:
    explicit Metric(
        const std::string &name, int window_size, int precision = 5, bool scientific = false);

    float last_value() const;
    float average_value();

    void add(float value);

    std::string to_string();

private:
    std::string name;
    int window_size;
    std::vector<float> values;

    bool float_display_scientific;
    int float_display_precision;
};

#endif//ARENAI_TRAIN_HOST_METRIC_H

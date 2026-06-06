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
        const std::string &name, int window_size, int precision = 4, bool scientific = false,
        bool print_std = false);

    float last_value() const;
    float mean();
    float std();

    void add(float value);

    std::string to_string();

    static std::string metrics_to_string(const std::vector<std::shared_ptr<Metric>> &metrics);

private:
    std::string name;
    int window_size;
    std::vector<float> values;

    bool float_display_scientific;
    int float_display_precision;

    bool print_std;
};

#endif//ARENAI_TRAIN_HOST_METRIC_H

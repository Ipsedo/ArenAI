//
// Created by samuel on 12/10/2025.
//

#ifndef PHYVR_TRAIN_HOST_METRIC_H
#define PHYVR_TRAIN_HOST_METRIC_H

#include <string>
#include <vector>

#include <torch/torch.h>

class Metric {
public:
    explicit Metric(const std::string &name, int window_size);

    float last_value() const;
    float average_value();

    void add(float value);

    std::string to_string();

private:
    std::string name;
    int window_size;
    std::vector<float> values;
};

#endif//PHYVR_TRAIN_HOST_METRIC_H

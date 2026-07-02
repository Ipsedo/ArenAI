//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_METRIC_H
#define ARENAI_TRAIN_HOST_METRIC_H

#include <string>
#include <vector>

#include <torch/torch.h>

namespace arenai::train {

    class AbstractMetric {
    public:
        virtual ~AbstractMetric() = default;

        explicit AbstractMetric(
            const std::string &name, int window_size, int precision = 4, bool scientific = false);

        void add(float value);
        float last_value() const;

        float compute_metric();

        std::string get_name() const;
        std::string to_string();

        static std::string
        metrics_to_string(const std::vector<std::shared_ptr<AbstractMetric>> &metrics);

    protected:
        virtual float to_stored_value(float value);

        virtual float compute_metric_impl(const std::vector<float> &curr_values) = 0;

    private:
        std::string name;
        int window_size;
        std::vector<float> values;

        bool float_display_scientific;
        int float_display_precision;
    };

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_METRIC_H

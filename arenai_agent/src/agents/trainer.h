//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_AGENTER_H
#define ARENAI_AGENTER_H

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "../metrics/metric.h"

namespace arenai::agent {

    class AbstractTrainer {
    public:
        virtual ~AbstractTrainer() = default;

        virtual void step() = 0;

        virtual std::vector<std::shared_ptr<AbstractMetric>> get_metrics() = 0;
        virtual std::map<std::string, std::string> get_config() const = 0;

        virtual void save(const std::filesystem::path &output_folder) = 0;

        virtual int count_parameters() = 0;

    protected:
        static int count_parameters_impl(const std::vector<torch::Tensor> &params);
    };

}// namespace arenai::agent

#endif//ARENAI_AGENTER_H

//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_TRAINER_H
#define ARENAI_TRAINER_H

#include <filesystem>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "../metrics/metric.h"

namespace arenai::train {

    class AbstractTrainer {
    public:
        virtual ~AbstractTrainer() = default;

        // Called after each environment step; each algorithm decides on its own
        // cadence (warmup / train_every for SAC, full rollout for PPO).
        virtual void step() = 0;

        virtual std::vector<std::shared_ptr<AbstractMetric>> get_metrics() = 0;

        virtual void save(const std::filesystem::path &output_folder) = 0;

        virtual int count_parameters() = 0;

    protected:
        static int count_parameters_impl(const std::vector<torch::Tensor> &params);
    };

}// namespace arenai::train

#endif//ARENAI_TRAINER_H

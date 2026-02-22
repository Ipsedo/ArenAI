//
// Created by samuel on 21/01/2026.
//

#ifndef ARENAI_TRAIN_HOST_AGENT_H
#define ARENAI_TRAIN_HOST_AGENT_H

#include <filesystem>
#include <vector>

#include <torch/torch.h>

#include "../utils/metric.h"
#include "../utils/replay_buffer.h"

struct agent_response {
    torch::Tensor continuous_action;
    torch::Tensor continuous_log_proba;

    torch::Tensor discrete_action;
    torch::Tensor discrete_log_proba;
};

class AbstractAgent {
public:
    virtual ~AbstractAgent() = default;

    virtual void
    train(const std::unique_ptr<ReplayBuffer> &replay_buffer, int epochs, int batch_size) = 0;
    virtual agent_response act(const torch::Tensor &vision, const torch::Tensor &sensors) = 0;

    virtual void set_train(bool train) = 0;

    virtual std::vector<std::shared_ptr<Metric>> get_metrics() = 0;

    virtual void save(const std::filesystem::path &output_folder) = 0;
    virtual void to(torch::Device device) = 0;

    virtual int count_parameters() = 0;

protected:
    static int count_parameters_impl(const std::vector<torch::Tensor> &params);
};

#endif//ARENAI_TRAIN_HOST_AGENT_H

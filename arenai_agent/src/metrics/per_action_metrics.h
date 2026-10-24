//
// Created by samuel on 24/10/2026.
//

#ifndef ARENAI_AGENT_HOST_PER_ACTION_METRICS_H
#define ARENAI_AGENT_HOST_PER_ACTION_METRICS_H

#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "./metric.h"

namespace arenai::agent {

    // one metric per action dimension, named "<prefix>_<action name>", absent from the
    // progress bar (CSV only)
    std::vector<std::shared_ptr<AbstractMetric>> make_per_action_metrics(
        const std::string &prefix, const std::vector<std::string> &action_names, int nb_actions,
        int window_size);

    // adds values[i] (a 1-D tensor, one entry per action dimension) to metrics[i]
    void add_per_action_values(
        const std::vector<std::shared_ptr<AbstractMetric>> &metrics, const torch::Tensor &values);

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_PER_ACTION_METRICS_H

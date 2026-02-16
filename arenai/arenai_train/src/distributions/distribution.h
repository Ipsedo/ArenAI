//
// Created by samuel on 16/02/2026.
//

#ifndef ARENAI_TRAIN_HOST_DISTRIBUTION_H
#define ARENAI_TRAIN_HOST_DISTRIBUTION_H

#include <torch/torch.h>

#include "../networks/actor.h"

class TorchDistribution {
public:
    virtual ~TorchDistribution() = default;

    virtual torch::Tensor sample() = 0;
    virtual torch::Tensor log_proba(const torch::Tensor &x) = 0;
};

class TorchDistributionFactory {
public:
    virtual ~TorchDistributionFactory() = default;

    virtual std::shared_ptr<TorchDistribution> get_distribution(actor_response input) = 0;
    virtual float get_target_entropy() = 0;
};

#endif//ARENAI_TRAIN_HOST_DISTRIBUTION_H

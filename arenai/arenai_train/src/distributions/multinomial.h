//
// Created by samuel on 22/02/2026.
//

#ifndef ARENAI_TRAIN_HOST_MULTINOMIAL_H
#define ARENAI_TRAIN_HOST_MULTINOMIAL_H

#include <torch/torch.h>

torch::Tensor gumbel_hard(const torch::Tensor &probabilities);
torch::Tensor
multinomial_log_proba(const torch::Tensor &action, const torch::Tensor &probabilities);
torch::Tensor multinomial_entropy(const torch::Tensor &probabilities);

float multinomial_target_entropy();

#endif//ARENAI_TRAIN_HOST_MULTINOMIAL_H

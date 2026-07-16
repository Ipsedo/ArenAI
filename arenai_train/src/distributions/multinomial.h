//
// Created by samuel on 22/02/2026.
//

#ifndef ARENAI_TRAIN_HOST_MULTINOMIAL_H
#define ARENAI_TRAIN_HOST_MULTINOMIAL_H

#include <torch/torch.h>

namespace arenai::train {

    torch::Tensor multinomial_sample(const torch::Tensor &probabilities);
    torch::Tensor multinomial_entropy(const torch::Tensor &probabilities);

    float multinomial_maximum_entropy(const int &nb_actions);

    float multinomial_target_entropy(const float &shoot_probability);

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_MULTINOMIAL_H

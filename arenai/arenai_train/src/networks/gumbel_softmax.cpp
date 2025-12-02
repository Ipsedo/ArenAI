//
// Created by samuel on 02/12/2025.
//

#include "./gumbel_softmax.h"

GumbelSoftmax::GumbelSoftmax(const int &dim, const float &tau, const float &epsilon)
    : dim(dim), tau(tau), epsilon(epsilon) {}

torch::Tensor GumbelSoftmax::forward(const torch::Tensor &x) const {
    const auto u = torch::rand_like(x, torch::TensorOptions().device(x.device()));
    const auto gumbel_noise = -torch::log(-torch::log(u + epsilon) + epsilon);
    const auto y = (x + gumbel_noise) / tau;
    return torch::softmax(y, dim);
}

//
// Created by samuel on 10/02/2026.
//

#ifndef ARENAI_AGENT_HOST_GAUSSIAN_TANH_H
#define ARENAI_AGENT_HOST_GAUSSIAN_TANH_H

#include <torch/torch.h>

namespace arenai::agent {

    // squashed normal: a = tanh(u), u ~ N(mu, sigma). Every function works on the
    // pre-tanh u, the action tanh(u) is never inverted

    // {tanh(u), u}, reparameterized
    std::pair<torch::Tensor, torch::Tensor>
    gaussian_tanh_sample(const torch::Tensor &mu, const torch::Tensor &sigma);

    torch::Tensor gaussian_tanh_mode(const torch::Tensor &mu);

    torch::Tensor gaussian_tanh_log_pdf(
        const torch::Tensor &u, const torch::Tensor &mu, const torch::Tensor &sigma);

    // one-sample reparameterized Monte-Carlo estimate of the per-dimension entropy,
    // differentiable in mu and sigma
    torch::Tensor gaussian_tanh_entropy(const torch::Tensor &mu, const torch::Tensor &sigma);

    // KL(p || q) per dimension: tanh is a bijection, the KL is the one of the pre-tanh normals
    torch::Tensor gaussian_tanh_kl(
        const torch::Tensor &mu_p, const torch::Tensor &sigma_p, const torch::Tensor &mu_q,
        const torch::Tensor &sigma_q);

    float gaussian_tanh_target_entropy(int nb_actions, float target_sigma, int nb_samples = 100000);

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_GAUSSIAN_TANH_H

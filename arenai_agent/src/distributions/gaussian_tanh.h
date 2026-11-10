#ifndef ARENAI_AGENT_HOST_GAUSSIAN_TANH_H
#define ARENAI_AGENT_HOST_GAUSSIAN_TANH_H

#include <torch/torch.h>

namespace arenai::agent {

    std::pair<torch::Tensor, torch::Tensor>
    gaussian_tanh_sample(const torch::Tensor &mu, const torch::Tensor &sigma);

    torch::Tensor gaussian_tanh_mode(const torch::Tensor &mu);

    torch::Tensor gaussian_tanh_log_pdf(
        const torch::Tensor &u, const torch::Tensor &mu, const torch::Tensor &sigma);

    torch::Tensor gaussian_tanh_entropy(const torch::Tensor &mu, const torch::Tensor &sigma);

    torch::Tensor gaussian_tanh_kl(
        const torch::Tensor &mu_p, const torch::Tensor &sigma_p, const torch::Tensor &mu_q,
        const torch::Tensor &sigma_q);

    float gaussian_tanh_target_entropy(int nb_actions, float target_sigma, int nb_samples = 100000);

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_GAUSSIAN_TANH_H

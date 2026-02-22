//
// Created by samuel on 22/02/2026.
//

#include "./multinomial.h"

torch::Tensor multinomial_sample(const torch::Tensor &probabilities) {
    const auto idx = torch::multinomial(probabilities, 1, true);
    const auto one_hot = torch::zeros_like(probabilities).scatter_(1, idx, 1.0);
    return (one_hot - probabilities).detach() + probabilities;
}

torch::Tensor
multinomial_log_proba(const torch::Tensor &action, const torch::Tensor &probabilities) {
    const auto idx = action.argmax(-1).to(torch::kLong).unsqueeze(-1);
    const auto p = torch::gather(probabilities, -1, idx);
    return torch::log(p);
}

torch::Tensor multinomial_entropy(const torch::Tensor &probabilities) {
    return -torch::sum(probabilities * torch::log(probabilities), -1, true);
}

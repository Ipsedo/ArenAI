//
// Created by samuel on 12/10/2025.
//

#include "./target_update.h"

void hard_update(
    const std::shared_ptr<torch::nn::Module> &to, const std::shared_ptr<torch::nn::Module> &from) {
    for (auto n_p: from->named_parameters()) {
        const auto &name = n_p.key();
        const auto &param = n_p.value();

        to->named_parameters()[name].data().copy_(param.data());
    }
}

void soft_update(
    const std::shared_ptr<torch::nn::Module> &to, const std::shared_ptr<torch::nn::Module> &from,
    const float tau) {
    for (auto n_p: from->named_parameters()) {
        const auto &name = n_p.key();
        const auto &param = n_p.value();

        to->named_parameters()[name].data().copy_(
            tau * param.data() + (1.0 - tau) * to->named_parameters()[name].data());
    }
}

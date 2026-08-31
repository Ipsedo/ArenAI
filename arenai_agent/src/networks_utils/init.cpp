//
// Created by samuel on 24/10/2025.
//

#include "./init.h"

#include "../networks/constants.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    void init_hidden_weights(torch::nn::Module &module) {
        if (auto *lin = module.as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(lin->weight, std::sqrt(2.f));
            if (lin->options.bias()) torch::nn::init::zeros_(lin->bias);
        } else if (auto *ln = module.as<torch::nn::LayerNorm>()) {
            if (ln->options.elementwise_affine()) {
                torch::nn::init::ones_(ln->weight);
                torch::nn::init::zeros_(ln->bias);
            }
        } else if (auto *gn = module.as<torch::nn::GroupNorm>()) {
            if (gn->options.affine()) {
                torch::nn::init::ones_(gn->weight);
                torch::nn::init::zeros_(gn->bias);
            }
        } else if (auto *conv = module.as<torch::nn::Conv2d>()) {
            torch::nn::init::orthogonal_(conv->weight, std::sqrt(2.f));
            if (conv->options.bias()) torch::nn::init::zeros_(conv->bias);
        }
    }

    void init_mu_output_weights(torch::nn::Module &module) {
        if (auto *lin = module.as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(lin->weight, 0.01f);
            if (lin->options.bias()) torch::nn::init::zeros_(lin->bias);
        }
    }

    void init_sigma_output_weights(torch::nn::Module &module, const float wanted_sigma) {
        const float min_log_sigma = std::log(SIGMA_MIN);
        const float max_log_sigma = std::log(SIGMA_MAX);

        const auto initial_sigma_sigmoid =
            (std::log(wanted_sigma) - min_log_sigma) / (max_log_sigma - min_log_sigma);
        const auto initial_sigma_logit =
            std::log(initial_sigma_sigmoid / (1.f - initial_sigma_sigmoid));

        if (auto *lin = module.as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(lin->weight, 0.01f);
            if (lin->options.bias()) torch::nn::init::constant_(lin->bias, initial_sigma_logit);
        }
    }

    void
    init_discrete_output_weights(torch::nn::Module &module, const float initial_fire_probability) {
        if (auto *lin = module.as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(lin->weight, 0.01f);

            if (lin->options.bias()) {
                torch::nn::init::zeros_(lin->bias);

                lin->bias.data().index_fill_(
                    0, torch::tensor({0}), std::log(initial_fire_probability));
                lin->bias.data().index_fill_(
                    0, torch::tensor({1}), std::log(1.f - initial_fire_probability));
            }
        }
    }

    void init_value_output_weights(torch::nn::Module &module) {
        if (auto *lin = module.as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(lin->weight, 1.f);
            if (lin->options.bias()) torch::nn::init::zeros_(lin->bias);
        }
    }

}// namespace arenai::agent

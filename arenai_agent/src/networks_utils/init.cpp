//
// Created by samuel on 24/10/2025.
//

#include "./init.h"

#include <algorithm>
#include <cmath>

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

    void init_liquid_weights(torch::nn::Module &module) {
        if (const auto *lin = module.as<torch::nn::Linear>()) {
            torch::nn::init::normal_(lin->weight, 0.f, 1e-2f);
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

    void init_concentration_output_weights(torch::nn::Module &module, const float wanted_sigma) {
        // Beta on [-1, 1]: var = 4 μ(1-μ) / (κ+1), so at μ = 0.5 a wanted action
        // std σ maps to κ = 1/σ² - 1
        const auto wanted_concentration = std::clamp(
            1.f / (wanted_sigma * wanted_sigma) - 1.f, CONCENTRATION_MIN, CONCENTRATION_MAX);

        const float min_log_excess = std::log(CONCENTRATION_MIN - 2.f);
        const float max_log_excess = std::log(CONCENTRATION_MAX - 2.f);

        const auto initial_sigmoid = (std::log(wanted_concentration - 2.f) - min_log_excess)
                                     / (max_log_excess - min_log_excess);
        const auto initial_logit = std::log(initial_sigmoid / (1.f - initial_sigmoid));

        if (auto *lin = module.as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(lin->weight, 0.01f);
            if (lin->options.bias()) torch::nn::init::constant_(lin->bias, initial_logit);
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

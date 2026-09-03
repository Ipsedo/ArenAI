//
// Created by samuel on 24/10/2025.
//

#include "./init.h"

#include "../distributions/beta_law.h"
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

    void init_kappa_output_weights(torch::nn::Module &module, const float init_kappa) {
        // must mirror the kappa head's RangeSigmoidOutput(MIN_KAPPA_SIGMOID, 1) bounds
        constexpr float min_kappa = MIN_KAPPA_SIGMOID;
        constexpr float max_kappa = 1.f;

        const auto initial_kappa_sigmoid = (init_kappa - min_kappa) / (max_kappa - min_kappa);
        const auto initial_kappa_logit =
            std::log(initial_kappa_sigmoid / (1.f - initial_kappa_sigmoid));

        if (auto *lin = module.as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(lin->weight, 0.01f);
            if (lin->options.bias()) torch::nn::init::constant_(lin->bias, initial_kappa_logit);
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

//
// Created by samuel on 24/10/2025.
//

#include "./init.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    void init_hidden_weights(torch::nn::Module &module) {
        if (auto *lin = dynamic_cast<torch::nn::LinearImpl *>(&module)) {
            torch::nn::init::kaiming_normal_(lin->weight, 0., torch::kFanIn, torch::kReLU);
            if (lin->options.bias()) torch::nn::init::zeros_(lin->bias);
        } else if (auto *ln = dynamic_cast<torch::nn::LayerNormImpl *>(&module)) {
            if (ln->options.elementwise_affine()) {
                torch::nn::init::ones_(ln->weight);
                torch::nn::init::zeros_(ln->bias);
            }
        } else if (auto *gn = dynamic_cast<torch::nn::GroupNormImpl *>(&module)) {
            if (gn->options.affine()) {
                torch::nn::init::ones_(gn->weight);
                torch::nn::init::zeros_(gn->bias);
            }
        } else if (auto *conv = dynamic_cast<torch::nn::Conv2dImpl *>(&module)) {
            torch::nn::init::kaiming_normal_(conv->weight, 0., torch::kFanIn, torch::kReLU);
            if (conv->options.bias()) torch::nn::init::zeros_(conv->bias);
        }
    }

    void init_mu_output_weights(torch::nn::Module &module) {
        if (auto *lin = dynamic_cast<torch::nn::LinearImpl *>(&module)) {
            torch::nn::init::uniform_(lin->weight, -1e-3f, 1e-3f);
            if (lin->options.bias()) torch::nn::init::zeros_(lin->bias);
        }
    }

    void init_sigma_output_weights(torch::nn::Module &module) {
        if (auto *lin = dynamic_cast<torch::nn::LinearImpl *>(&module)) {
            torch::nn::init::uniform_(lin->weight, -1e-3f, 1e-3f);
            if (lin->options.bias()) torch::nn::init::constant_(lin->bias, std::log(0.1f));
        }
    }

    void init_discrete_output_weights(torch::nn::Module &module) {
        if (auto *lin = dynamic_cast<torch::nn::LinearImpl *>(&module)) {
            torch::nn::init::uniform_(lin->weight, -1e-3f, 1e-3f);
            if (lin->options.bias()) torch::nn::init::zeros_(lin->bias);
        }
    }

    void init_value_output_weights(torch::nn::Module &module) {
        if (auto *lin = dynamic_cast<torch::nn::LinearImpl *>(&module)) {
            torch::nn::init::uniform_(lin->weight, -3e-3f, 3e-3f);
            if (lin->options.bias()) torch::nn::init::zeros_(lin->bias);
        }
    }

}// namespace arenai::agent

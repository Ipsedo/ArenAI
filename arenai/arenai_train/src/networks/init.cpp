//
// Created by samuel on 24/10/2025.
//

#include "./init.h"

void init_weights(torch::nn::Module &module) {
    if (auto *lin = dynamic_cast<torch::nn::LinearImpl *>(&module)) {
        torch::nn::init::kaiming_normal_(lin->weight, 0.0, torch::kFanIn, torch::kReLU);
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
        torch::nn::init::kaiming_normal_(conv->weight, 0.0, torch::kFanIn, torch::kReLU);
        if (conv->options.bias()) torch::nn::init::zeros_(conv->bias);
    }
}

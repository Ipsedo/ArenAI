//
// Created by samuel on 03/10/2025.
//

#ifndef PHYVR_TRAIN_HOST_SAVER_H
#define PHYVR_TRAIN_HOST_SAVER_H

#include <filesystem>

#include <torch/torch.h>

void export_state_dict_neutral(torch::nn::Module m, const std::filesystem::path &outdir);

#endif// PHYVR_TRAIN_HOST_SAVER_H

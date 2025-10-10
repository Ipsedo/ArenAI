//
// Created by samuel on 03/10/2025.
//

#ifndef PHYVR_TRAIN_HOST_SAVER_H
#define PHYVR_TRAIN_HOST_SAVER_H

#include <filesystem>

#include <torch/torch.h>

void export_state_dict_neutral(torch::nn::Module m, const std::filesystem::path &outdir);

void save_png_rgb(
    const std::vector<std::vector<std::vector<uint8_t>>> &image, const std::string &filename);

#endif// PHYVR_TRAIN_HOST_SAVER_H

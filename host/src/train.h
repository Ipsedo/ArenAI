//
// Created by samuel on 03/10/2025.
//

#ifndef PHYVR_TRAIN_HOST_TRAIN_H
#define PHYVR_TRAIN_HOST_TRAIN_H

#include <filesystem>

#include "./networks/agent.h"
#include "./networks/entropy.h"

void train_main(
    const std::filesystem::path &output_folder, const std::filesystem::path &android_assets_path);

#endif// PHYVR_TRAIN_HOST_TRAIN_H

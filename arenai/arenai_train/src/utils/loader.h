//
// Created by samuel on 11/03/2026.
//

#ifndef ARENAI_TRAIN_HOST_LOADER_H
#define ARENAI_TRAIN_HOST_LOADER_H

#include <filesystem>

#include <torch/torch.h>

template<typename T>
void load_torch(
    const std::string &input_folder_path, const T &to_load, const std::string &file_name) {

    const std::filesystem::path path(input_folder_path);

    if (!std::filesystem::exists(path))
        throw std::runtime_error("Could not find " + input_folder_path);

    const auto file = path / file_name;
    torch::serialize::InputArchive archive;
    archive.load_from(file, torch::kCPU);
    to_load->load(archive);
}

#endif//ARENAI_TRAIN_HOST_LOADER_H

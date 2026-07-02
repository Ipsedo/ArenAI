//
// Created by samuel on 11/03/2026.
//

#ifndef ARENAI_TRAIN_HOST_LOADER_H
#define ARENAI_TRAIN_HOST_LOADER_H

#include <filesystem>

#include <torch/torch.h>

namespace arenai::train {

    template<typename T>
    void load_torch(
        const std::filesystem::path &input_folder_path, const T &to_load,
        const std::filesystem::path &file_name) {

        if (!std::filesystem::exists(input_folder_path))
            throw std::runtime_error("Could not find " + input_folder_path.string());

        const auto file = input_folder_path / file_name;
        torch::serialize::InputArchive archive;
        archive.load_from(file.string(), torch::kCPU);
        to_load->load(archive);
    }

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_LOADER_H

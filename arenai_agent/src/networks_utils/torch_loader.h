//
// Created by samuel on 11/03/2026.
//

#ifndef ARENAI_AGENT_HOST_LOADER_H
#define ARENAI_AGENT_HOST_LOADER_H

#include <filesystem>

#include <torch/torch.h>

#include <arenai_utils/exceptions.h>

namespace arenai::agent {

    template<typename T>
    void load_torch(
        const std::filesystem::path &input_folder_path, const T &to_load,
        const std::filesystem::path &file_name) {

        const auto file = input_folder_path / file_name;

        if (!std::filesystem::exists(file)) throw utils::FileDoesNotExistException(file);

        try {
            torch::serialize::InputArchive archive;
            archive.load_from(file.string(), torch::kCPU);
            to_load->load(archive);
        } catch (const std::exception &e) {
            std::throw_with_nested(utils::ModelLoadException(file));
        }
    }

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_LOADER_H

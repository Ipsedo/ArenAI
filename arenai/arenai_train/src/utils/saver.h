//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_SAVER_H
#define ARENAI_TRAIN_HOST_SAVER_H

#include <filesystem>
#include <memory>

#include <torch/torch.h>

#include "../agents/agent.h"

void export_state_dict_neutral(
    const std::shared_ptr<torch::nn::Module> &m, const std::filesystem::path &outdir);

void save_png_rgb(
    const std::vector<std::vector<std::vector<uint8_t>>> &image, const std::string &filename);

template<typename T>
void save_torch(
    const std::string &output_folder_path, const T &to_save, const std::string &file_name) {
    const std::filesystem::path path(output_folder_path);

    if (!std::filesystem::exists(path))
        throw std::runtime_error("Could not find " + output_folder_path);

    const auto file = path / file_name;
    torch::serialize::OutputArchive archive;
    to_save->save(archive);
    archive.save_to(file);
}

class Saver {
public:
    Saver(
        const std::shared_ptr<AbstractAgent> &agent, const std::filesystem::path &output_path,
        int save_every);

    void attempt_save();

private:
    std::shared_ptr<AbstractAgent> agent;
    long curr_step;
    int save_every;
    std::filesystem::path output_path;
};

#endif// ARENAI_TRAIN_HOST_SAVER_H

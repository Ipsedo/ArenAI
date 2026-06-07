//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_SAVER_H
#define ARENAI_TRAIN_HOST_SAVER_H

#include <filesystem>
#include <memory>

#include <torch/torch.h>

#include <arenai_train/agent.h>

void export_state_dict_neutral(
    const std::shared_ptr<torch::nn::Module> &m, const std::filesystem::path &outdir);

template<typename T>
void save_torch(
    const std::filesystem::path &output_folder_path, const T &to_save,
    const std::filesystem::path &file_name) {

    if (!std::filesystem::exists(output_folder_path))
        throw std::runtime_error("Could not find " + output_folder_path.string());

    const auto file = output_folder_path / file_name;
    torch::serialize::OutputArchive archive;
    to_save->save(archive);
    archive.save_to(file);
}

class MetricCsvSaver {
public:
    MetricCsvSaver(
        const std::filesystem::path &output_folder,
        const std::vector<std::shared_ptr<Metric>> &metrics, int save_every);

    void attempt_append_to_csv();

private:
    std::filesystem::path csv_file_path;
    std::vector<std::shared_ptr<Metric>> metrics;

    std::string sep;

    int save_every;
    long index;
};

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

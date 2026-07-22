//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_SAVER_H
#define ARENAI_TRAIN_HOST_SAVER_H

#include <filesystem>
#include <memory>

#include <torch/torch.h>

#include "../agents/trainer.h"

namespace arenai::train {

    template<typename T>
    void save_torch(
        const std::filesystem::path &output_folder_path, const T &to_save,
        const std::filesystem::path &file_name) {

        if (!std::filesystem::exists(output_folder_path))
            throw std::runtime_error("Could not find " + output_folder_path.string());

        const auto file = output_folder_path / file_name;
        torch::serialize::OutputArchive archive;
        to_save->save(archive);
        archive.save_to(file.string());
    }

    class AgentSaver {
    public:
        AgentSaver(
            const std::shared_ptr<AbstractTrainer> &trainer,
            const std::filesystem::path &output_path, int save_every);

        void attempt_save();

    private:
        std::shared_ptr<AbstractTrainer> trainer;
        long curr_step;
        int save_every;
        std::filesystem::path output_path;
    };

}// namespace arenai::train

#endif// ARENAI_TRAIN_HOST_SAVER_H

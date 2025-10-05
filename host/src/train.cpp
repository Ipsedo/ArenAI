//
// Created by samuel on 03/10/2025.
//

#include "./train.h"

#include "./train_environment.h"
#include "./utils/linux_file_reader.h"
#include "./utils/replay_buffer.h"

void train(const std::filesystem::path &output_folder) {

  auto env = std::make_unique<TrainTankEnvironment>(4, 4);
}

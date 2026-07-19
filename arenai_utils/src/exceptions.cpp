//
// Created by samuel on 19/07/2026.
//

#include <arenai_utils/exceptions.h>

using namespace arenai;
using namespace arenai::utils;

FileDoesNotExistException::FileDoesNotExistException(const std::filesystem::path &file_path)
    : std::runtime_error("File does not exists : " + file_path.string()), file_path(file_path) {}

std::filesystem::path FileDoesNotExistException::missing_file() { return file_path; }

/*
 * Wrong state dict file
 */

ModelLoadException::ModelLoadException(const std::filesystem::path &state_dict_file_path)
    : std::runtime_error("Wrong state dict file " + state_dict_file_path.string()),
      state_dict_file_path(state_dict_file_path) {}

std::filesystem::path ModelLoadException::wrong_state_dict_file() { return state_dict_file_path; }

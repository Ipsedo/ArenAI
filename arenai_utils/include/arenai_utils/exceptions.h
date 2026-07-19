//
// Created by samuel on 19/07/2026.
//

#ifndef ARENAI_EXCEPTIONS_H
#define ARENAI_EXCEPTIONS_H

#include <filesystem>

namespace arenai::utils {
    class FileDoesNotExistException : public std::runtime_error {
    public:
        explicit FileDoesNotExistException(const std::filesystem::path &file_path);

        std::filesystem::path missing_file();

    private:
        std::filesystem::path file_path;
    };

    class ModelLoadException : public std::runtime_error {
    public:
        explicit ModelLoadException(const std::filesystem::path &state_dict_file_path);

        std::filesystem::path wrong_state_dict_file();

    private:
        std::filesystem::path state_dict_file_path;
    };

}// namespace arenai::utils

#endif//ARENAI_EXCEPTIONS_H

//
// Created by samuel on 01/07/2026.
//

#ifndef ARENAI_CORE_TESTS_LOCAL_FILE_READER_H
#define ARENAI_CORE_TESTS_LOCAL_FILE_READER_H

#include <filesystem>

#include <arenai_utils/file_reader.h>

class LocalAssetFileReader final : public arenai::utils::AbstractResourceFileReader {
public:
    explicit LocalAssetFileReader(const std::filesystem::path &path_to_assets);

    std::string read_text(const std::filesystem::path &file_path) override;

    arenai::utils::ImageChannels read_png(const std::filesystem::path &png_file_path) override;

private:
    std::filesystem::path path_to_assets;
};

#endif// ARENAI_CORE_TESTS_LOCAL_FILE_READER_H

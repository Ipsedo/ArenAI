//
// Created by samuel on 01/07/2026.
//

#ifndef ARENAI_MODEL_TESTS_LOCAL_FILE_READER_H
#define ARENAI_MODEL_TESTS_LOCAL_FILE_READER_H

#include <filesystem>

#include <arenai_utils/file_reader.h>

class LocalAssetFileReader final : public AbstractFileReader {
public:
    explicit LocalAssetFileReader(const std::filesystem::path &path_to_assets);

    std::string read_text(const std::filesystem::path &file_path) override;

    ImageChannels read_png(const std::filesystem::path &png_file_path) override;

private:
    std::filesystem::path path_to_assets;
};

#endif// ARENAI_MODEL_TESTS_LOCAL_FILE_READER_H

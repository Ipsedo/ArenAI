//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_LINUX_FILE_READER_H
#define ARENAI_TRAIN_HOST_LINUX_FILE_READER_H

#include <filesystem>

#include <arenai_utils/file_reader.h>

namespace arenai::train {

    class DesktopAssetFileReader final : public utils::AbstractResourceFileReader {
    public:
        explicit DesktopAssetFileReader(const std::filesystem::path &path_to_resources);

        std::string read_text(const std::filesystem::path &file_path) override;

        utils::ImageChannels read_png(const std::filesystem::path &png_file_path) override;

    private:
        std::filesystem::path path_to_resources;
    };

}// namespace arenai::train

#endif// ARENAI_TRAIN_HOST_LINUX_FILE_READER_H

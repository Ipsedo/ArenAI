//
// Created by samuel on 03/10/2025.
//

#include "linux_file_reader.h"

#include <fstream>

LinuxAndroidAssetFileReader::LinuxAndroidAssetFileReader(
    const std::filesystem::path &path_to_assets)
    : path_to_assets(path_to_assets) {}

std::string LinuxAndroidAssetFileReader::read_text(const std::string &file_name) {
    std::ifstream t(path_to_assets / file_name);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

img_rgb LinuxAndroidAssetFileReader::read_png(const std::string &png_file_path) {}
